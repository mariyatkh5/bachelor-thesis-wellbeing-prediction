# main.py
import os
import argparse
import numpy as np
import pandas as pd

from paths import (
    LABELS1_CSV, LABELS1_PAPER_CSV, LABELS2_CSV, LABELS3_CSV, LABELS8_CSV,
    OUTPUT1_DIR, OUTPUT2_DIR, OUTPUT3_DIR, OUTPUT8_DIR,
    D1_SIGNALS, D2_SIGNALS, D3_SIGNALS, D8_SIGNALS,
)

from loaders import load_d1_signal, load_d2_signal, load_d3_signal, load_d8_signal
from utils import zscore_window

# ---------------- Windowing defaults ----------------
DEF_INTERVAL = 72.0
DEF_STEP = 20.0
DEF_MINPTS = 71  # ~ just under full window at 4 Hz

COMBO_SR = 128.0  # must match loaders


# ---------------- Save Helpers ----------------------

def finalize_meta(meta_in: pd.DataFrame, ensure_sr: float = None) -> pd.DataFrame:
    """Ensure meta has seq/row ids and optional sampling_rate."""
    m = meta_in.copy()
    if "row_id" not in m.columns:
        m.insert(0, "row_id", np.arange(len(m), dtype=np.int64))
    if "seq_id" not in m.columns:
        m.insert(0, "seq_id", np.arange(len(m), dtype=np.int64))
    if "sampling_rate" not in m.columns and ensure_sr is not None:
        m["sampling_rate"] = int(round(ensure_sr))
    return m


def save_npz_meta(out_dir: str, base: str, X: np.ndarray, y, y_type: str, meta: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, base + ".npz"), X=X, y=y, y_type=y_type)
    meta.to_csv(os.path.join(out_dir, base + "_meta.csv"), index=False)
    print(f"[SAVE] {os.path.join(out_dir, base+'.npz')} (+ {base}_meta.csv)")


def save_all_variants(out_dir: str, prefix: str, X_none: np.ndarray, X_filt: np.ndarray,
                      meta_sel: pd.DataFrame, y, y_type: str):
    """Save raw/filtered and window-zscored variants."""
    prefix = prefix.lower()
    meta = finalize_meta(meta_sel)
    save_npz_meta(out_dir, f"{prefix}_none", X_none, y, y_type, meta)
    save_npz_meta(out_dir, f"{prefix}_z_window", zscore_window(X_none), y, y_type, meta)
    save_npz_meta(out_dir, f"{prefix}_filtered_none", X_filt, y, y_type, meta)
    save_npz_meta(out_dir, f"{prefix}_filtered_z_window", zscore_window(X_filt), y, y_type, meta)


# ---------------- Label Join & y typing ---------------

def _norm_for_merge(df: pd.DataFrame, cols):
    """Lower/strip selected columns for stable joins."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.lower()
    return out


def join_labels(meta: pd.DataFrame, labels: pd.DataFrame, keys, label_col: str) -> pd.DataFrame:
    """
    Merge meta with labels on given keys (many-to-one).
    If labels duplicate on keys, aggregate deterministically.
    Returns meta+label with 'orig_idx' for array slicing.
    """
    if meta.empty:
        return meta
    keys_use = [k for k in keys if (k in meta.columns and k in labels.columns)]
    if not keys_use:
        print(f"[WARN] Label join: no common keys in meta/labels (asked: {keys})")
        return pd.DataFrame()

    m = meta.reset_index().rename(columns={"index": "orig_idx"})
    m = _norm_for_merge(m, keys_use)
    l = _norm_for_merge(labels.copy(), keys_use)

    if label_col not in labels.columns:
        print(f"[WARN] Label join: label column '{label_col}' not found.")
        return pd.DataFrame()

    l = l[keys_use + [label_col]].dropna(subset=[label_col])

    if l.duplicated(subset=keys_use).any():
        if pd.api.types.is_numeric_dtype(l[label_col]):
            l = l.groupby(keys_use, as_index=False)[label_col].mean()
        else:
            def _mode_or_first(s):
                try:
                    m0 = s.mode(dropna=True)
                    return m0.iloc[0] if len(m0) else s.iloc[0]
                except Exception:
                    return s.iloc[0]
            l = l.groupby(keys_use, as_index=False)[label_col].agg(_mode_or_first)

    df = m.merge(l, on=keys_use, how="left", validate="m:1")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    return df


def infer_y(y: np.ndarray):
    """Infer y type: binary, multiclass, or continuous; return (y_array, y_type)."""
    y_num = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    if np.all(~np.isnan(y_num)):
        u = np.unique(y_num.astype(float))
        if set(u).issubset({0.0, 1.0}):
            return y_num.astype(np.int64), "binary"
        if u.size <= 10 and np.allclose(u, np.round(u)):
            return y_num.astype(np.int64), "multiclass"
        return y_num.astype(np.float32), "continuous"
    _, inv = np.unique(y.astype(str), return_inverse=True)
    return inv.astype(np.int64), "multiclass"


# ---------------- Combo-Builder -----------------------

def build_and_save_combo(eda_pack: dict, ecg_pack: dict, out_dir: str, label_col: str):
    """
    Expect packs with 128 Hz arrays:
      eda_pack/ecg_pack: dict(XN128, XF128, meta, y, y_type)
    Build intersection by keys + start/end, then save variants.
    """
    XN_e128, XF_e128, M_e, y_e, y_type_e = eda_pack["XN128"], eda_pack["XF128"], eda_pack["meta"], eda_pack["y"], eda_pack["y_type"]
    XN_d, XF_d, M_d, y_d, y_type_d = ecg_pack["XN128"], ecg_pack["XF128"], ecg_pack["meta"], ecg_pack["y"], ecg_pack["y_type"]

    if len(M_e) == 0 or len(M_d) == 0:
        print("[SKIP] combo: missing meta – no intersection.")
        return

    candidate_keys = ["Participant", "Task", "Speed", "Robots", "Marker", "Stimuli"]
    keys = [k for k in candidate_keys if (k in M_e.columns and k in M_d.columns)]

    # rounded time keys
    for m in (M_e, M_d):
        m["Start_sec_r"] = np.round(m["Start_sec"].astype(float), 3)
        m["End_sec_r"] = np.round(m["End_sec"].astype(float), 3)
    keys = keys + ["Start_sec_r", "End_sec_r"]

    A = M_e.reset_index().rename(columns={"index": "idx_e"})
    B = M_d.reset_index().rename(columns={"index": "idx_d"})
    J = A.merge(B, on=keys, how="inner", suffixes=("_e", "_d"))
    if J.empty:
        print("[WARN] combo: no shared window found.")
        return

    idx_e = J["idx_e"].to_numpy()
    idx_d = J["idx_d"].to_numpy()

    X_e_none_128 = XN_e128[idx_e] if XN_e128.size else np.empty((0, 1, 1), np.float32)
    X_e_filt_128 = XF_e128[idx_e] if XF_e128.size else np.empty((0, 1, 1), np.float32)
    X_d_none = XN_d[idx_d]
    X_d_filt = XF_d[idx_d]

    assert X_e_none_128.shape[1] == X_d_none.shape[1], "Combo length mismatch (none)"
    assert X_e_filt_128.shape[1] == X_d_filt.shape[1], "Combo length mismatch (filtered)"

    # (N, T, 2): [EDA, ECG]
    X_none = np.concatenate([X_e_none_128, X_d_none], axis=2)
    X_filt = np.concatenate([X_e_filt_128, X_d_filt], axis=2)

    # prefer ECG labels when available
    if y_d is not None and len(y_d) == len(M_d):
        y, yt = y_d[idx_d], y_type_d
        meta_combo = M_d.iloc[idx_d].copy()
    else:
        y, yt = y_e[idx_e], y_type_e
        meta_combo = M_e.iloc[idx_e].copy()

    meta_combo["sampling_rate"] = int(COMBO_SR)
    meta_combo = finalize_meta(meta_combo, ensure_sr=COMBO_SR)

    save_all_variants(out_dir, "combo", X_none, X_filt, meta_combo, y, yt)


# ---------------- Runner -------------------------------------------

def run_dataset(tag: str, out_dir: str, labels_csv: str, signals_map: dict,
                loader_fn, join_keys, sigs, args):
    print(f"=== DATASET {tag} ===")
    os.makedirs(out_dir, exist_ok=True)
    lbl = pd.read_csv(labels_csv) if (labels_csv and os.path.isfile(labels_csv)) else None

    packs = {}

    for s in sigs:
        if s not in signals_map:
            print(f"[WARN] D{tag}: unknown signal '{s}'")
            continue

        XN, XF, XN128, XF128, meta = loader_fn(s, args)
        print(f"{s}: windows none={XN.shape}, filtered={XF.shape}, meta={meta.shape}")
        if meta.empty:
            print(f"[WARN] D{tag}: no windows for {s}.")
            continue

        if lbl is not None:
            metaY = join_labels(meta, lbl, keys=join_keys, label_col=args.label_col)
            if metaY.empty:
                print(f"[WARN] D{tag}: label join returned 0 rows for {s}.")
                continue
            idx = metaY["orig_idx"].to_numpy()

            # save singles
            XN_sel, XF_sel = XN[idx], XF[idx]
            y, y_type = infer_y(metaY[args.label_col].to_numpy())
            save_all_variants(out_dir, s.lower(), XN_sel, XF_sel, metaY, y, y_type)

            # stash 128 Hz copies for combo
            packs[s.lower()] = dict(
                XN128=XN128[idx], XF128=XF128[idx], meta=metaY, y=y, y_type=y_type
            )
        else:
            print(f"[WARN] D{tag}: labels.csv missing – singles not saved for {s}.")

    # build combo if both signals exist
    if "eda" in packs and "ecg" in packs:
        build_and_save_combo(packs["eda"], packs["ecg"], out_dir, args.label_col)
    else:
        print(f"[INFO] D{tag}: combo skipped (need both EDA and ECG with labels).")


# ---------------- CLI / Main --------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["1", "2", "3", "8", "all"],
        default="8",
        help="Dataset to process (1=default labels, 1_l=paper labels for D1)",
    )
    parser.add_argument("--participants", nargs="*", type=int, help="Subset of participants")
    parser.add_argument("--tasks", nargs="*", type=int, help="Subset of tasks")
    parser.add_argument("--speeds", nargs="*", type=int)
    parser.add_argument("--robots", nargs="*", type=int)
    parser.add_argument("--signals", type=str, default="ECG,EDA")
    parser.add_argument("--interval", type=float, default=DEF_INTERVAL)
    parser.add_argument("--step", type=float, default=DEF_STEP)
    parser.add_argument("--min_points", type=int, default=DEF_MINPTS)
    parser.add_argument("--auto_min_points", action="store_true", default=False)
    parser.add_argument("--eda_lp_cutoff", type=float, default=1.0)   # EDA pipeline has fixed 5 Hz in filters.py
    parser.add_argument("--ecg_notch_hz", type=float, default=None)   # ECG pipeline without notch
    parser.add_argument("--label_col", type=str, default="Well-being")

    # D1 label control only relevant for --dataset all (or if forced)
    parser.add_argument(
        "--labels_variant",
        choices=["default", "paper"],
        default="default",
        help="Which labels to use for Dataset 1 when --dataset=all",
    )
    parser.add_argument(
        "--labels_csv_override",
        type=str,
        default=None,
        help="Custom labels CSV for Dataset 1 (overrides variant selection)",
    )

    args = parser.parse_args()

    sigs = [s.strip().upper() for s in args.signals.split(",") if s.strip()]

    # Dataset 1 (default labels)
    if args.dataset in {"1", "all"}:
        labels_csv_1 = args.labels_csv_override or LABELS1_CSV
        run_dataset("1", OUTPUT1_DIR, labels_csv_1, D1_SIGNALS, load_d1_signal, ["Participant", "Task"], sigs, args)

    # Dataset 2
    if args.dataset in {"2", "all"}:
        run_dataset("2", OUTPUT2_DIR, LABELS2_CSV, D2_SIGNALS, load_d2_signal, ["Participant", "Speed", "Robots"], sigs, args)

    # Dataset 3
    if args.dataset in {"3", "all"}:
        run_dataset("3", OUTPUT3_DIR, LABELS3_CSV, D3_SIGNALS, load_d3_signal, ["Participant", "Task"], sigs, args)

    # Dataset 8
    if args.dataset in {"8", "all"}:
        run_dataset("8", OUTPUT8_DIR, LABELS8_CSV, D8_SIGNALS, load_d8_signal, ["Participant", "Task", "Stimuli"], sigs, args)

    if args.dataset == "all":
        labels_csv_1 = args.labels_csv_override or (
            LABELS1_PAPER_CSV if args.labels_variant == "paper" else LABELS1_CSV
        )
        run_dataset("1", OUTPUT1_DIR, labels_csv_1, D1_SIGNALS, load_d1_signal, ["Participant", "Task"], sigs, args)


if __name__ == "__main__":
    main()
