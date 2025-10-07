
import os, re, json, argparse, shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Literal
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
    precision_score, recall_score
)


from model import build_cnn_single, build_cnn_mh
from config import get_cnn_config_space

# ==== Paths & Defaults ====
DATA_ROOT    = Path("output")
RESULTS_ROOT = Path("results")
WEIGHTS_DIR  = RESULTS_ROOT / "weights"
ARCH_SINGLE  = "CNN"      # ecg OR eda
ARCH_COMBO   = "CNN_MH"   # multi-head (ecg + eda)

DEF_FOLDS    = 5
DEF_CONFIGS  = 5
BATCH_SIZE_D = 128
NUM_WORKERS  = 0
PATIENCE     = 25
MAX_EPOCHS   = 1024

# ==== Grouping keys to build leakage-safe groups per dataset ====

GROUP_KEYS_MAP = {
    "dataset1": ["Participant", "Task"],
    "dataset2": ["Participant", "Speed", "Robots"],
    "dataset3": ["Participant", "Task"],
    "dataset8": ["Participant", "Task", "Marker"],
}


# ---- small helpers ----
def _sanitize_tag(s: str) -> str:
    """Make a filesystem- and csv-friendly tag: lowercase, keep [A-Za-z0-9._-]."""
    return re.sub(r'[^A-Za-z0-9._-]+', '-', str(s)).strip('-').lower()

def _as_list_datasets(tag: str) -> List[str]:
    """Split a dataset tag 'a+b+c' into ['a','b','c']."""
    return [t for t in str(tag).split("+") if t]

# ==== Auto variant discovery (expects flat files: <mode>_<variant>.npz) ====
def discover_variants_for_mode(datasets: List[str], mode: str) -> List[str]:
    """Scan output/<dataset>/ for files matching '{mode}_<variant>.npz' and collect variants."""
    pat = re.compile(rf"^{re.escape(mode)}_(.+)\.npz$")
    found = set()
    for ds in datasets:
        base = (DATA_ROOT / ds)
        if not base.exists():
            continue
        for f in base.glob(f"{mode}_*.npz"):
            m = pat.match(f.name)
            if m:
                found.add(m.group(1))
    return sorted(found)

def list_variants_per_dataset(datasets: List[str], mode: str) -> Dict[str, List[str]]:
    """List variants per dataset (to produce helpful error messages if auto-scan finds none)."""
    pat = re.compile(rf"^{re.escape(mode)}_(.+)\.npz$")
    view: Dict[str, List[str]] = {}
    for ds in datasets:
        base = DATA_ROOT / ds
        vs = []
        if base.exists():
            for f in base.glob(f"{mode}_*.npz"):
                m = pat.match(f.name)
                if m:
                    vs.append(m.group(1))
        view[ds] = sorted(set(vs))
    return view

# ==== Loaders ====
def _npz_pack(base: Path, dataset: str, signal: str, variant: str):
    """
    Load a single-modality pack (ecg OR eda):
    - npz must contain X (N,T[,1]) and y (N,), meta CSV must match length.
    - Ensures (N,T,1) shape for X.
    """
    root = base / dataset
    npz_path  = root / f"{signal}_{variant}.npz"
    meta_path = root / f"{signal}_{variant}_meta.csv"
    if not npz_path.exists() or not meta_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], np.asarray(d["y"]).astype(int)
    if X.ndim == 2:  # (N,T) -> (N,T,1)
        X = X[:, :, None]
    meta = pd.read_csv(meta_path)
    assert len(X) == len(y) == len(meta), f"Length mismatch in {npz_path}"
    if signal in ("ecg","eda"):
        assert X.ndim == 3 and X.shape[-1] == 1, f"Expect (N,T,1), got {X.shape}"
    return {"X": X, "y": y, "meta": meta}

def _npz_pack_combo(base: Path, dataset: str, variant: str):
    """
    Load a combo pack (ecg + eda):
    - npz must contain X (N,T,2) and y (N,), meta CSV must match length.
    """
    root = base / dataset
    npz_path  = root / f"combo_{variant}.npz"
    meta_path = root / f"combo_{variant}_meta.csv"
    if not npz_path.exists() or not meta_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], np.asarray(d["y"]).astype(int)
    assert X.ndim == 3 and X.shape[-1] == 2, f"combo_{variant}.npz expects (N,T,2), got {X.shape}"
    meta = pd.read_csv(meta_path)
    assert len(X) == len(y) == len(meta), f"Length mismatch in {npz_path}"
    return {"X": X, "y": y, "meta": meta}

def load_combo_for_dataset(dataset: str, variant: str) -> Optional[Dict[str, Any]]:
    """Load combo pack for a single dataset."""
    return _npz_pack_combo(DATA_ROOT, dataset, variant)

def load_combo_multi(datasets: List[str], variant: str) -> Optional[Dict[str, Any]]:
    """Concatenate combo packs over multiple datasets; ensure consistent (T,2) across sets."""
    Xs, ys, Ms = [], [], []
    for ds in datasets:
        pack = load_combo_for_dataset(ds, variant)
        if pack is None: 
            continue
        Xs.append(pack["X"]); ys.append(pack["y"])
        m = pack["meta"].copy(); m["Dataset"] = ds; Ms.append(m)
    if not Xs:
        return None
    shapes = {(x.shape[1], x.shape[2]) for x in Xs}
    if len(shapes) != 1:
        raise ValueError(f"Mismatched input shapes: {shapes}")
    return {"X": np.concatenate(Xs, axis=0),
            "y": np.concatenate(ys, axis=0),
            "meta": pd.concat(Ms, axis=0, ignore_index=True)}

def load_single_multi(datasets: List[str], variant: str, modality: str) -> Optional[Dict[str, Any]]:
    """Concatenate single-modality packs over datasets; ensure consistent (T,1)."""
    Xs, ys, Ms = [], [], []
    for ds in datasets:
        pack = _npz_pack(DATA_ROOT, ds, modality, variant)
        if pack is None:
            continue
        X, y, meta = pack["X"], pack["y"], pack["meta"].copy()
        meta["Dataset"] = ds
        Xs.append(X); ys.append(y); Ms.append(meta)
    if not Xs:
        return None
    shapes = {(x.shape[1], x.shape[2] if x.ndim==3 else 1) for x in Xs}
    if len(shapes) != 1:
        raise ValueError(f"Mismatched input shapes: {shapes}")
    return {"X": np.concatenate(Xs, axis=0),
            "y": np.concatenate(ys, axis=0),
            "meta": pd.concat(Ms, axis=0, ignore_index=True)}

# ==== Grouping ====
def make_groups(meta: pd.DataFrame, dataset_tag: str) -> np.ndarray:
    """
    Build group IDs used by StratifiedGroupKFold:
    - Prefer dataset-specific keys from GROUP_KEYS_MAP if present in meta.
    - Otherwise, fall back to available columns from GROUP_PRIORITY.
    """
    m = meta.copy()
    if "Dataset" not in m.columns:
        m["Dataset"] = dataset_tag
    fallback_keys_all = [c for c in GROUP_PRIORITY if c in m.columns]
    if not fallback_keys_all:
        # As a last resort, fall back to unique per-row group IDs.
        return np.arange(len(m)).astype(str)
    groups = pd.Series(index=m.index, dtype="object")
    for ds in m["Dataset"].astype(str).unique():
        mask = (m["Dataset"].astype(str) == ds)
        base = GROUP_KEYS_MAP.get(ds, None)
        keys = (["Dataset"] + base) if base and all(k in m.columns for k in base) else fallback_keys_all
        groups.loc[mask] = m.loc[mask, keys].astype(str).agg("|".join, axis=1)
    return groups.astype(str).to_numpy()

# ==== Leakage check ====
def _assert_no_key_overlap(meta: pd.DataFrame, tr_idx: np.ndarray, va_idx: np.ndarray):
    """
    Ensure no group leakage between train/val (or train/test):
    - For each dataset, reconstruct the grouping key and verify empty intersection.
    """
    if "Dataset" not in meta.columns:
        raise RuntimeError("Missing 'Dataset' column for overlap check.")
    for ds in meta["Dataset"].astype(str).unique():
        keys = GROUP_KEYS_MAP.get(ds, None)
        if not keys or not all(k in meta.columns for k in keys):
            continue
        tr = set(meta.iloc[tr_idx].loc[meta["Dataset"]==ds, keys].astype(str).agg("|".join, axis=1))
        va = set(meta.iloc[va_idx].loc[meta["Dataset"]==ds, keys].astype(str).agg("|".join, axis=1))
        if tr & va:
            raise RuntimeError(f"Group leakage detected in dataset '{ds}'")

# ==== Threshold selection (no calibration) ====
def choose_threshold(y_true: np.ndarray, probs: np.ndarray, mode: Literal["f1","youden"]="f1") -> float:
    """
    Pick a fixed threshold on validation:
    - mode='f1': maximize F1 on the grid.
    - mode='youden': maximize TPR - FPR (Youden's J).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs).astype(float)
    grid = np.unique(np.concatenate([np.linspace(0,1,201), p, [0.0,1.0]]))
    best_thr, best_score = 0.5, -1.0
    for thr in grid:
        y_hat = (p >= thr).astype(int)
        if mode == "f1":
            score = f1_score(y, y_hat, zero_division=0)
        else:
            tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0,1]).ravel()
            tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
            fpr = fp/(fp+tn) if (fp+tn)>0 else 1.0
            score = tpr - fpr
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return float(best_thr)

# ==== Metrics / Eval ====
@torch.inference_mode()
def evaluate_model(model: pl.LightningModule, dataset: TensorDataset, batch_size=128, threshold: float = 0.5):
    """
    Evaluate a trained model on a dataset and return common classification metrics.
    Uses a fixed decision threshold for F1/precision/recall and confusion matrix.
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    logits_list, y_list = [], []
    for xb, yb in loader:
        logit = model(xb.to(device)).detach().view(-1).cpu()
        logits_list.append(logit); y_list.append(yb.view(-1).cpu())
    if not logits_list:
        return {
            "accuracy": float("nan"), "auroc": float("nan"), "auprc": float("nan"),
            "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
            "n": 0, "confusion_matrix": [[0,0],[0,0]], "threshold": float(threshold)
        }
    logits = torch.cat(logits_list).numpy()
    y_true = torch.cat(y_list).numpy().astype(int)
    # Stable sigmoid for large magnitudes
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    thr = float(threshold)
    y_pred = (probs >= thr).astype(int)

    labels = np.unique(y_true)
    has_both = (labels.size == 2)
    auroc = roc_auc_score(y_true, probs) if has_both else float("nan")
    auprc = average_precision_score(y_true, probs) if has_both else float("nan")
    f1v   = f1_score(y_true, y_pred, zero_division=0) if has_both else float("nan")
    prec  = precision_score(y_true, y_pred, zero_division=0) if has_both else float("nan")
    rec   = recall_score(y_true, y_pred, zero_division=0) if has_both else float("nan")
    cm    = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    acc   = float((y_pred == y_true).mean())
    return {
        "accuracy": acc, "auroc": float(auroc), "auprc": float(auprc),
        "f1": float(f1v), "precision": float(prec), "recall": float(rec),
        "n": int(len(y_true)), "confusion_matrix": cm, "threshold": thr
    }

def class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """Count class balance: N, class_0, class_1, and positive rate."""
    y = np.asarray(y).astype(int)
    c0 = int(np.sum(y==0)); c1 = int(np.sum(y==1)); n = int(len(y))
    return {"N": n, "class_0": c0, "class_1": c1, "ratio_1": (c1/n if n>0 else float("nan"))}

# ==== I/O ====
def save_json(path: Path, payload: Dict[str, Any]):
    """Write a JSON file with indent and ensure parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def save_weights(arch: str, train_tag: str, variant: str, cfg_idx: int, seed: int,
                 fold: int, model: pl.LightningModule, modality: Optional[str], cross_tag: str):
    """Save model weights per (arch/train/variant/config/modality/fold/seed/cross) to results/weights/."""
    out_dir = WEIGHTS_DIR / arch / train_tag / variant / f"config_{cfg_idx+1}"
    if modality: out_dir = out_dir / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = modality or "combo"
    ctag = _sanitize_tag(cross_tag)
    path = out_dir / f"{tag}_fold{fold}_seed{seed}_{ctag}.pt"
    torch.save(model.state_dict(), path)
    print(f"[WEIGHTS] {path}")

def save_split_csv(arch: str, train_tag: str, variant: str, modality: Optional[str],
                   cfg_idx: int, seed: int, fold: int,
                   meta: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                   tr_idx: np.ndarray, va_idx: np.ndarray, cross_tag: str):
    """Persist a CSV describing the train/val split, including labels and group IDs."""
    df_tr = meta.iloc[tr_idx].copy(); df_tr["__index"]=tr_idx; df_tr["split"]="train"; df_tr["label"]=y[tr_idx]; df_tr["group_id"]=groups[tr_idx]
    df_va = meta.iloc[va_idx].copy(); df_va["__index"]=va_idx; df_va["split"]="val";   df_va["label"]=y[va_idx]; df_va["group_id"]=groups[va_idx]
    df = pd.concat([df_tr, df_va], axis=0).reset_index(drop=True)
    df["arch"]=arch; df["train_tag"]=train_tag; df["variant"]=variant; df["modality"]=modality or "combo"
    df["config_id"]=cfg_idx+1; df["seed"]=seed; df["fold_id"]=fold
    df["cross"]=cross_tag
    df["trained_on_datasets"]="+".join(_as_list_datasets(train_tag))
    out = RESULTS_ROOT / "splits" / arch / train_tag / variant / (modality or "combo")
    out.mkdir(parents=True, exist_ok=True)
    ctag = _sanitize_tag(cross_tag)
    p = out / f"splits_config_{cfg_idx+1}_seed{seed}_fold{fold}_{ctag}.csv"
    df.to_csv(p, index=False); print(f"[SPLIT] {p}")

def save_test_split_csv(arch: str, train_tag: str, variant: str, modality: Optional[str],
                        cfg_idx: int, seed: int, meta: pd.DataFrame, y: np.ndarray,
                        groups: np.ndarray, te_idx: np.ndarray, cross_tag: str, test_tag_suffix: str = ""):
    """Persist a CSV describing the test split (holdout/internal20/etc.)."""
    df_te = meta.iloc[te_idx].copy()
    df_te["__index"] = te_idx
    df_te["split"] = "test"
    df_te["label"] = y[te_idx]
    df_te["group_id"] = groups[te_idx]
    df_te["arch"]=arch; df_te["train_tag"]=train_tag; df_te["variant"]=variant; df_te["modality"]=modality or "combo"
    df_te["config_id"]=cfg_idx+1; df_te["seed"]=seed; df_te["fold_id"]=0
    df_te["cross"]=cross_tag
    df_te["trained_on_datasets"]="+".join(_as_list_datasets(train_tag))
    out = RESULTS_ROOT / "splits" / arch / train_tag / variant / (modality or "combo")
    out.mkdir(parents=True, exist_ok=True)
    ctag = _sanitize_tag(cross_tag)
    suffix = f"_{_sanitize_tag(test_tag_suffix)}" if test_tag_suffix else ""
    p = out / f"testsplit_config_{cfg_idx+1}_seed{seed}_{ctag}{suffix}.csv"
    df_te.to_csv(p, index=False); print(f"[TEST-SPLIT] {p}")

def _nanmeanstd(vs: List[float]) -> Tuple[float,float]:
    """Mean and std that ignore NaNs (returns NaN if input is empty)."""
    arr = np.asarray(vs, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))

def subset_pack(pack: Dict[str,Any], idx: np.ndarray) -> Dict[str,Any]:
    """Slice a loaded pack by indices and reset meta index."""
    X, y, meta = pack["X"], pack["y"], pack["meta"]
    return {"X": X[idx], "y": y[idx], "meta": meta.iloc[idx].reset_index(drop=True)}

# ==== Core training/evaluation ====
def train_and_eval(train_pack: Dict[str,Any], train_tag: str, arch_name: str, variant: str, modality: Optional[str],
                   cfg: Dict[str,Any], seed: int, folds: int, batch_size: int, use_amp: bool,
                   external_tests: Optional[Dict[str, Dict[str,Any]]] = None,
                   thr_mode: Literal["f1","youden"]="f1",
                   cross_tag: str = "none"):
    """
    Cross-validated training on train_pack:
    - For each inner fold, pick threshold on validation only (by thr_mode).
    - Save per-fold JSON + weights.
    - Optionally evaluate on external test packs per fold (holdout/transfer).
    - Aggregate summaries across folds + external tests and save a seed-level SUMMARY.
    """
    X, y, meta = train_pack["X"], train_pack["y"], train_pack["meta"].copy()
    if "Dataset" not in meta.columns:
        meta["Dataset"] = train_tag
    input_shape = X.shape[1:]
    bs = int(cfg.get("batch_size", batch_size))
    pl.seed_everything(seed, workers=True)

    groups = make_groups(meta, train_tag)
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_logs: List[Dict[str, Any]] = []
    cm_sum = np.zeros((2,2), dtype=int); n_total = 0
    train_ds_list = _as_list_datasets(train_tag)

    # prepare containers for any external tests
    ext_logs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in (external_tests or {}).keys()}

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups), start=1):
        if len(tr_idx) < 2 or len(np.unique(y[tr_idx])) < 2:
            print(f"[WARN] fold {fold}: insufficient class variety – skipped.")
            continue

        _assert_no_key_overlap(meta, tr_idx, va_idx)
        save_split_csv(arch_name, train_tag, variant, modality, cfg_idx=cfg["__cfg_idx__"], seed=seed, fold=fold,
                       meta=meta, y=y, groups=groups, tr_idx=tr_idx, va_idx=va_idx, cross_tag=cross_tag)

        # --- build TensorDataset for train/val
        tr_ds = TensorDataset(torch.tensor(X[tr_idx], dtype=torch.float32), torch.tensor(y[tr_idx], dtype=torch.long))
        va_ds = TensorDataset(torch.tensor(X[va_idx], dtype=torch.float32), torch.tensor(y[va_idx], dtype=torch.long))

        # --- class imbalance handling via BCEWithLogits pos_weight
        binc = np.bincount(y[tr_idx], minlength=2)
        pos_w = float(binc[0] / max(1, binc[1])) if binc[1] > 0 else None

        # --- choose model architecture
        model = build_cnn_mh(cfg, input_shape, pos_weight=pos_w) if modality is None else build_cnn_single(cfg, input_shape, pos_weight=pos_w)

        # --- Lightning Trainer
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=MAX_EPOCHS,
            precision=("16-mixed" if use_amp and torch.cuda.is_available() else 32),
            logger=False,
            enable_checkpointing=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")],
            gradient_clip_val=1.0,
        )
        trainer.fit(
            model,
            DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, drop_last=True),
            DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
        )

        # --- choose threshold on validation predictions only
        device = next(model.parameters()).device
        val_logits, val_targets = [], []
        with torch.inference_mode():
            for xb, yb in DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS):
                val_logits.append(model(xb.to(device)).detach().view(-1).cpu())
                val_targets.append(yb.view(-1).cpu())
        val_logits = torch.cat(val_logits).numpy()
        val_targets = torch.cat(val_targets).numpy().astype(int)
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -50, 50)))
        thr_fixed = choose_threshold(val_targets, val_probs, mode=thr_mode)

        # --- validation metrics with fixed threshold
        metrics_val = evaluate_model(model, va_ds, batch_size=bs, threshold=thr_fixed)

        # --- persist per-fold JSON + weights
        out_dir = RESULTS_ROOT / arch_name / train_tag / variant / (modality or "combo")
        out_dir.mkdir(parents=True, exist_ok=True)
        ctag = _sanitize_tag(cross_tag)
        per_fold_path = out_dir / f"{train_tag}_{variant}_{modality or 'combo'}_config_{cfg['__cfg_idx__']+1}_seed{seed}_fold{fold}_{ctag}.json"
        save_json(per_fold_path, {
            "arch": arch_name, "train_tag": train_tag, "variant": variant, "modality": modality or "combo",
            "config_id": cfg["__cfg_idx__"]+1, "seed": seed, "fold": fold,
            "cross": cross_tag,
            "trained_on_datasets": train_ds_list,
            "tested_on_datasets": train_ds_list,
            "hyperparameters": {k: (v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v,'item') else v))
                                for k,v in cfg.items() if not k.startswith("__")},
            "metrics_val": {
                "accuracy": metrics_val["accuracy"], "auroc": metrics_val["auroc"], "auprc": metrics_val["auprc"],
                "f1": metrics_val["f1"], "precision": metrics_val["precision"], "recall": metrics_val["recall"],
                "n": metrics_val["n"], "confusion_matrix": metrics_val["confusion_matrix"],
                "threshold": float(thr_fixed)
            },
            "train_distribution": class_distribution(y[tr_idx]),
            "val_distribution": class_distribution(y[va_idx])
        })
        save_weights(arch_name, train_tag, variant, cfg_idx=cfg["__cfg_idx__"], seed=seed, fold=fold,
                     model=model, modality=modality, cross_tag=cross_tag)

        # --- collect fold logs for summary
        fold_logs.append({
            "fold": fold,
            "threshold": float(thr_fixed),
            "accuracy": metrics_val["accuracy"], "auroc": metrics_val["auroc"], "auprc": metrics_val["auprc"],
            "f1": metrics_val["f1"], "precision": metrics_val["precision"], "recall": metrics_val["recall"],
            "n": metrics_val["n"], "confusion_matrix": metrics_val["confusion_matrix"],
            "trained_on_datasets": train_ds_list, "tested_on_datasets": train_ds_list, "cross": cross_tag,
            "train_distribution": class_distribution(y[tr_idx]),
            "val_distribution": class_distribution(y[va_idx]),
        })
        cm_sum += np.array(metrics_val["confusion_matrix"], dtype=int)
        n_total += int(metrics_val["n"])

        # --- optional: external tests per fold (transfer/pairwise/pool/holdout)
        if external_tests:
            ctag = _sanitize_tag(cross_tag)
            ho_dir = out_dir / "holdout"
            ho_dir.mkdir(parents=True, exist_ok=True)
            for test_name, pack in external_tests.items():
                Xt, yt, _ = pack["X"], pack["y"], pack["meta"]
                test_ds = TensorDataset(torch.tensor(Xt, dtype=torch.float32), torch.tensor(yt, dtype=torch.long))
                m_ho = evaluate_model(model, test_ds, batch_size=bs, threshold=thr_fixed)
                ho_path = ho_dir / f"{train_tag}_TO_{test_name}_{variant}_{modality or 'combo'}_config_{cfg['__cfg_idx__']+1}_seed{seed}_fold{fold}_{ctag}.json"
                save_json(ho_path, {
                    "arch": arch_name, "train_tag": train_tag, "test_tag": test_name, "variant": variant, "modality": modality or "combo",
                    "config_id": cfg["__cfg_idx__"]+1, "seed": seed, "fold": fold, "cross": cross_tag,
                    "trained_on_datasets": train_ds_list, "tested_on_datasets": [str(test_name)],
                    "hyperparameters": {k: (v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v,'item') else v))
                                        for k,v in cfg.items() if not k.startswith("__")},
                    "metrics_holdout": {
                        "accuracy": m_ho["accuracy"], "auroc": m_ho["auroc"], "auprc": m_ho["auprc"],
                        "f1": m_ho["f1"], "precision": m_ho["precision"], "recall": m_ho["recall"],
                        "n": m_ho["n"], "confusion_matrix": m_ho["confusion_matrix"],
                        "threshold": float(thr_fixed)
                    },
                    "test_distribution": class_distribution(yt)
                })
                ext_logs[test_name].append({
                    "fold": fold,
                    "accuracy": m_ho["accuracy"], "auroc": m_ho["auroc"], "auprc": m_ho["auprc"],
                    "f1": m_ho["f1"], "precision": m_ho["precision"], "recall": m_ho["recall"],
                    "n": m_ho["n"], "confusion_matrix": m_ho["confusion_matrix"],
                    "threshold": float(thr_fixed),
                })

        # free CUDA mem and clean PL logs per fold
        torch.cuda.empty_cache()
        shutil.rmtree("lightning_logs", ignore_errors=True)

    # === Seed-level summary (validation) ===
    acc_m, acc_s   = _nanmeanstd([m["accuracy"] for m in fold_logs])
    auc_m, auc_s   = _nanmeanstd([m["auroc"] for m in fold_logs])
    aupr_m, aupr_s = _nanmeanstd([m["auprc"] for m in fold_logs])
    f1_m, f1_s     = _nanmeanstd([m["f1"] for m in fold_logs])
    prec_m, prec_s = _nanmeanstd([m["precision"] for m in fold_logs])
    rec_m,  rec_s  = _nanmeanstd([m["recall"] for m in fold_logs])
    thr_m, thr_s   = _nanmeanstd([m["threshold"] for m in fold_logs])

    # === Seed-level summaries for external tests (aggregated over folds) ===
    test_summaries = {}
    for tname, logs in ext_logs.items():
        if not logs:
            continue
        accM, accS   = _nanmeanstd([m["accuracy"] for m in logs])
        aucM, aucS   = _nanmeanstd([m["auroc"] for m in logs])
        auprM, auprS = _nanmeanstd([m["auprc"] for m in logs])
        f1M, f1S     = _nanmeanstd([m["f1"] for m in logs])
        precM, precS = _nanmeanstd([m["precision"] for m in logs])
        recM,  recS  = _nanmeanstd([m["recall"] for m in logs])
        thrM, thrS   = _nanmeanstd([m["threshold"] for m in logs])
        cm_sum_test  = np.zeros((2,2), dtype=int)
        n_total_test = 0
        for m in logs:
            cm_sum_test += np.array(m["confusion_matrix"], dtype=int)
            n_total_test += int(m["n"])
        test_summaries[tname] = {
            "accuracy_mean": accM, "accuracy_std": accS,
            "auroc_mean": aucM, "auroc_std": aucS,
            "auprc_mean": auprM, "auprc_std": auprS,
            "f1_mean": f1M, "f1_std": f1S,
            "precision_mean": precM, "precision_std": precS,
            "recall_mean": recM, "recall_std": recS,
            "threshold_mean": thrM, "threshold_std": thrS,
            "n_total": int(n_total_test),
            "confusion_matrix_sum": cm_sum_test.tolist(),
            "trained_on_datasets": train_ds_list,
            "tested_on_datasets": [str(tname)],
            "cross": cross_tag
        }

    # === Persist seed-level SUMMARY JSON ===
    summary = {
        "arch": arch_name, "train_tag": train_tag, "variant": variant, "modality": modality or "combo",
        "config_id": cfg["__cfg_idx__"]+1, "seed": seed, "folds": len(fold_logs),
        "cross": cross_tag,
        "trained_on_datasets": train_ds_list,
        "tested_on_datasets": train_ds_list if not external_tests else sorted(list({str(k) for k in (external_tests or {}).keys()})),
        "hyperparameters": {k: (v.tolist() if isinstance(v, np.ndarray) else (v.item() if hasattr(v,'item') else v))
                            for k,v in cfg.items() if not k.startswith("__")},
        "val_summary": {
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
            "auroc_mean": auc_m, "auroc_std": auc_s,
            "auprc_mean": aupr_m, "auprc_std": aupr_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "precision_mean": prec_m, "precision_std": prec_s,
            "recall_mean": rec_m, "recall_std": rec_s,
            "threshold_mean": thr_m, "threshold_std": thr_s,
            "n_total": int(n_total),
            "confusion_matrix_sum": cm_sum.tolist()
        },
        "test_summaries": test_summaries,
        "val_per_fold": fold_logs
    }

    out_dir = RESULTS_ROOT / arch_name / train_tag / variant / (modality or "combo")
    ctag = _sanitize_tag(cross_tag)
    sp = out_dir / f"SUMMARY_config_{cfg['__cfg_idx__']+1}_seed{seed}_{ctag}.json"
    save_json(sp, summary)
    print(f"[SUMMARY] {sp}")

# ==== Orchestration for different cross-dataset regimes ====
def run_cross(datasets: List[str], variant: str, mode: str, cfg_idx: int, seed: int,
              folds: int, batch_size: int, use_amp: bool, cross: str,
              thr_mode: Literal["f1","youden"]="f1"):
    """
    Run a single configuration (cfg_idx) for a given mode/variant across different regimes:
    - none: train/test split within each dataset (internal holdout 20%)
    - pairwise: train on pairs, eval on the remaining datasets
    - lodo: leave-one-dataset-out
    - pool: train on all, eval on each dataset
    - transfer: train on one, eval on the others
    """
    # Sample a CNN config (both single and combo use the same space)
    cs  = get_cnn_config_space(seed=seed)
    cfg = dict(cs.sample_configuration())
    cfg = {k: (v.item() if hasattr(v,"item") else v) for k,v in cfg.items()}
    cfg["__cfg_idx__"] = cfg_idx

    # Helper to build the combined training pack per regime
    def build_pack(ds_list: List[str]):
        return load_combo_multi(ds_list, variant) if mode == "combo" else load_single_multi(ds_list, variant, mode)

    # Preload per-dataset packs for quick reuse
    ds_packs = {ds: (load_combo_for_dataset(ds, variant) if mode=="combo" else _npz_pack(DATA_ROOT, ds, mode, variant))
                for ds in datasets}

    # If no external datasets exist for a regime, carve out an internal 20% test split
    def ensure_external_or_internal(train_pack: Dict[str,Any], train_tag: str,
                                    arch_name: str, modality: Optional[str], cross_tag: str) -> Dict[str, Dict[str,Any]]:
        if train_pack is None:
            return {}
        meta_all = train_pack["meta"].copy()
        if "Dataset" not in meta_all.columns:
            meta_all["Dataset"] = train_tag
        y_all = train_pack["y"]
        groups_all = make_groups(meta_all, dataset_tag=train_tag)
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        tr_idx, te_idx = next(outer.split(np.zeros(len(y_all)), y_all, groups_all))
        _assert_no_key_overlap(meta_all, tr_idx, te_idx)
        save_test_split_csv(
            (ARCH_COMBO if mode=="combo" else ARCH_SINGLE), train_tag, variant, (None if mode=="combo" else mode),
            cfg_idx=cfg["__cfg_idx__"], seed=seed,
            meta=meta_all, y=y_all, groups=groups_all, te_idx=te_idx,
            cross_tag=cross_tag, test_tag_suffix="INTERNAL20"
        )
        pack_te = subset_pack({"X": train_pack["X"], "y": train_pack["y"], "meta": meta_all}, te_idx)
        return {f"{train_tag}_INTERNAL20": pack_te}

    arch_name = (ARCH_COMBO if mode=="combo" else ARCH_SINGLE)
    modality  = (None if mode=="combo" else mode)

    # === NONE: per-dataset internal holdout
    if cross in ("none","all"):
        for ds in datasets:
            pack_full = ds_packs[ds]
            if pack_full is None:
                print(f"[SKIP] {ds}:{variant}:{mode} – no data")
                continue
            meta = pack_full["meta"].copy()
            if "Dataset" not in meta.columns:
                meta["Dataset"] = ds
                pack_full = {"X": pack_full["X"], "y": pack_full["y"], "meta": meta}
            y_all = pack_full["y"]
            groups_all = make_groups(pack_full["meta"], dataset_tag=ds)
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            tr_idx, te_idx = next(outer.split(np.zeros(len(y_all)), y_all, groups_all))
            _assert_no_key_overlap(pack_full["meta"], tr_idx, te_idx)
            save_test_split_csv(arch_name, ds, variant, modality,
                                cfg_idx=cfg["__cfg_idx__"], seed=seed,
                                meta=pack_full["meta"], y=y_all, groups=groups_all, te_idx=te_idx,
                                cross_tag="none")
            pack_tr = subset_pack(pack_full, tr_idx)
            pack_te = subset_pack(pack_full, te_idx)
            train_and_eval(pack_tr, train_tag=ds, arch_name=arch_name,
                           variant=variant, modality=modality,
                           cfg=cfg, seed=seed, folds=folds, batch_size=batch_size, use_amp=use_amp,
                           external_tests={f"{ds}_HOLDOUT20": pack_te},
                           thr_mode=thr_mode, cross_tag="none")

    # === PAIRWISE: train on pairs, test on others
    if cross in ("pairwise","all"):
        for a, b in combinations(datasets, 2):
            pack_pair = build_pack([a, b])
            if pack_pair is None:
                print(f"[PAIRWISE] skip {a}+{b} (missing data)")
                continue
            others = [d for d in datasets if d not in (a, b)]
            external = {d: ds_packs[d] for d in others if ds_packs.get(d) is not None}
            if not external:
                external = ensure_external_or_internal(pack_pair, train_tag=f"{a}+{b}",
                                                       arch_name=arch_name, modality=modality, cross_tag="pairwise")
            train_and_eval(pack_pair, train_tag=f"{a}+{b}", arch_name=arch_name,
                           variant=variant, modality=modality,
                           cfg=cfg, seed=seed, folds=folds, batch_size=batch_size, use_amp=use_amp,
                           external_tests=external, thr_mode=thr_mode, cross_tag="pairwise")

    # === LODO: leave-one-dataset-out
    if cross in ("lodo","all"):
        for hold in datasets:
            train_list = [d for d in datasets if d != hold]
            pack_tr = build_pack(train_list)
            pack_ho = ds_packs.get(hold, None)
            if pack_tr is None or pack_ho is None:
                print(f"[LODO] skip train={train_list} holdout={hold} (missing data)")
                continue
            train_and_eval(pack_tr, train_tag="+".join(train_list), arch_name=arch_name,
                           variant=variant, modality=modality,
                           cfg=cfg, seed=seed, folds=folds, batch_size=batch_size, use_amp=use_amp,
                           external_tests={hold: pack_ho}, thr_mode=thr_mode, cross_tag="lodo")

    # === POOL: train on all, test on each dataset (or carve internal if none)
    if cross in ("pool","all"):
        pack_all = build_pack(datasets)
        if pack_all is not None:
            tests = {ds: pk for ds, pk in ds_packs.items() if pk is not None}
            if not tests:
                tests = ensure_external_or_internal(pack_all, train_tag="+".join(datasets),
                                                    arch_name=arch_name, modality=modality, cross_tag="pool")
            train_and_eval(pack_all, train_tag="+".join(datasets), arch_name=arch_name,
                           variant=variant, modality=modality,
                           cfg=cfg, seed=seed, folds=folds, batch_size=batch_size, use_amp=use_amp,
                           external_tests=tests, thr_mode=thr_mode, cross_tag="pool")

    # === TRANSFER: train on one, test on all the others
    if cross in ("transfer","all"):
        for tr in datasets:
            pack_tr = build_pack([tr])
            if pack_tr is None:
                print(f"[TRANSFER] skip train={tr} (missing data)")
                continue
            external = {d: ds_packs[d] for d in datasets if d != tr and ds_packs.get(d) is not None}
            if not external:
                external = ensure_external_or_internal(pack_tr, train_tag=tr,
                                                       arch_name=arch_name, modality=modality, cross_tag="transfer")
            train_and_eval(pack_tr, train_tag=tr, arch_name=arch_name,
                           variant=variant, modality=modality,
                           cfg=cfg, seed=seed, folds=folds, batch_size=batch_size, use_amp=use_amp,
                           external_tests=external, thr_mode=thr_mode, cross_tag="transfer")

# ==== CLI / Main ====
def parse_args():
    """Command-line interface for running cross-dataset experiments."""
    ap = argparse.ArgumentParser("CNN Cross-Training (combo/ecg/eda) – no calibration, with holdout & JSON logs")
    ap.add_argument("--datasets", default="dataset1,dataset2,dataset3,dataset8 ", help="Comma-separated dataset list")
    ap.add_argument("--modes", default="ecg", help="combo, ecg, eda or all")
    ap.add_argument("--variants", default="auto", help="Comma-separated list (e.g., none,z_window) or 'auto'")
    ap.add_argument("--configs", type=int, default=DEF_CONFIGS)
    ap.add_argument("--folds",   type=int, default=DEF_FOLDS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE_D)
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision even if CUDA is available")
    ap.add_argument("--seed_base", type=int, default=42, help="seed = seed_base + cfg_idx")
    ap.add_argument("--cross", choices=["none","pairwise","lodo","pool","transfer","all"], default="transfer")
    ap.add_argument("--thr_mode", choices=["f1","youden"], default="youden",
                    help="How to pick validation threshold; applied to holdout/test")
    return ap.parse_args()

def main():
    args = parse_args()
    use_amp = not args.no_amp
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    modes_raw = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    modes = ["combo","ecg","eda"] if (len(modes_raw)==1 and modes_raw[0]=="all") else modes_raw

    # Detect/validate variants for each mode
    mode_to_variants: Dict[str, List[str]] = {}
    for mode in modes:
        if args.variants == "auto":
            auto_vs = discover_variants_for_mode(datasets, mode)
            if not auto_vs:
                per_ds = list_variants_per_dataset(datasets, mode)
                msg = [f"Auto-scan found NO variants for mode='{mode}'. Expected files: output/<dataset>/{mode}_<variant>.npz"]
                msg.append("Available variants per dataset:")
                for ds, vs in per_ds.items():
                    msg.append(f"  - {ds}: {vs}")
                msg.append("Fix: create files (e.g., ecg_none.npz + ecg_none_meta.csv) or set --variants explicitly.")
                raise RuntimeError("\n".join(msg))
            mode_to_variants[mode] = auto_vs
        else:
            mode_to_variants[mode] = [s.strip() for s in args.variants.split(",") if s.strip()]

    # Run
    print("== RUN (CNN) ==")
    print(f"Datasets={datasets} | Modes={modes} | Variants per mode={mode_to_variants}")
    print(f"Configs={args.configs} | Folds={args.folds} | Batch={args.batch_size} | AMP={'on' if use_amp else 'off'}")
    print(f"Seed policy: seed = seed_base + cfg_idx (seed_base={args.seed_base}) | Cross={args.cross}")
    print(f"Threshold policy: chosen on Validation by {args.thr_mode}, applied to Test/Holdout")

    for mode in modes:
        variants = mode_to_variants[mode]
        for variant in variants:
            for cfg_idx in range(args.configs):
                seed = int(args.seed_base) + cfg_idx
                run_cross(datasets, variant, mode, cfg_idx, seed, args.folds, args.batch_size, use_amp,
                          cross=args.cross, thr_mode=args.thr_mode)

if __name__ == "__main__":
    # Enable MPS with CUDA fallback on Apple devices if needed.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
