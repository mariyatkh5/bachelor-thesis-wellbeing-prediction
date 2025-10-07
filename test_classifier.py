# test_classifiers_correct.py
from __future__ import annotations

import sys, json, os, glob, platform
from typing import List, Tuple
import pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

from utils.feature_loader import load_data
from utils.learner_pipeline import get_pipeline_from_config
import joblib


# ---------- Helpers ----------
def _split_participant_task(groups) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if isinstance(groups, pd.Series):
            parts = groups.astype(str).str.split("|", n=1, expand=True)
            if parts.shape[1] == 2:
                return parts[0].astype(str).values, parts[1].astype(str).values
            return parts[0].astype(str).values, np.array(["?"] * len(parts))
        if isinstance(groups, pd.DataFrame) and {"Participant","Task"}.issubset(groups.columns):
            return groups["Participant"].astype(str).values, groups["Task"].astype(str).values
        p = np.array([str(t[0]) for t in groups])
        t = np.array([str(t[1]) for t in groups])
        return p, t
    except Exception:
        n = len(groups) if hasattr(groups, "__len__") else 0
        return np.array(["?"] * n), np.array(["?"] * n)

def _safe_scores(model, X, n_classes: int):
    try:
        proba = model.predict_proba(X)
        if n_classes == 2 and getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return np.ravel(proba) if n_classes == 2 else proba
    except Exception:
        scores = model.decision_function(X)
        if n_classes == 2 and getattr(scores, "ndim", 1) == 2 and scores.shape[1] > 1:
            return scores[:, 1]
        return scores

def _ap_score(y_true, y_score, n_classes: int):
    y_true = np.asarray(y_true)
    if n_classes == 2:
        return average_precision_score(y_true, y_score)
    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)
    return average_precision_score(y_bin, y_score, average="macro")

def _roc_auc(y_true, y_score, n_classes: int):
    y_true = np.asarray(y_true)
    if n_classes == 2:
        return roc_auc_score(y_true, y_score)
    return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

def _fit_and_eval_one(Xa: pd.DataFrame, ya: pd.Series, pl_template, i_tr, i_val, n_classes: int):
    X_tr, y_tr = Xa.iloc[i_tr], ya.iloc[i_tr]
    X_va, y_va = Xa.iloc[i_val], ya.iloc[i_val]
    model = clone(pl_template).fit(X_tr, y_tr.values.ravel())
    y_pred = model.predict(X_va)
    y_sc = _safe_scores(model, X_va, n_classes)
    return (
        accuracy_score(y_va, y_pred),
        _roc_auc(y_va, y_sc, n_classes),
        f1_score(y_va, y_pred, average="macro"),
        _ap_score(y_va, y_sc, n_classes),
        precision_score(y_va, y_pred, average="macro", zero_division=0),
        recall_score(y_va, y_pred, average="macro", zero_division=0),
        model,
        np.array(i_tr, dtype=int),
        np.array(i_val, dtype=int),
    )

def _find_history_file(label: str, scoring: str, ds_list: List[int], signals: List[str], bsl_tag: str) -> str:
    ds_key = "_".join(map(str, sorted(ds_list)))
    sig_key = "_".join(signals)
    base_dir = "results1/autoML_classifiers"
    pref_new = os.path.join(base_dir, f"naml_history_{label}_{scoring}_ds_analysis_{ds_key}_signal_types_{sig_key}{bsl_tag}")
    pref_old = os.path.join(base_dir, f"naml_history_{label}_{scoring}_ds_{ds_key}_signals_{sig_key}{bsl_tag}")

    for pref in (pref_new, pref_old):
        p = pref + ".csv"
        if os.path.exists(p):
            print(f"[INFO] Verwende AutoML-History: {p}")
            return p

    candidates = sorted(glob.glob(pref_new + "_config*_fold*.csv")) + \
                 sorted(glob.glob(pref_old + "_config*_fold*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "AutoML-Config nicht gefunden:\n"
            f"  {pref_new}.csv\n  {pref_old}.csv\n"
            "oder *_config*_fold*.csv"
        )

    best_file, best_score = None, -np.inf
    for fp in candidates:
        try:
            df = pd.read_csv(fp)
            if scoring in df.columns:
                mx = pd.to_numeric(df[scoring], errors="coerce").max()
                if pd.notna(mx) and float(mx) > best_score:
                    best_score, best_file = float(mx), fp
        except Exception:
            continue
    if best_file is None:
        raise FileNotFoundError("Gefundene History-Dateien ohne Spalte '{scoring}'.")
    print(f"[INFO] Verwende beste AutoML-History: {best_file} (max {scoring}={best_score:.4f})")
    return best_file


# ---------- Main ----------
if __name__ == "__main__":
    if platform.system() == "Darwin":
        matplotlib.use("QtAgg")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        plt.rcParams.update({"font.size": 20})

    if len(sys.argv) > 1:
        hyper = json.loads(sys.argv[1])
        signal_types: List[str]      = hyper["signal_types"]
        analysis_datasets: List[int] = hyper["analysis_datasets"]
        test_datasets: List[int]     = hyper["test_datasets"]
        scoring: str                 = hyper["scoring"]
        label: str                   = hyper["label"]
        bsl: bool                    = hyper["bsl"]
        n_repeats: int               = int(hyper.get("number_of_repeats", 100))
        val_size: float              = float(hyper.get("val_size", 0.2))
        seed: int                    = int(hyper.get("random_state", 42))
        labels_merge_path = hyper.get("labels_merge_path")
        if not labels_merge_path:
            if isinstance(test_datasets, (list, tuple)) and len(test_datasets) == 1:
                labels_merge_path = f"agg_data/dataset{test_datasets[0]}/labels.csv"
            else:
                labels_merge_path = None
    else:
        signal_types      = ["ecg"]
        analysis_datasets = [1,3]
        test_datasets     = [8]
        scoring           = "roc_auc"
        label             = "Well-being"
        bsl               = True
        n_repeats         = 100
        val_size          = 0.2
        seed              = 42
        labels_merge_path = f"agg_data/dataset{test_datasets[0]}/labels.csv" if len(test_datasets)==1 else None

    bsl_tag = "_bsl" if bsl else ""
    Xa, ya_ser, ga = load_data(analysis_datasets, signal_types, bsl=bsl, group_by=("Participant","Task"))
    Xt, yt_ser, gt = load_data(test_datasets,     signal_types, bsl=bsl, group_by=("Participant","Task"))

    if np.isnan(Xa.to_numpy(dtype=float)).any() or np.isnan(ya_ser.to_numpy()).any():
        raise ValueError("NaNs in Analyse-Daten.")
    if np.isnan(Xt.to_numpy(dtype=float)).any() or np.isnan(yt_ser.to_numpy()).any():
        raise ValueError("NaNs in Test-Daten.")

    ya = ya_ser
    yt = yt_ser.to_numpy().ravel()
    n_classes = int(np.unique(ya.to_numpy().ravel()).size)

    try:
        overlap = set(map(str, ga.values)) & set(map(str, gt.values))
        if overlap:
            print(f"[WARN] {len(overlap)} (Participant,Task)-Gruppen kommen in Analyse UND Test vor.")
    except Exception:
        pass

    history_csv = _find_history_file(label, scoring, analysis_datasets, signal_types, bsl_tag)
    pl_template = get_pipeline_from_config(history_csv, scoring)

    gss = GroupShuffleSplit(n_splits=n_repeats, test_size=val_size, random_state=seed)

    acc_all, roc_all, f1_all, ap_all = [], [], [], []
    prec_all, rec_all = [], []
    models = []
    split_rows = []  # <- hier speichern wir pro Fit die verwendeten Gruppen

    futures = []
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, n_repeats)) as ex:
        for i_tr, i_val in gss.split(Xa, ya, ga):
            y_val = ya.iloc[i_val]
            if pd.Series(y_val).nunique() < 2:
                continue
            futures.append(ex.submit(_fit_and_eval_one, Xa, ya, pl_template, i_tr, i_val, n_classes))

        results1 = []
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                results1.append(fut.result())
            except Exception as e:
                print("[WARN] Fit/Eval fehlgeschlagen:", repr(e))

    # Einsammeln + Split-Protokoll erstellen
    for fit_id, (acc, roc, f1m, ap, prec, rec, model, idx_tr, idx_val) in enumerate(results1, start=1):
        acc_all.append(acc); roc_all.append(roc); f1_all.append(f1m); ap_all.append(ap)
        prec_all.append(prec); rec_all.append(rec)
        models.append(model)

        # Gruppen aufschreiben (welche Participant|Task im Train/Val waren)
        g_tr = pd.Series(ga).iloc[idx_tr].astype(str).unique()
        g_va = pd.Series(ga).iloc[idx_val].astype(str).unique()
        for g in g_tr:
            p, t = (g.split("|", 1) + [""])[:2]
            split_rows.append({"fit_id": fit_id, "phase": "train", "participant": p, "task": t})
        for g in g_va:
            p, t = (g.split("|", 1) + [""])[:2]
            split_rows.append({"fit_id": fit_id, "phase": "val", "participant": p, "task": t})

    if not models:
        raise RuntimeError("Keine gültigen Fits erzeugt (evtl. zu kleine/unausgewogene Folds?).")

    # Test-Evaluation + Detail-Preds
    p_all, t_all = _split_participant_task(gt)
    labels_for_cm = sorted(list(set(np.unique(yt))))

    test_accs, test_rocs, test_f1m, test_prs = [], [], [], []
    test_prec_m, test_rec_m = [], []
    cms = []
    detail_rows = []

    for fit_id, m in enumerate(models, start=1):
        y_pred = m.predict(Xt)
        y_sc   = _safe_scores(m, Xt, n_classes)

        test_accs.append(accuracy_score(yt, y_pred))
        test_rocs.append(_roc_auc(yt, y_sc, n_classes))
        test_f1m.append(f1_score(yt, y_pred, average='macro'))
        test_prs.append(_ap_score(yt, y_sc, n_classes))
        test_prec_m.append(precision_score(yt, y_pred, average='macro', zero_division=0))
        test_rec_m.append(recall_score(yt, y_pred, average='macro', zero_division=0))
        cms.append(confusion_matrix(yt, y_pred, labels=labels_for_cm))

        detail_rows.append(pd.DataFrame({
            "fit_id": fit_id,
            "participant": p_all,
            "task": t_all,
            "y_true": yt,
            "y_pred": y_pred,
            "y_score": (y_sc if n_classes == 2 else np.nan),
        }))

    # Speichern
    ads = "_".join(map(str, sorted(analysis_datasets)))
    tds = "_".join(map(str, sorted(test_datasets)))
    sigs = "_".join(signal_types)

    out_csv = f"results1/test/{label}_{scoring}_ds_analysis_{ads}_ds_test_{tds}_signal_types_{sigs}{bsl_tag}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if len(labels_for_cm) == 2 and n_classes == 2:
        (l0, l1) = labels_for_cm
        cm00 = [int(cm[0,0]) for cm in cms]
        cm01 = [int(cm[0,1]) for cm in cms]
        cm10 = [int(cm[1,0]) for cm in cms]
        cm11 = [int(cm[1,1]) for cm in cms]
        df_out = pd.DataFrame({
            # CV
            "cv_accuracy": acc_all, "cv_roc_auc": roc_all, "cv_f1_macro": f1_all, "cv_pr_auc": ap_all,
            "cv_precision_macro": prec_all, "cv_recall_macro": rec_all,
            # TEST
            "test_accuracy": test_accs, "test_roc_auc": test_rocs, "test_f1_macro": test_f1m, "test_pr_auc": test_prs,
            "test_precision_macro": test_prec_m, "test_recall_macro": test_rec_m,
            # CM
            f"test_cm_{l0}{l0}": cm00, f"test_cm_{l0}{l1}": cm01, f"test_cm_{l1}{l0}": cm10, f"test_cm_{l1}{l1}": cm11,
        })
    else:
        df_out = pd.DataFrame({
            "cv_accuracy": acc_all, "cv_roc_auc": roc_all, "cv_f1_macro": f1_all, "cv_pr_auc": ap_all,
            "cv_precision_macro": prec_all, "cv_recall_macro": rec_all,
            "test_accuracy": test_accs, "test_roc_auc": test_rocs, "test_f1_macro": test_f1m, "test_pr_auc": test_prs,
            "test_precision_macro": test_prec_m, "test_recall_macro": test_rec_m,
            "test_confusion_matrix": [json.dumps(cm.tolist()) for cm in cms],
            "cm_labels": [json.dumps(labels_for_cm)] * len(cms),
        })
    df_out.to_csv(out_csv, index=False)

    preds = pd.concat(detail_rows, ignore_index=True)
    preds_path = out_csv.replace(".csv", "_preds.csv")
    preds.to_csv(preds_path, index=False)
    print(f"[SAVED] Detailed predictions → {preds_path}")

    # Splits speichern (welche Gruppen in welchem Fit/Fold)
    splits_df = pd.DataFrame(split_rows)
    splits_path = out_csv.replace(".csv", "_splits.csv")
    splits_df.to_csv(splits_path, index=False)
    print(f"[SAVED] Train/Val-Gruppen pro Fit → {splits_path}")

    # Optionales Merge mit labels.csv
    try:
        if labels_merge_path and os.path.exists(labels_merge_path):
            df_lab = pd.read_csv(labels_merge_path)
            cand_p = [c for c in df_lab.columns if c.lower() in {"participant","participant_id","subject","pid"}]
            cand_t = [c for c in df_lab.columns if c.lower() in {"task","task_id","session","sid"}]
            if cand_p and cand_t:
                df_lab = df_lab.rename(columns={cand_p[0]:"participant", cand_t[0]:"task"})
                df_lab["participant"] = df_lab["participant"].astype(str)
                df_lab["task"] = df_lab["task"].astype(str)
                merged = preds.merge(df_lab, on=["participant","task"], how="left")
                merged_path = out_csv.replace(".csv", "_preds_merged.csv")
                merged.to_csv(merged_path, index=False)
                print(f"[SAVED] Merged predictions with labels → {merged_path}")
            else:
                print(f"[WARN] labels.csv ohne passende Spalten (participant/task): {list(df_lab.columns)}")
        else:
            print(f"[WARN] labels_merge_path nicht gefunden: {labels_merge_path}")
    except Exception as e:
        print(f"[WARN] Merge mit labels.csv fehlgeschlagen: {e}")

    # Reporting
    print("----------------------------------")
    sigs_print = "_".join(signal_types)
    print(f"Setting label={label}, test_ds={tds}, signals={sigs_print}")
    print(f"CV accuracy        : {np.mean(acc_all):.4f} ± {np.std(acc_all, ddof=1):.4f}")
    print(f"CV roc_auc         : {np.mean(roc_all):.4f} ± {np.std(roc_all, ddof=1):.4f}")
    print(f"CV f1_macro        : {np.mean(f1_all):.4f} ± {np.std(f1_all, ddof=1):.4f}")
    print(f"CV pr_auc          : {np.mean(ap_all):.4f} ± {np.std(ap_all, ddof=1):.4f}")
    print(f"CV precision_macro : {np.mean(prec_all):.4f} ± {np.std(prec_all, ddof=1):.4f}")
    print(f"CV recall_macro    : {np.mean(rec_all):.4f} ± {np.std(rec_all, ddof=1):.4f}")
    print(f"TEST accuracy      : {np.mean(test_accs):.4f} (± {np.std(test_accs, ddof=1):.4f})")
    print(f"TEST roc_auc       : {np.mean(test_rocs):.4f} (± {np.std(test_rocs, ddof=1):.4f})")
    print(f"TEST f1_macro      : {np.mean(test_f1m):.4f} (± {np.std(test_f1m, ddof=1):.4f})")
    print(f"TEST pr_auc        : {np.mean(test_prs):.4f} (± {np.std(test_prs, ddof=1):.4f})")
    print(f"TEST precision_macro: {np.mean(test_prec_m):.4f} (± {np.std(test_prec_m, ddof=1):.4f})")
    print(f"TEST recall_macro   : {np.mean(test_rec_m):.4f} (± {np.std(test_rec_m, ddof=1):.4f})")
    try:
        mean_cm = np.mean(np.stack(cms, axis=0), axis=0)
        print("Mean confusion matrix (over fits):")
        print(np.round(mean_cm, 2))
    except Exception:
        pass
    print("----------------------------------")

    # Final Refit auf allen Analyse-Daten
    final_model = clone(pl_template).fit(Xa, ya.values.ravel())
    ads = "_".join(map(str, sorted(analysis_datasets)))
    tds = "_".join(map(str, sorted(test_datasets)))
    out_model = f"results1/models/{label}_{scoring}_ds_analysis_{ads}_ds_test_{tds}_signal_types_{sigs_print}{bsl_tag}.pkl"
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(final_model, out_model, compress=1)
    print(f"[SAVED] Final refit model -> {out_model}")
