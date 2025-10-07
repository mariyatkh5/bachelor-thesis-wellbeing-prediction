import os, sys, json, math, platform, logging, ast, re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import naiveautoml
from utils.feature_loader import load_data

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline as SkPipeline

# ---------- Matplotlib backend (safe for headless) ----------
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 20})
else:
    matplotlib.use('Agg')

# ---------- Small I/O helpers ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

# ---------- Robust ROC-AUC scorer (binary & multiclass macro-OVR) ----------
def _scores_for_auc(est, X, y):

    y = np.asarray(y)
    K = len(np.unique(y))
    proba = None
    try:
        proba = est.predict_proba(X)
    except Exception:
        pass
    if proba is None:
        try:
            proba = est.decision_function(X)
        except Exception:
            return None, K

    # binary
    if K == 2:
        if hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1], K  # positive-class probability
        return np.ravel(proba), K

    # multiclass
    if not (hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] >= K):
        return None, K
    return proba, K

def _roc_auc_callable(est, X, y):
    """
    Callable passed to NaiveAutoML:
      - binary: standard ROC-AUC
      - multiclass: macro-OVR ROC-AUC
    """
    scores, K = _scores_for_auc(est, X, y)
    if scores is None or K < 2:
        return float("nan")
    y = np.asarray(y)
    if K == 2:
        return roc_auc_score(y, np.ravel(scores))
    return roc_auc_score(y, scores, multi_class="ovr", average="macro")

# ---------- External reporting metrics ----------
def _predict_scores_full(model, X):
    """
    Return model scores for reporting:
      - If predict_proba exists → return it (binary: (n,) or (n,2); multi: (n,K))
      - Else use decision_function
      - Else None
    """
    try:
        return model.predict_proba(X)
    except Exception:
        try:
            return model.decision_function(X)
        except Exception:
            return None

def evaluate_metric(y_true, y_pred, y_score, name: str, mode: str):
    """
    Compute metrics for reporting:
      - accuracy
      - roc_auc:  binary classic; multiclass macro-OVR (needs (n,K))
      - f1:       binary 'binary', multiclass 'macro'
    """
    y_true = np.asarray(y_true)
    n_classes = len(np.unique(y_true))

    if name == "accuracy":
        return accuracy_score(y_true, y_pred) if y_pred is not None else math.nan

    if name == "roc_auc":
        if y_score is None or n_classes < 2:
            return math.nan
        if mode == "binary" or n_classes == 2:
            s = y_score
            if hasattr(s, "ndim") and s.ndim == 2 and s.shape[1] > 1:
                s = s[:, 1]
            else:
                s = np.ravel(s)
            return roc_auc_score(y_true, s)
        # multiclass
        if not (hasattr(y_score, "ndim") and y_score.ndim == 2 and y_score.shape[1] >= n_classes):
            return math.nan
        return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

    if name == "f1":
        if y_pred is None or n_classes < 2:
            return math.nan
        avg = "binary" if (mode == "binary" or n_classes == 2) else "macro"
        return f1_score(y_true, y_pred, average=avg)

    raise ValueError(name)

# ---------- Group utilities ----------
def _parse_group_value(v):
    if isinstance(v, tuple) and len(v) == 2: return str(v[0]), str(v[1])
    if isinstance(v, str):
        if '|' in v:
            left, right = v.split('|', 1); return left.strip(), right.strip()
        try:
            t = ast.literal_eval(v)
            if isinstance(t, tuple) and len(t) == 2: return str(t[0]), str(t[1])
        except Exception: pass
    return str(v), ""

def _to_group_labels(groups):
    if isinstance(groups, pd.DataFrame) and {'Participant','Task'}.issubset(groups.columns):
        return (groups['Participant'].astype(str) + '|' + groups['Task'].astype(str)).values
    s = pd.Series(groups)
    return np.array([f"{_parse_group_value(v)[0]}|{_parse_group_value(v)[1]}" for v in s])

def _rows_for_groups(trial, fold, phase, split_name, labels_iter):
    items = [] if labels_iter is None else (list(labels_iter) if not isinstance(labels_iter,(str,bytes)) else [labels_iter])
    uniq, seen = [], set()
    for x in items:
        k = str(x)
        if k not in seen: seen.add(k); uniq.append(x)
    out = []
    for lab in uniq:
        p,t = (_parse_group_value(lab) if not (isinstance(lab, tuple) and len(lab)==2) else (str(lab[0]), str(lab[1])))
        out.append({"trial": trial, "fold": int(fold), "phase": phase,
                    "split": split_name, "group": f"{p}|{t}", "participant": p, "task": t})
    return out

# ---------- Pipeline helpers ----------
def _extract_picklable_pipeline(naml_obj):
    """Find the actual sklearn Pipeline inside a NaiveAutoML object."""
    for attr in ("pipeline_", "best_pipeline_", "final_pipeline_", "pipeline"):
        if hasattr(naml_obj, attr):
            pipe = getattr(naml_obj, attr)
            if hasattr(pipe, "predict"): return pipe
    return naml_obj  # fallback

def _pipeline_to_config_row(pipe: SkPipeline, scoring_name: str, scoring_value: float):
    """Save a one-row config descriptor for later reproducibility/debugging."""
    pipe_str = repr(pipe)
    pipe_str = re.sub(r"<function chi2[^>]*>", "f_classif", pipe_str)  # avoid non-picklable repr bits

    data_pp = None; feat_pp = None; learner = None
    if isinstance(pipe, SkPipeline):
        for _, step in pipe.steps:
            cls = step.__class__
            full = f"{cls.__module__}.{cls.__name__}"
            if data_pp is None:   data_pp = full
            elif feat_pp is None: feat_pp = full
            learner = full

    row = {
        "pipeline": pipe_str,
        "data-pre-processor_class": data_pp or "",
        "feature-pre-processor_class": feat_pp or "",
        "learner_class": learner or "",
        scoring_name: scoring_value
    }
    if scoring_name != "accuracy": row["accuracy"] = np.nan
    if scoring_name != "roc_auc":  row["roc_auc"]  = np.nan
    if scoring_name != "f1":       row["f1"]       = np.nan
    return row

def fmt_mean_std(values):
    """Format mean ± std nicely; ignores NaNs."""
    vals = [v for v in values if isinstance(v,(int,float)) and not math.isnan(v)]
    if not vals: return (math.nan, math.nan, "NaN")
    m = float(np.mean(vals)); s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
    return (m, s, f"{m:.4f} (±{s:.4f})")

# ---------- Config parsing ----------
def _load_hyper_from_argv():
    """
    Accepts either:
      1) Inline JSON, e.g.: '{"signal_types":["eda"],"analysis_datasets":[1]}'
      2) Path to JSON file, e.g.: config.json
    """
    if len(sys.argv) <= 1: return None
    arg = sys.argv[1]
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f: return json.load(f)
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        try:
            return json.loads(arg.replace("'", '"'))
        except Exception as e:
            raise SystemExit("Invalid configuration: provide inline JSON or a path to a JSON file.") from e

# ---------- NaiveAutoML builder (ROC-AUC primary) ----------
def build_automl_rocauc(random_state: int, mccv_splits: int = 30, num_cpus: int = 1):
    kwargs_common = dict(
        max_hpo_iterations=1024,
        max_hpo_iterations_without_imp=100,
        max_hpo_time_without_imp=0.1100 * 65,
        timeout_candidate=10,
        scoring=("roc_auc", _roc_auc_callable),  # (name, callable) to avoid 'needs_proba' issues
        num_cpus=num_cpus,
        show_progress=True,
        random_state=random_state,
        kwargs_as={'excluded_components': {"learner": ["MultinomialNB", "HistGradientBoostingClassifier"]}},
    )

    # Try the API variant seen in some versions (your snippet)
    try:
        return naiveautoml.NaiveAutoML(
            evaluation_metric="mccv",
            kwargs_evaluation_function={"n_splits": int(mccv_splits)},
            **kwargs_common
        )
    except TypeError:
        # Fall back to alternative naming
        return naiveautoml.NaiveAutoML(
            evaluation_fun="mccv",
            kwargs_evaluation_fun={"n_splits": int(mccv_splits)},
            **kwargs_common
        )

# ---------- Main ----------
if __name__ == "__main__":
    # Logging
    logger = logging.getLogger('naiveautoml')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # Settings (with sensible defaults)
    hyper = _load_hyper_from_argv()
    if hyper:
        signal_types       = hyper.get("signal_types", ["eda"])
        analysis_datasets  = hyper.get("analysis_datasets", [1])
        label_name         = hyper.get("label", "Well-being")
        bsl                = bool(hyper.get("bsl", True))
        max_configs        = int(hyper.get("max_configs", 10))
        base_seed          = int(hyper.get("random_state", 42))
        n_splits_outer     = int(hyper.get("outer_folds", 5))
        target_threshold   = float(hyper.get("target_threshold", 0.70))
        mode               = hyper.get("mode", "multiclass").lower()      # "binary" | "multiclass"
        bin_pos_ge         = int(hyper.get("bin_positive_if_ge", 1))      # threshold for binary mode
        mccv_splits        = int(hyper.get("mccv_splits", 30))            # internal MCCV splits
        num_cpus           = int(hyper.get("num_cpus", 1))                # CPUs for AutoML
    else:
        signal_types       = ["eda"]
        analysis_datasets  = [1]
        label_name         = "Well-being"
        bsl                = True
        max_configs        = 10
        base_seed          = 42
        n_splits_outer     = 5
        target_threshold   = 0.70
        mode               = "multiclass"   # or "binary"
        bin_pos_ge         = 1
        mccv_splits        = 30
        num_cpus           = 1

    dataset_key = "ds_" + "_".join(map(str, sorted(analysis_datasets)))
    signal_key  = "+".join([str(s).strip().lower() for s in signal_types])
    bsl_tag     = "_bsl" if bsl else ""

    # Base results directories
    base_results = "results"
    dir_models   = ensure_dir(os.path.join(base_results, "model",  dataset_key))
    dir_split    = ensure_dir(os.path.join(base_results, "split",  dataset_key, signal_key))
    dir_autoML   = ensure_dir(os.path.join(base_results, "autoML_classifiers", signal_key))
    dir_summary  = ensure_dir(os.path.join(base_results, "summary", signal_key))
    ensure_dir(os.path.join(base_results, "autoML_classifiers"))

    # Load data, labels, and groups
    out = load_data(
        datasets=analysis_datasets, signal_types=signal_types, bsl=bsl,
        group_by=("Participant","Task")
    )
    if isinstance(out, tuple) and len(out) == 3:
        X, y_obj, groups_raw = out
    else:
        X, y_obj = out
        groups_raw = None

    # Extract target
    if isinstance(y_obj, pd.Series):
        y_all_raw = y_obj.to_numpy().ravel()
    else:
        if hasattr(y_obj, "columns") and (label_name in y_obj.columns):
            y_all_raw = y_obj[label_name].to_numpy().ravel()
        else:
            y_all_raw = y_obj.iloc[:, 0].to_numpy().ravel()

    # Apply mode
    if mode == "binary":
        # positive class = 1 if label >= bin_pos_ge (e.g., Well-being >= 1)
        y_all = (pd.Series(y_all_raw).astype(float) >= float(bin_pos_ge)).astype(int).to_numpy()
    else:
        # multiclass: keep integer classes
        y_all = pd.Series(y_all_raw).astype(int).to_numpy()

    # Basic NaN checks
    if np.isnan(X.to_numpy(dtype=float)).any() or np.isnan(pd.Series(y_all, dtype=float)).any():
        raise ValueError("NaNs found – loader should operate in 'no-imputation' mode (clean features/labels).")

    # Groups
    group_labels = _to_group_labels(groups_raw) if groups_raw is not None else None

    # Class info
    classes = np.unique(y_all)
    K = len(classes)
    print(f"[INFO] Classes: {list(classes)} (K={K}) | mode={mode}")

    # Collectors
    summary_rows = []
    summary_confusions = {}  # config -> {"classes": [...], "mean_confusion": [[...]]}

    # Loop configs (seeds)
    for config in range(1, max_configs + 1):
        seed = base_seed + config
        trial = config

        # Config dirs
        cfg_model_dir = ensure_dir(os.path.join(dir_models, f"config{config}"))
        cfg_split_dir = ensure_dir(os.path.join(dir_split, f"config{config}"))
        cfg_auto_dir  = ensure_dir(os.path.join(dir_autoML, f"config{config}"))

        # Outer CV: stratified by labels, grouped by (Participant|Task)
        skf = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

        fold_scores_acc = []
        fold_scores_auc = []
        fold_scores_f1  = []
        cm_sum = np.zeros((K, K), dtype=float)
        cm_count = 0

        all_group_rows = []
        all_fold_log_rows = []

        for i, (tr_idx, te_idx) in enumerate(skf.split(X, y_all, group_labels), start=1):
            # Split data
            x_tr = X.iloc[tr_idx].reset_index(drop=True)
            x_te = X.iloc[te_idx].reset_index(drop=True)
            y_tr = y_all[tr_idx]
            y_te = y_all[te_idx]
            if group_labels is not None:
                g_tr = pd.Series(group_labels[tr_idx])
                g_te = pd.Series(group_labels[te_idx])
            else:
                g_tr = pd.Series([""] * len(tr_idx))
                g_te = pd.Series([""] * len(te_idx))

            # Sanity check: no leakage
            assert set(g_tr.unique()).isdisjoint(set(g_te.unique())), "Group leakage between train/test!"

            # Per-fold dirs
            fold_model_dir = ensure_dir(os.path.join(cfg_model_dir, f"fold{i}"))
            fold_split_dir = ensure_dir(os.path.join(cfg_split_dir, f"fold{i}"))
            fold_auto_dir  = ensure_dir(os.path.join(cfg_auto_dir,  f"fold{i}"))

            # Group logs
            all_group_rows += _rows_for_groups(trial, i, "outer", "train", g_tr.unique().tolist())
            all_group_rows += _rows_for_groups(trial, i, "outer", "test",  g_te.unique().tolist())
            pd.DataFrame({"group": sorted(map(str, g_te.unique()))}).to_csv(
                os.path.join(fold_auto_dir, "test_groups.csv"), index=False
            )

            # Build & fit NaiveAutoML (primary ROC-AUC with our callable)
            naml = build_automl_rocauc(random_state=seed, mccv_splits=mccv_splits, num_cpus=num_cpus)
            naml.fit(x_tr, y_tr)

            # Save per-fold AutoML history (if available)
            try:
                if hasattr(naml, "history") and naml.history is not None:
                    naml.history.to_csv(os.path.join(fold_auto_dir, "history.csv"), index=False)
            except Exception:
                pass

            # Predict on test
            y_score = _predict_scores_full(naml, x_te)   # (n,), (n,2) or (n,K)
            try:
                y_pred = naml.predict(x_te)
            except Exception:
                y_pred = None

            # Metrics
            acc = evaluate_metric(y_te, y_pred, y_score, "accuracy", mode)
            auc = evaluate_metric(y_te, y_pred, y_score, "roc_auc", mode)
            f1  = evaluate_metric(y_te, y_pred, y_score, "f1", mode)

            fold_scores_acc.append(acc)
            fold_scores_auc.append(auc)
            fold_scores_f1.append(f1)

            # Confusion (K×K) when y_pred available
            cm = None
            if y_pred is not None and len(np.unique(y_te)) >= 2:
                cm = confusion_matrix(y_te, y_pred, labels=classes)
                cm_sum += cm
                cm_count += 1

            # Per-fold: save preds
            preds_payload = {
                "y_true": y_te,
                "y_pred": y_pred if y_pred is not None else np.nan,
                "group": g_te.values
            }
            # Add scores: for binary (n,) or (n,2) → store pos-class as 'score'
            # for multiclass (n,K) → store per-class columns 'score_<class>'
            if y_score is not None:
                s = y_score
                if getattr(s, "ndim", 1) == 1:
                    preds_payload["score"] = np.ravel(s)
                else:
                    if mode == "binary" and s.shape[1] >= 2:
                        preds_payload["score"] = s[:, 1]
                    else:
                        # multiclass: store all columns
                        for j, cls in enumerate(classes):
                            preds_payload[f"score_{cls}"] = s[:, j]
            pd.DataFrame(preds_payload).to_csv(os.path.join(fold_split_dir, "preds.csv"), index=False)

            # Per-fold: save metrics.json
            metrics_payload = {
                "dataset": dataset_key,
                "signal": signal_key,
                "config": config,
                "fold": i,
                "mode": mode,
                "classes": list(map(lambda x: x.item() if hasattr(x,"item") else x, classes)),
                "metrics": {
                    "accuracy": None if (not isinstance(acc, float) or math.isnan(acc)) else acc,
                    "roc_auc":  None if (not isinstance(auc, float) or math.isnan(auc)) else auc,
                    "f1":       None if (not isinstance(f1,  float) or math.isnan(f1))  else f1,
                },
                "confusion_matrix": cm.tolist() if cm is not None else None,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx))
            }
            with open(os.path.join(fold_split_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

            # Save model pipeline + columns
            pipeline = _extract_picklable_pipeline(naml)
            try:
                from joblib import dump as joblib_dump
                joblib_dump(pipeline, os.path.join(fold_model_dir, "pipeline.pkl"), compress=1)
            except Exception:
                try:
                    import cloudpickle
                    with open(os.path.join(fold_model_dir, "pipeline.pkl"), "wb") as f:
                        cloudpickle.dump(pipeline, f)
                except Exception:
                    pass
            try:
                with open(os.path.join(fold_model_dir, "columns.json"), "w", encoding="utf-8") as f:
                    json.dump(list(x_tr.columns), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # Save one-line pipeline config with the *primary* value (AUC)
            try:
                primary_val = float(auc) if (isinstance(auc,float) and not math.isnan(auc)) else float("nan")
                row = _pipeline_to_config_row(pipeline, "roc_auc", primary_val)
                pd.DataFrame([row]).to_csv(os.path.join(fold_model_dir, "pipeline_config.csv"), index=False)
            except Exception:
                pass

            # Fold log row
            all_fold_log_rows.append({
                "trial": trial, "fold": i, "phase": "outer",
                "score_accuracy": float(acc) if (isinstance(acc,float) and not math.isnan(acc)) else math.nan,
                "score_roc_auc": float(auc) if (isinstance(auc,float) and not math.isnan(auc)) else math.nan,
                "score_f1": float(f1) if (isinstance(f1,float) and not math.isnan(f1)) else math.nan,
                "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))
            })

            # Console info
            acc_s = "NaN" if (not isinstance(acc,float) or math.isnan(acc)) else f"{acc:.4f}"
            auc_s = "NaN" if (not isinstance(auc,float) or math.isnan(auc)) else f"{auc:.4f}"
            f1_s  = "NaN" if (not isinstance(f1,float)  or math.isnan(f1))  else f"{f1:.4f}"
            print(f"[TEST] config{config} fold{i}  acc={acc_s}  auc={auc_s}  f1={f1_s}  (K={K}, mode={mode})")

        # Save large per-config logs
        pd.DataFrame(all_group_rows).to_csv(os.path.join(cfg_auto_dir, "groupfolds.csv"), index=False)
        pd.DataFrame(all_fold_log_rows).to_csv(os.path.join(cfg_auto_dir, "foldlog.csv"), index=False)

        # Aggregate mean/std and mean confusion
        acc_m, acc_sd, acc_str = fmt_mean_std(fold_scores_acc)
        auc_m, auc_sd, auc_str = fmt_mean_std(fold_scores_auc)
        f1_m,  f1_sd,  f1_str  = fmt_mean_std(fold_scores_f1)
        mean_cm = (cm_sum / cm_count).tolist() if cm_count > 0 else None
        summary_confusions[f"config{config}"] = {
            "classes": list(map(lambda x: x.item() if hasattr(x,"item") else x, classes)),
            "mean_confusion": mean_cm
        }
        summary_rows.append({
            "dataset": dataset_key,
            "signal": signal_key,
            "config": f"config{config}",
            "mode": mode,
            "accuracy_mean_std": acc_str,
            "roc_auc_mean_std": auc_str,
            "f1_mean_std": f1_str,
            "folds_valid_for_cm": cm_count
        })

        # Optional: refit on ALL data if AUC mean meets target
        prim_m = auc_m
        if isinstance(prim_m, float) and not math.isnan(prim_m) and prim_m >= target_threshold:
            print(f"[REFIT] config{config}: roc_auc mean {prim_m:.4f} ≥ target {target_threshold:.2f} → refit on ALL data")
            naml_final = build_automl_rocauc(random_state=seed, mccv_splits=mccv_splits, num_cpus=num_cpus)
            naml_final.fit(X.reset_index(drop=True), y_all)
            # Save refit history (if any)
            refit_hist_name = (
                f"naml_history_{label_name}_roc_auc"
                f"_ds_analysis_{'_'.join(map(str, sorted(analysis_datasets)))}"
                f"_signal_types_{'_'.join(signal_types)}{('_bsl' if bsl else '')}"
                f"_config{config}_refit_all.csv"
            )
            try:
                if hasattr(naml_final, "history") and naml_final.history is not None:
                    naml_final.history.to_csv(os.path.join("results", "autoML_classifiers", refit_hist_name), index=False)
                    print(f"[SAVED] Classic history (refit_all) → results/autoML_classifiers/{refit_hist_name}")
            except Exception as e:
                print(f"[WARN] Could not save classic history (refit_all): {e}")

    # Save per-signal summaries
    summary_csv_path = os.path.join(dir_summary, f"summary_{dataset_key}_{bsl_tag}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"[SAVED] Summary table → {summary_csv_path}")

    summary_cm_path = os.path.join(dir_summary, f"confusion_{dataset_key}_{bsl_tag}.json")
    with open(summary_cm_path, "w", encoding="utf-8") as f:
        json.dump(summary_confusions, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Mean confusion matrices → {summary_cm_path}")

    print("[DONE]")
