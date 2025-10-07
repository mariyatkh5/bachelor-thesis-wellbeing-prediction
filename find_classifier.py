import os, sys, json, platform, logging, ast
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import naiveautoml
from utils.feature_loader import load_data

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SkPipeline

# ---------- Backend setup ----------
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('Agg')

# ---------- Helpers ----------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def build_automl(scoring: str, random_state: int):
    return naiveautoml.NaiveAutoML(
        max_hpo_iterations=1024,
        max_hpo_iterations_without_imp=30,
        scoring=scoring,
        evaluation_fun="mccv",              # internal HPO eval only (no external test here)
        kwargs_evaluation_fun={"n_splits": 30},
        num_cpus=1,
        show_progress=True,
        random_state=random_state,
        kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier","MultinomialNB"]}}
    )


def _parse_group_value(v):
    if isinstance(v, tuple) and len(v) == 2:
        return str(v[0]), str(v[1])
    if isinstance(v, str):
        if '|' in v:
            left, right = v.split('|', 1)
            return left.strip(), right.strip()
        try:
            t = ast.literal_eval(v)
            if isinstance(t, tuple) and len(t) == 2:
                return str(t[0]), str(t[1])
        except Exception:
            pass
    return str(v), ""


def _to_group_labels(groups):
    if isinstance(groups, pd.DataFrame) and {'Participant','Task'}.issubset(groups.columns):
        return (groups['Participant'].astype(str) + '|' + groups['Task'].astype(str)).values
    s = pd.Series(groups)
    return np.array([f"{_parse_group_value(v)[0]}|{_parse_group_value(v)[1]}" for v in s])


if __name__ == "__main__":
    # Logging
    logger = logging.getLogger('naiveautoml')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # Settings (CLI JSON overrides defaults)
    if len(sys.argv) > 1:
        hyper = json.loads(sys.argv[1])
        signal_types       = hyper["signal_types"]            # ["eda"], ["ecg"], ...
        analysis_datasets  = hyper["analysis_datasets"]       # [1], ...
        scoring_primary    = hyper["scoring"]                 # "roc_auc" | "accuracy" | ...
        label_name         = hyper.get("label", "target")
        bsl                = hyper.get("bsl", False)
        max_configs        = int(hyper.get("max_configs", 3))
        base_seed          = int(hyper.get("random_state", 42))
        n_splits_outer     = int(hyper.get("outer_folds", 5))
    else:
        signal_types       = ["eda","ecg"]
        analysis_datasets  = [2]
        scoring_primary    = "roc_auc"
        label_name         = "Well-being"
        bsl                = False
        max_configs        = 1
        base_seed          = 42
        n_splits_outer     = 5

    dataset_key = "ds_" + "_".join(map(str, sorted(analysis_datasets)))
    signal_key  = "+".join([str(s).strip().lower() for s in signal_types])
    bsl_tag     = "_bsl" if bsl else ""

    # Directory layout (no timestamps)
    base_results = "results"
    dir_models   = ensure_dir(os.path.join(base_results, "model",  dataset_key))
    dir_autoML   = ensure_dir(os.path.join(base_results, "autoML_classifiers", signal_key))
    ensure_dir(os.path.join(base_results, "autoML_classifiers"))  # for classic history files

    # Load data, labels, and groups
    X, y_obj, groups_raw = load_data(
        datasets=analysis_datasets, signal_types=signal_types, bsl=bsl,
        group_by=("Participant","Task")
    )

    # Extract label
    if isinstance(y_obj, pd.Series):
        y_all = y_obj.to_numpy().ravel()
    else:
        if label_name in y_obj.columns:
            y_all = y_obj[label_name].to_numpy().ravel()
        elif label_name.lower() == "stress" and "Well-being" in y_obj.columns:
            y_all = (1 - y_obj["Well-being"].astype(float)).to_numpy().ravel()
        else:
            y_all = y_obj.iloc[:, 0].to_numpy().ravel()

    # Hard stop if NaNs (clean/impute upstream)
    if np.isnan(y_all).any() or np.isnan(X.to_numpy(dtype=float)).any():
        raise ValueError("NaNs in data – please impute/clean first.")

    group_labels = _to_group_labels(groups_raw)

    # -------- Multiple configs (different seeds) --------
    for config in range(1, max_configs + 1):
        seed = base_seed + config   # seed per config → +1 per config
        cfg_key = f"config{config}"

        # Paths for this config
        cfg_model_dir = ensure_dir(os.path.join(dir_models, cfg_key))
        cfg_auto_dir  = ensure_dir(os.path.join(dir_autoML, cfg_key))

        # Group logging across folds (TRAIN groups only)
        all_group_rows = []

        # Outer CV used only to create different TRAIN subsets
        skf = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)

        for i, (tr_idx, te_idx_unused) in enumerate(skf.split(X, y_all, group_labels), start=1):
            x_tr = X.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_all[tr_idx]
            g_tr = pd.Series(group_labels[tr_idx])

            # Which groups were used in training
            uniq_train_groups = sorted(map(str, g_tr.unique()))
            all_group_rows += [{"config": cfg_key, "fold": i, "phase": "train", "group": g} for g in uniq_train_groups]

            # Per-fold dirs
            fold_model_dir = ensure_dir(os.path.join(cfg_model_dir, f"fold{i}"))
            fold_auto_dir  = ensure_dir(os.path.join(cfg_auto_dir,  f"fold{i}"))

            # Train NaiveAutoML
            naml = build_automl(scoring_primary, random_state=seed + (i-1))  # per-fold seed: base+config+(fold-1)
            logger.info(f"[TRAIN] {cfg_key} fold{i}: seed={seed+(i-1)} | n_train={len(tr_idx)}")
            naml.fit(x_tr, y_tr)

            # Save per-fold history (inside fold dir)
            try:
                if hasattr(naml, "history") and naml.history is not None:
                    naml.history.to_csv(os.path.join(fold_auto_dir, "history.csv"), index=False)
            except Exception:
                pass

            # Also write a classic history file (global location) with config & fold
            classic_hist_name = (
                f"naml_history_{label_name}_{scoring_primary}"
                f"_ds_analysis_{'_'.join(map(str, sorted(analysis_datasets)))}"
                f"_signal_types_{'_'.join(signal_types)}{bsl_tag}"
                f"_config{config}_fold{i}.csv"
            )
            try:
                if hasattr(naml, "history") and naml.history is not None:
                    naml.history.to_csv(os.path.join("results", "autoML_classifiers", classic_hist_name), index=False)
                    print(f"[SAVED] Classic history → results/autoML_classifiers/{classic_hist_name}")
            except Exception as e:
                print(f"[WARN] Could not save classic history (fold): {e}")

            # Save model per fold (pipeline + columns)
            pipeline = getattr(naml, "chosen_model", None)
            if pipeline is None:
                # Fallback attribute names used by some implementations
                for attr in ("pipeline_", "best_pipeline_", "final_pipeline_", "pipeline"):
                    if hasattr(naml, attr):
                        pipeline = getattr(naml, attr)
                        break
                if pipeline is None:
                    pipeline = naml

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

        # Groups across folds for this config (TRAIN only)
        pd.DataFrame(all_group_rows).to_csv(os.path.join(cfg_auto_dir, "groupfolds.csv"), index=False)

    print("[DONE] Train-only: configs × folds → models & histories saved.")
