import os
import sys
import pandas as pd

# --- Paths & Imports ---
# Expects constants.py with BASE_DIR and OUTPUT_DIR
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import constants as const
    BASE_DIR = const.BASE_DIR
    OUTPUT_DIR = const.OUTPUT_DIR
except Exception:
    BASE_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(os.getcwd(), "agg_data")

# --- Directories ---
base_dir = os.path.join(BASE_DIR, "dataset3_MAUS", "Data", "Raw_data")
output_folder = os.path.join(OUTPUT_DIR, "dataset3")
os.makedirs(output_folder, exist_ok=True)

PARTICIPANTS = range(1, 26)

# --- NASA-TLX Settings ---
ALL_SCALES = [
    "Mental Demand",
    "Physical Demand",
    "Temporal Demand",
    "Performance",
    "Effort",
    "Frustration",
]
# NASA default: invert Performance (100 - raw). Your version: no inversion.
INVERT_PERFORMANCE = False


def process_labels():
    """
    Read each participant’s NASA_TLX.csv, use all 6 scales with weights,
    compute weighted workload and derive a well-being label per trial.
    """
    source_folder = os.path.join(BASE_DIR, "dataset3_MAUS", "Subjective_rating")
    task_mapping = {
        "Trial 1: 0_back": 1,
        "Trial 2: 2_back": 2,
        "Trial 3: 3_back": 3,
        "Trial 4: 2_back": 4,
        "Trial 5: 3_back": 5,
        "Trial 6: 0_back": 6,
    }
    results = []

    if not os.path.isdir(source_folder):
        print(f"Warning: folder not found: {source_folder}")
        return None

    for pid in os.listdir(source_folder):
        folder = os.path.join(source_folder, pid)
        if not os.path.isdir(folder):
            continue

        csv_path = os.path.join(folder, "NASA_TLX.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = df[df["Scale Title"].isin(ALL_SCALES)].copy()
        if df.empty:
            continue

        # Read weights, fallback = 1.0
        weights = (
            df.set_index("Scale Title")["Weight"].astype(float)
            if "Weight" in df.columns
            else pd.Series(1.0, index=ALL_SCALES)
        ).reindex(ALL_SCALES).fillna(0.0)

        # Identify trial columns
        trial_cols = [c for c in df.columns if c in task_mapping]
        if not trial_cols:
            continue

        df_trials = (
            df.set_index("Scale Title")[trial_cols]
            .apply(pd.to_numeric, errors="coerce")
            .reindex(ALL_SCALES)
        )

        if INVERT_PERFORMANCE and "Performance" in df_trials.index:
            df_trials.loc["Performance"] = 100 - df_trials.loc["Performance"]

        # Weighted workload per trial
        wsum = weights.sum() or 1.0
        weighted_workload = df_trials.mul(weights, axis=0).sum(axis=0) / wsum

        for trial_name, ww in weighted_workload.items():
            task = task_mapping.get(trial_name)
            if task is None or pd.isna(ww):
                continue

            wb_index = 100.0 - float(ww)
            well_being = 1 if wb_index > 70 else 0

            try:
                participant_num = int(str(pid).lstrip("0") or "0")
            except ValueError:
                participant_num = None

            row = {
                "Participant": participant_num,
                "Task": task,
                "Well-being": well_being,
                "Well-being Index": round(wb_index, 2),
                "Weighted Workload": round(float(ww), 2),
            }

            # Optional raw scale values
            for s in ALL_SCALES:
                val = df_trials.loc[s, trial_name]
                row[s] = None if pd.isna(val) else float(val)

            results.append(row)

    if results:
        df_out = pd.DataFrame(results).sort_values(["Participant", "Task"]).reset_index(drop=True)
        return df_out
    return None


def main():
    print("Processing labels...")
    labels_df = process_labels()
    if labels_df is not None:
        out_path = os.path.join(output_folder, "labels.csv")
        labels_df.to_csv(out_path, index=False)
        print(f"Labels saved to {out_path}")
    else:
        print("No labels found.")


if __name__ == "__main__":
    main()

    labels_path = os.path.join(output_folder, "labels.csv")
    if os.path.exists(labels_path):
        df = pd.read_csv(labels_path)
        print("Well-being label counts:")
        print(df["Well-being"].value_counts(dropna=False))
