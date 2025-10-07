import os
import sys
import math
import warnings
import pandas as pd
import neurokit2 as nk

warnings.filterwarnings("ignore")

# ---- Projektpfade & Utils ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# ---- Verzeichnisse ----
base_dir = os.path.join(const.BASE_DIR, "dataset3_MAUS/Data/Raw_data")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset3")
os.makedirs(output_folder, exist_ok=True)

features_out = os.path.join(output_folder, "gsr_features.csv")
summary_out  = os.path.join(output_folder, "gsr_slice_summary.csv")

participants = range(1, 26)  # P01..P25
results = []
summary_rows = []

for participant in participants:
    # 128 Hz für P01, sonst 256 Hz
    sampling_rate = 128 if participant == 1 else 256

    # Fenster/Schritt in SAMPLES (auf ganze int runden)
    window_length_samples = int(const.INTERVAL * sampling_rate)
    step_samples = int(const.STEP * sampling_rate)

    participant_dir = os.path.join(base_dir, f"{participant:03}")
    resting_file = os.path.join(participant_dir, "inf_resting.csv")
    task_file = os.path.join(participant_dir, "inf_gsr.csv")

    if not os.path.exists(resting_file):
        print(f"Resting file not found: {resting_file}")
        continue
    if not os.path.exists(task_file):
        print(f"Task file not found: {task_file}")
        continue

    try:
        # ---- Baseline verarbeiten ----
        resting_data = pd.read_csv(resting_file)
        if "Resting_GSR" not in resting_data.columns:
            print(f"'Resting_GSR' column not found for participant {participant}. Skipping.")
            continue

        baseline_signal = pd.to_numeric(resting_data["Resting_GSR"].dropna(), errors="coerce").values
        if len(baseline_signal) < 15:
            print(f"Baseline signal too short for participant {participant}. Skipping.")
            continue

        processed_baseline, _ = nk.eda_process(baseline_signal, sampling_rate=sampling_rate)
        analyzed_baseline = nk.eda_analyze(
            processed_baseline,
            sampling_rate=sampling_rate,
            method="interval-related"
        )
        baseline_features = {
            f"Baseline_{k}": v for k, v in extract_scalar_features(analyzed_baseline).items()
        }

        # ---- Taskdaten spaltenweise verarbeiten ----
        task_data = pd.read_csv(task_file)

        for idx, column in enumerate(task_data.columns, start=1):
            # Signal extrahieren (nicht-numerisches wird zu NaN -> dropna)
            signal = pd.to_numeric(task_data[column].dropna(), errors="coerce").values

            # Slice-Anzahl bestimmen
            if len(signal) >= window_length_samples and step_samples > 0:
                total_slices = ((len(signal) - window_length_samples) // step_samples) + 1
            else:
                total_slices = 0

            # Erste 40% Slices (aufgerundet) überspringen
            skip_n = math.ceil(0.4 * total_slices)
            kept_slices = max(total_slices - skip_n, 0)

            # Übersicht erfassen (auch wenn Signal zu kurz ist -> total_slices==0)
            summary_rows.append({
                "Participant": participant,
                "Task": idx,
                "Column": column,
                "Total_Slices": total_slices,
                "Skipped_First_20pct": skip_n,
                "Kept_Slices": kept_slices
            })

            if total_slices == 0:
                print(f"Task signal too short in column {column} for participant {participant}. Skipping.")
                continue

            # ---- Sliding Window & Features (ab Slice skip_n+1) ----
            slice_idx = 1
            for start_idx in range(0, len(signal) - window_length_samples + 1, step_samples):
                end_idx = start_idx + window_length_samples

                # überspringe die ersten 20%
                if slice_idx <= skip_n:
                    slice_idx += 1
                    continue

                window_segment = signal[start_idx:end_idx]
                processed_task, _ = nk.eda_process(window_segment, sampling_rate=sampling_rate)
                analyzed_task = nk.eda_analyze(
                    processed_task,
                    sampling_rate=sampling_rate,
                    method="interval-related"
                )
                task_features = extract_scalar_features(analyzed_task)

                combined = {
                    "Participant": participant,
                    "Task": idx,
                    "Slice": slice_idx,

                }
                combined.update(task_features)
                combined.update(baseline_features)
                results.append(combined)

                slice_idx += 1

    except Exception as e:
        print(f"Error processing data for participant {participant}: {e}")

# ---- Dateien schreiben ----
if results:
    df = pd.DataFrame(results)
    cols_front = ["Participant", "Task", "Slice"]
    df = df[cols_front + [c for c in df.columns if c not in cols_front]]
    df.to_csv(features_out, index=False)
    print(f"GSR features saved to {features_out}")
else:
    print("No features extracted. Please check input data.")

if summary_rows:
    sm = pd.DataFrame(summary_rows)
    sm = sm.sort_values(["Participant", "Task", "Column"]).reset_index(drop=True)
    sm.to_csv(summary_out, index=False)
    print(f"Slice summary saved to {summary_out}")
