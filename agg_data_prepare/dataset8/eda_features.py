import os
import sys
import math
import pandas as pd
import neurokit2 as nk

# ---- Projektpfade ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# ---- Verzeichnisse ----
input_dir = os.path.join(const.BASE_DIR, "dataset8_POPANE")
output_file = os.path.join("agg_data", "dataset8", "eda_features.csv")
summary_file = os.path.join("agg_data", "dataset8", "eda_slice_summary.csv")
baseline_dir = os.path.join(input_dir, "Baselines")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ---- Sampling & Sliding Window ----
SAMPLING_RATE = 1000
window_length_sec = 72
step_sec = 20
window_length_samples = window_length_sec * SAMPLING_RATE  # 72 * 1000 = 72000
step_samples = step_sec * SAMPLING_RATE                    # 20 * 1000 = 20000

data_list = []
summary_list = []

# ---- Dateien durchlaufen ----
for stimuli_folder in os.listdir(input_dir):
    if stimuli_folder == "Baselines" or "Neutral" in stimuli_folder:
        continue
    stimuli_path = os.path.join(input_dir, stimuli_folder)
    if not os.path.isdir(stimuli_path):
        continue

    for file_name in os.listdir(stimuli_path):
        if "Neutral" in file_name or not file_name.endswith(".csv"):
            continue

        parts = file_name.split("_")
        if len(parts) < 3:
            continue
        task = parts[0][1:]
        participant = parts[1][1:]

        # ---- Baseline laden ----
        baseline_file_name = f"S{task}_P{participant}_Baseline.csv"
        baseline_file_path = os.path.join(baseline_dir, baseline_file_name)
        baseline_features = {}

        if os.path.exists(baseline_file_path):
            try:
                with open(baseline_file_path, "r") as f:
                    lines = f.readlines()
                data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
                baseline_data = pd.read_csv(baseline_file_path, skiprows=data_start_idx)
                if "EDA" in baseline_data.columns:
                    baseline_signal = baseline_data["EDA"].dropna().values
                    if len(baseline_signal) >= 15:
                        processed_baseline, _ = nk.eda_process(baseline_signal, sampling_rate=SAMPLING_RATE)
                        analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")
                        baseline_features = {f"Baseline_{k}": v for k, v in extract_scalar_features(analyzed_baseline).items()}
            except Exception as e:
                print(f"Error processing baseline file {baseline_file_path}: {e}")

        if not baseline_features:
            print(f"⚠️ Baseline features missing for participant {participant} in task {task}. Skipping…")
            continue

        # ---- Emotion-File laden ----
        emotion_file_path = os.path.join(stimuli_path, file_name)
        try:
            with open(emotion_file_path, "r") as f:
                lines = f.readlines()
            data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
            emotion_data = pd.read_csv(emotion_file_path, skiprows=data_start_idx)

            if "EDA" not in emotion_data.columns:
                continue
            emotion_signal = emotion_data["EDA"].dropna().values
            if len(emotion_signal) < 15:
                continue

            # Marker holen, falls vorhanden
            emotion_data.columns = emotion_data.columns.str.strip()
            first_marker = (emotion_data["marker"].dropna().iloc[0]
                            if "marker" in emotion_data.columns and not emotion_data["marker"].dropna().empty
                            else None)

            # ---- Slice-Anzahl berechnen ----
            if len(emotion_signal) >= window_length_samples and step_samples > 0:
                total_slices = ((len(emotion_signal) - window_length_samples) // step_samples) + 1
            else:
                total_slices = 0

            skip_n = math.ceil(0.4 * total_slices)  # erste 40% überspringen
            kept_slices = max(total_slices - skip_n, 0)

            summary_list.append({
                "Participant": int(participant),
                "Task": int(task),
                "Total_Slices": total_slices,
                "Skipped_First_20pct": skip_n,
                "Kept_Slices": kept_slices
            })

            # ---- Features je Slice ----
            slice_idx = 1
            for start_idx in range(0, len(emotion_signal) - window_length_samples + 1, step_samples):
                end_idx = start_idx + window_length_samples

                if slice_idx <= skip_n:  # überspringen
                    slice_idx += 1
                    continue

                segment = emotion_signal[start_idx:end_idx]
                processed_emotion, _ = nk.eda_process(segment, sampling_rate=SAMPLING_RATE)
                analyzed_emotion = nk.eda_analyze(processed_emotion, sampling_rate=SAMPLING_RATE, method="interval-related")
                emotion_features = extract_scalar_features(analyzed_emotion)

                combined_features = {
                    "Participant": int(participant),
                    "Task": int(task),
                    "Marker": first_marker,
                    "Slice": slice_idx,

                }
                combined_features.update(emotion_features)
                combined_features.update(baseline_features)
                data_list.append(combined_features)
                slice_idx += 1

        except Exception as e:
            print(f"Error processing emotion file {emotion_file_path}: {e}")
            continue

# ---- Ergebnisse speichern ----
if data_list:
    results_df = pd.DataFrame(data_list)
    results_df = results_df.sort_values(by=["Participant", "Task"]).reset_index(drop=True)
    results_df = results_df[["Participant", "Task", "Marker", "Slice"] +
                            [c for c in results_df.columns if c not in ["Participant", "Task", "Marker", "Slice"
                                                                       ]]]
    results_df.to_csv(output_file, index=False)
    print(f"✅ EDA results saved to '{output_file}'")

if summary_list:
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(summary_file, index=False)
    print(f"📊 Slice summary saved to '{summary_file}'")
