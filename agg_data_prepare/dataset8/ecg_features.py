import os
import pandas as pd
import neurokit2 as nk
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# Base directories
input_dir = os.path.join(const.BASE_DIR, "dataset8_POPANE")
output_file_ecg = os.path.join(const.OUTPUT_DIR, "dataset8/ecg_features.csv")
baseline_dir = os.path.join(input_dir, "Baselines")
SAMPLING_RATE = 1000
window_length = int(const.INTERVAL * SAMPLING_RATE)
step = int(const.STEP * SAMPLING_RATE)

def process_ecg_signals():
    os.makedirs(os.path.dirname(output_file_ecg), exist_ok=True)
    data_list = []

    for stimuli_folder in os.listdir(input_dir):
        if "Neutral" in stimuli_folder or stimuli_folder == "Baselines":
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

            # Process baseline file
            baseline_file_name = f"S{task}_P{participant}_Baseline.csv"
            baseline_file_path = os.path.join(baseline_dir, baseline_file_name)
            baseline_features = {}

            if os.path.exists(baseline_file_path):
                try:
                    with open(baseline_file_path, "r") as f:
                        lines = f.readlines()
                    data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
                    baseline_data = pd.read_csv(baseline_file_path, skiprows=data_start_idx)
                    if "ECG" not in baseline_data.columns:
                        print(f"No 'ECG' column in baseline file {baseline_file_path}. Skipping baseline.")
                    else:
                        baseline_signal = baseline_data["ECG"].dropna().values
                        if len(baseline_signal) >= 15:
                            processed_baseline, _ = nk.ecg_process(baseline_signal, sampling_rate=SAMPLING_RATE)
                            analyzed_baseline = nk.ecg_analyze(
                                processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related"
                            )
                            baseline_features = {
                                f"Baseline_{key}": val
                                for key, val in extract_scalar_features(analyzed_baseline).items()
                            }
                        else:
                            print(f"Baseline signal too short in {baseline_file_path}. Skipping baseline.")
                except Exception as e:
                    print(f"Error processing baseline file {baseline_file_path}: {e}")
            else:
                print(f"Baseline file not found: {baseline_file_name}")

            if not baseline_features:
                print(f"Baseline features missing for participant {participant} in task {task}. Skipping...")
                continue

            # Process emotion file (windowed)
            emotion_file_path = os.path.join(stimuli_path, file_name)
            try:
                with open(emotion_file_path, "r") as f:
                    lines = f.readlines()
                data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
                emotion_data = pd.read_csv(emotion_file_path, skiprows=data_start_idx)

                if "ECG" not in emotion_data.columns:
                    print(f"No 'ECG' column in {emotion_file_path}. Skipping.")
                    continue

                emotion_signal = emotion_data["ECG"].dropna().values
                if len(emotion_signal) < window_length:
                    print(f"ECG signal too short in {emotion_file_path}. Skipping.")
                    continue

                emotion_data.columns = emotion_data.columns.str.strip()
                first_marker = (
                    emotion_data["marker"].dropna().iloc[0]
                    if "marker" in emotion_data.columns and not emotion_data["marker"].dropna().empty
                    else None
                )

                slice_count = 1
                start_idx = 0
                total_samples = len(emotion_signal)

                while start_idx + window_length <= total_samples:
                    window_signal = emotion_signal[start_idx:start_idx + window_length]
                    if len(window_signal) < 15:
                        start_idx += step
                        slice_count += 1
                        continue

                    processed_emotion, _ = nk.ecg_process(window_signal, sampling_rate=SAMPLING_RATE)
                    analyzed_emotion = nk.ecg_analyze(processed_emotion, sampling_rate=SAMPLING_RATE, method="interval-related")
                    emotion_features = extract_scalar_features(analyzed_emotion)

                    combined_features = {
                        "Participant": int(participant),
                        "Task": int(task),
                        "Marker": first_marker,
                        "Slice": slice_count
                    }
                    combined_features.update(emotion_features)
                    combined_features.update(baseline_features)
                    data_list.append(combined_features)

                    start_idx += step
                    slice_count += 1

            except Exception as e:
                print(f"Error processing emotion file {emotion_file_path}: {e}")
                continue

    # Save results
    if data_list:
        results_df = pd.DataFrame(data_list)
        output_df = results_df.sort_values(by=["Participant", "Task", "Slice"]).reset_index(drop=True)
        output_df.to_csv(output_file_ecg, index=False)
        print(f"ECG results saved to '{output_file_ecg}'!")
    else:
        print("No data processed.")

# Run ECG processing
process_ecg_signals()