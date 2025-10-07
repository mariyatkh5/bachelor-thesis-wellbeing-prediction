import pandas as pd
import neurokit2 as nk
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# Base directories
base_dir = os.path.join(const.BASE_DIR, "dataset3_MAUS/Data/Raw_data")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset3")
os.makedirs(output_folder, exist_ok=True)

# Sampling rates
SAMPLING_RATE = {
    "ppg": 256,
    "pixart": 100,
}

PARTICIPANTS = range(1, 26)

# Window settings 
def get_window_settings(signal_type):
    sampling_rate = SAMPLING_RATE[signal_type]
    window_length_samples = int(const.INTERVAL * sampling_rate)
    step_samples = int(const.STEP * sampling_rate)
    return window_length_samples, step_samples

def process_signal(participant, participant_dir, signal_type, resting_file, resting_column, task_file_name):
    results = []

    resting_path = os.path.join(participant_dir, resting_file)
    task_path = os.path.join(participant_dir, task_file_name)

    if not os.path.exists(resting_path):
        print(f"Resting file not found for {signal_type}, Participant {participant}: {resting_path}")
        return results
    if not os.path.exists(task_path):
        print(f"Task file not found for {signal_type}, Participant {participant}: {task_path}")
        return results

    try:
        sampling_rate = SAMPLING_RATE[signal_type]
        window_length_samples, step_samples = get_window_settings(signal_type)

        # Baseline
        resting_data = pd.read_csv(resting_path)
        if resting_column not in resting_data.columns:
            print(f"Column '{resting_column}' not found in {resting_path}")
            return results

        baseline_signal = resting_data[resting_column].dropna().values
        if len(baseline_signal) < 15:
            print(f"Baseline signal too short for Participant {participant}, {signal_type}. Skipping.")
            return results

        processed_baseline, _ = nk.ppg_process(baseline_signal, sampling_rate=sampling_rate)
        analyzed_baseline = nk.ppg_analyze(processed_baseline, sampling_rate=sampling_rate, method="interval-related")

        baseline_features_raw = extract_scalar_features(analyzed_baseline)
        baseline_features_raw.pop("Slice", None)
        baseline_features = {f"Baseline_{key}": val for key, val in baseline_features_raw.items()}

        # Task
        task_data = pd.read_csv(task_path)
        for idx, column in enumerate(task_data.columns, start=1):
            signal = task_data[column].dropna().values

            if len(signal) < window_length_samples:
                print(f"Task signal too short in column {column} for Participant {participant}, {signal_type}. Skipping.")
                continue

            slice_count = 1
            for start_idx in range(0, len(signal) - window_length_samples + 1, step_samples):
                end_idx = start_idx + window_length_samples
                window_segment = signal[start_idx:end_idx]

                processed_task, _ = nk.ppg_process(window_segment, sampling_rate=sampling_rate)
                analyzed_task = nk.ppg_analyze(processed_task, sampling_rate=sampling_rate, method="interval-related")

                task_features = extract_scalar_features(analyzed_task)
                task_features.pop("Slice", None)

                combined_features = {
                    "Participant": participant,
                    "Task": idx,
                    "Slice": slice_count
                }
                combined_features.update(task_features)
                combined_features.update(baseline_features)

                results.append(combined_features)
                slice_count += 1

    except Exception as e:
        print(f"Error processing {signal_type} data for Participant {participant}: {e}")
    return results


# PIXART
pixart_results = []
for participant in PARTICIPANTS:
    participant_dir = f"{base_dir}/{participant:03}"
    pixart_results.extend(process_signal(
        participant=participant,
        participant_dir=participant_dir,
        signal_type="pixart",
        resting_file="pixart_resting.csv",
        resting_column="Resting",
        task_file_name="pixart.csv"
    ))

pixart_file_path = os.path.join(output_folder, "ppg_features_pixart.csv")
if pixart_results:
    df = pd.DataFrame(pixart_results)
    columns_order = ["Participant", "Task", "Slice"] + [c for c in df.columns if c not in ["Participant", "Task", "Slice"]]
    df = df[columns_order]
    df.to_csv(pixart_file_path, index=False)
    print(f"Pixart PPG features saved to {pixart_file_path}")


# PPG
ppg_results = []
for participant in PARTICIPANTS:
    participant_dir = f"{base_dir}/{participant:03}"
    ppg_results.extend(process_signal(
        participant=participant,
        participant_dir=participant_dir,
        signal_type="ppg",
        resting_file="inf_resting.csv",
        resting_column="Resting_PPG",
        task_file_name="inf_ppg.csv"
    ))

ppg_file_path = os.path.join(output_folder, "ppg_features.csv")
if ppg_results:
    df = pd.DataFrame(ppg_results)
    columns_order = ["Participant", "Task", "Slice"] + [c for c in df.columns if c not in ["Participant", "Task", "Slice"]]
    df = df[columns_order]
    df.to_csv(ppg_file_path, index=False)
    print(f"PPG features saved to {ppg_file_path}")
