import pandas as pd
import numpy as np
import neurokit2 as nk
import os
import warnings
import sys

# Ermöglicht Import aus übergeordnetem Verzeichnis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Directories
base_dir = os.path.join(const.BASE_DIR, "dataset1_SenseCobot/EEG_Enobio20_Signals")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset1")
os.makedirs(output_folder, exist_ok=True)

# Parameters
SAMPLING_RATE = 500
window_length_samples = int(const.INTERVAL * SAMPLING_RATE)
step_samples = int(const.STEP * SAMPLING_RATE)

participants = range(1, 26)
tasks = range(1, 6)
eeg_channels = ["P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8", "C4", "F4", "Fz", "C3", "F3", "T7", "F7"]
bands = ["Gamma", "Beta", "Alpha", "Theta", "Delta"]

# Define columns
task_feature_columns = [f"{band}_Ch{ch}" for ch in range(1, len(eeg_channels) + 1) for band in bands]
baseline_feature_columns = [f"Baseline_{col}" for col in task_feature_columns]
columns = ["Participant", "Task", "Slice"] + task_feature_columns + baseline_feature_columns

# Process EEG data
def process_eeg():
    results = []

    for participant in participants:
        participant_id = f"{participant:02}"

        # Process baseline
        baseline_file = os.path.join(base_dir, f"EEG_Baseline_P_{participant_id}.csv")
        baseline_features = {}
        if os.path.exists(baseline_file):
            try:
                baseline_data = pd.read_csv(baseline_file)
                for idx, channel in enumerate(eeg_channels):
                    if channel in baseline_data.columns:
                        eeg_matrix = baseline_data[channel].dropna().values.reshape(1, -1)
                        res = nk.eeg_power(eeg_matrix, sampling_rate=SAMPLING_RATE)
                        if "Channel" in res.columns:
                            res.drop(columns=["Channel"], inplace=True)
                        for i, band in enumerate(bands):
                            baseline_features[f"Baseline_{band}_Ch{idx + 1}"] = res.iloc[0, i]
            except Exception as e:
                print(f"Error processing Baseline for Participant {participant_id}: {e}")

        # Process tasks
        for task in tasks:
            task_file = os.path.join(base_dir, f"EEG_Task {task}_P_{participant_id}.csv")
            if os.path.exists(task_file):
                try:
                    task_data = pd.read_csv(task_file)
                    total_samples = len(task_data)
                    slice_count = 1
                    start_idx = 0

                    while start_idx + window_length_samples <= total_samples:
                        combined_features = {
                            "Participant": participant,
                            "Task": task,
                            "Slice": slice_count
                        }

                        for idx, channel in enumerate(eeg_channels):
                            if channel in task_data.columns:
                                segment = task_data[channel].dropna().values[start_idx:start_idx + window_length_samples]
                                if len(segment) == window_length_samples:
                                    eeg_matrix = segment.reshape(1, -1)
                                    res = nk.eeg_power(eeg_matrix, sampling_rate=SAMPLING_RATE)
                                    if "Channel" in res.columns:
                                        res.drop(columns=["Channel"], inplace=True)
                                    for i, band in enumerate(bands):
                                        combined_features[f"{band}_Ch{idx + 1}"] = res.iloc[0, i]

                        combined_features.update(baseline_features)
                        results.append(combined_features)

                        slice_count += 1
                        start_idx += step_samples

                except Exception as e:
                    print(f"Error processing Task {task} for Participant {participant_id}: {e}")

    # Save results
    results_df = pd.DataFrame(results, columns=columns)
    output_file = os.path.join(output_folder, "eeg_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"EEG features saved to: {output_file}")

# Main function
def main():
    process_eeg()

if __name__ == "__main__":
    main()