#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import neurokit2 as nk
import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Directories
base_dir = os.path.join(const.BASE_DIR, "dataset1_SenseCobot/GSR_Shimmer3_Signals")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset1")
os.makedirs(output_folder, exist_ok=True)

# Sampling rate (Fallback) and participants/tasks
SAMPLING_RATE = 512  
window_length = int(const.INTERVAL * SAMPLING_RATE)  
step = int(const.STEP * SAMPLING_RATE)              
participants = range(1, 26)
tasks = range(1, 6)

def _smart_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";", decimal=",")
        except Exception:
            return pd.read_csv(path)

def _infer_fs_from_timestamp(ts: pd.Series):
    try:
        t = pd.to_datetime(ts, errors="coerce").dropna().reset_index(drop=True)
        if t.size < 3:
            return None
        dt = t.diff().dropna().dt.total_seconds()
        dt = dt[(dt > 0) & dt.notna()]
        if dt.empty:
            return None
        period = float(dt.median())
        if period <= 0:
            return None
        return float(1.0 / period)
    except Exception:
        return None

def process_gsr():
    results = []
    gsr_column = "GSR Conductance CAL"   # ggf. anpassen
    ts_column  = "Timestamp"

    for participant in participants:
        participant_id = f"{participant:02}"
        baseline_file = f"{base_dir}/GSR_Baseline_P_{participant_id}.csv"
        baseline_features = {}

        # ---------------- Baseline ----------------
        if os.path.exists(baseline_file):
            try:
                print(f"Processing baseline for Participant {participant_id}")
                baseline_data = _smart_read_csv(baseline_file)
                if gsr_column in baseline_data.columns:
                    fs_b = _infer_fs_from_timestamp(baseline_data[ts_column]) if ts_column in baseline_data.columns else None
                    if not fs_b or not pd.notna(fs_b) or fs_b <= 0:
                        fs_b = float(SAMPLING_RATE)
                    sr_b = int(round(fs_b))  # *** WICHTIG: NeuroKit als int ***

                    x0 = pd.to_numeric(baseline_data[gsr_column], errors="coerce")
                    x0 = x0.interpolate(limit_direction="both").bfill().ffill().to_numpy()

                    if len(x0) >= int(15 * sr_b):  # 15 Sekunden
                        processed_baseline, _ = nk.eda_process(x0, sampling_rate=sr_b)
                        analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=sr_b, method="interval-related")
                        baseline_features = {f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()}
                else:
                    print(f"Column '{gsr_column}' missing in {baseline_file}")
            except Exception as e:
                print(f"Error processing baseline for Participant {participant_id}: {e}")
        else:
            print(f"Baseline file missing for Participant {participant_id}")

        # ---------------- Tasks ----------------
        for task in tasks:
            task_file = f"{base_dir}/GSR_Task {task}_P_{participant_id}.csv"

            if os.path.exists(task_file):
                try:
                    print(f"Processing Task {task} for Participant {participant_id}")
                    task_data = _smart_read_csv(task_file)
                    if gsr_column not in task_data.columns:
                        print(f"Column '{gsr_column}' missing in {task_file}")
                        continue
                    if ts_column not in task_data.columns:
                        print(f"Column '{ts_column}' missing in {task_file}")
                        continue

                    t_all = pd.to_datetime(task_data[ts_column], errors="coerce")
                    sig_series = pd.to_numeric(task_data[gsr_column], errors="coerce")
                    valid = t_all.notna() & sig_series.notna()

                    t = t_all[valid].reset_index(drop=True)
                    x = sig_series[valid].reset_index(drop=True)
                    x = x.interpolate(limit_direction="both").bfill().ffill().to_numpy()

                    if t.empty or x.size < 2:
                        print(f"[SKIP] P={participant} T={task}: no valid data.")
                        continue

                    t_sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
                    # strikt monoton
                    import numpy as np
                    keep = np.r_[True, np.diff(t_sec) > 0]
                    t_sec = t_sec[keep]
                    x = x[keep]

                    if t_sec.size < 2 or x.size < 2:
                        print(f"[SKIP] P={participant} T={task}: not enough monotonic timestamps.")
                        continue

                    # fs für NeuroKit schätzen (als int!)
                    fs = _infer_fs_from_timestamp(t)
                    sr = int(round(fs)) if fs and pd.notna(fs) and fs > 0 else int(SAMPLING_RATE)

                    duration = float(t_sec[-1])  # t_sec[0]==0
                    if duration < float(const.INTERVAL):
                        print(f"[SKIP] P={participant} T={task}: duration {duration:.2f}s < {const.INTERVAL}s.")
                        continue

                    # Anzahl Zeitfenster
                    n_win = int((duration - float(const.INTERVAL)) // float(const.STEP) + 1)
                    print(f"[INFO] P={participant} T={task}: fs≈{sr:.2f} Hz, duration≈{duration:.2f}s, windows={n_win}")

                    slice_count = 1
                    for k in range(n_win):
                        t0 = k * float(const.STEP)
                        t1 = t0 + float(const.INTERVAL)

                        # Indizes via searchsorted (immer int!)
                        s_idx = int(np.searchsorted(t_sec, t0, side="left"))
                        e_idx = int(np.searchsorted(t_sec, t1, side="right"))

                        # Grenzen absichern
                        s_idx = max(0, min(s_idx, x.shape[0] - 1))
                        e_idx = max(s_idx + 1, min(e_idx, x.shape[0]))

                        seg = x[s_idx:e_idx]
                        if seg.size < 2:
                            slice_count += 1
                            continue

                        try:
                            processed_segment, _ = nk.eda_process(seg, sampling_rate=sr)  # *** int sr ***
                            analyzed_task = nk.eda_analyze(processed_segment, sampling_rate=sr, method="interval-related")

                            combined_features = {
                                "Participant": participant,
                                "Task": task,
                                "Slice": slice_count
                            }
                            combined_features.update(extract_scalar_features(analyzed_task))
                            combined_features.update(baseline_features)
                            results.append(combined_features)
                        except Exception as ex:
                            print(f"[WARN] P={participant} T={task} slice#{slice_count} failed: {ex}")
                        finally:
                            slice_count += 1

                except Exception as e:
                    print(f"Error processing Task {task} for Participant {participant_id}: {e}")
            else:
                print(f"Task file missing for Participant {participant_id}, Task {task}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_folder, "gsr_features.csv")
        results_df.to_csv(output_file, index=False)
        print(f"GSR features saved to: {output_file}")

def main():
    process_gsr()

if __name__ == "__main__":
    main()
