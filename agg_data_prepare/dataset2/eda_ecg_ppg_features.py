import os
import sys
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk
from bisect import bisect_left

# Projektpfade/Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

warnings.filterwarnings("ignore")

# ---------------------------------- Pfade ----------------------------------

base_dir = os.path.join(const.BASE_DIR, "dataset2_RobotBehaviour", "Measurements_fixed")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset2")
os.makedirs(output_folder, exist_ok=True)

# Nominale Raten dienen nur als Fallback, falls keine TimeStamp-Spalte vorhanden ist.
SIGNALS = {
    "EDA": {"sampling_rate": 4, "ext": "EDA", "method": "eda"},
    "ECG": {"sampling_rate": 130, "ext": "ECG", "method": "ecg"},
    "BVP": {"sampling_rate": 64, "ext": "BVP", "method": "ppg"},  # BVP == PPG in NeuroKit
}

participants = range(1, 26)
speeds = [1, 2]
robots = [1, 2, 3]

# ---------------------------------- 72s: Spalten entfernen ----------------------------------
# HRV-Features, die bei 72 s unzuverlässig sind (nur für ECG entfernen)
ECG_DROP_72S = [
    # Langzeit-Zeitbereich

]

# EDA-Features, die bei 72 s besser weggelassen werden
EDA_DROP_72S = [
 
]

# ------------------------------- Hilfsfunktionen ----------------------------

def _to_float(x):
    """Konvertiert Strings mit deutschem Komma zu float, leere zu NaN."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s == "":
        return float("nan")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _detect_signal_column(df, ext):
    """Findet die Signalspalte (genauer Name 'ext' oder teilweiser Match, sonst 2. Spalte bei 2 Spalten)."""
    if ext in df.columns:
        return ext
    for col in df.columns:
        if ext.lower() in str(col).lower():
            return col
    if len(df.columns) == 2:
        # CSVs vom Typ "TimeStamp;EDA"
        return df.columns[1]
    return None

def _detect_time_column(df):
    """
    Sucht nach Zeitspalte: bevorzugt 'TimeStamp', 'time', 'zeit', 'timestamp', 'sec', 'seconds' (case-insensitive).
    Fallback: erste Spalte, wenn streng monoton steigend und numerisch.
    """
    # Namensbasierte Suche
    for col in df.columns:
        name = str(col).strip().lower()
        if any(k in name for k in ["timestamp", "time", "zeit", "sec", "seconds"]):
            t = df[col].apply(_to_float)
            if t.notna().sum() >= 2:
                t = t.values.astype(float)
                diffs = np.diff(t[np.isfinite(t)])
                if diffs.size > 0 and np.all(diffs > 0):
                    return col

    # Fallback: erste Spalte prüfen
    first = df.columns[0]
    t = df[first].apply(_to_float).values.astype(float)
    finite = np.isfinite(t)
    if finite.sum() >= 2:
        diffs = np.diff(t[finite])
        if diffs.size > 0 and np.all(diffs > 0):
            return first

    return None

def _detect_sampling_rate_from_time(times):
    """Schätzt Hz aus median(Δt); robust gegen Ausreißer."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 3:
        return None
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    dt = np.median(diffs)
    if dt <= 0:
        return None
    return 1.0 / dt

def _make_time_based_windows(times, interval_sec, step_sec):
    """
    Bildet Fenster anhand echter Zeitstempel.
    Rückgabe: Liste (start_idx, end_idx_inclusive).
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    if n == 0:
        return []

    windows = []
    t_cur = times[0]
    t_last = times[-1]

    while t_cur + interval_sec <= t_last:
        i0 = bisect_left(times, t_cur)                 # erster Index mit time >= t_cur
        t_end = t_cur + interval_sec
        i1 = bisect_left(times, t_end) - 1             # letzter Index mit time < t_end
        if 0 <= i0 < n and 0 <= i1 < n and i1 >= i0:
            windows.append((int(i0), int(i1)))
        t_cur += step_sec

    return windows

# ----------------------------- Hauptverarbeitung ----------------------------

def process_signal(signal_name, config):
    results = []

    # Vorgabewerte (nur Fallback, falls keine TimeStamp-Spalte gefunden)
    nominal_sr = float(config["sampling_rate"])
    method = config["method"]   # "eda", "ecg" oder "ppg"
    ext = config["ext"]

    interval_sec = 72
    step_sec = 20

    for participant in participants:
        for speed in speeds:
            for robot in robots:
                file_path = os.path.join(base_dir, f"p_{participant}", f"{speed}_{robot}", f"{ext}.csv")
                if not os.path.exists(file_path):
                    print(f"[MISS] {file_path}")
                    continue

                try:
                    df = pd.read_csv(file_path, sep=";")
                    # Signalspalte suchen
                    sig_col = _detect_signal_column(df, ext)
                    if sig_col is None:
                        print(f"[ERR ] Missing signal column for '{ext}' in {file_path} (columns={list(df.columns)})")
                        continue

                    # Zeitspalte suchen (z. B. 'TimeStamp')
                    time_col = _detect_time_column(df)

                    # Numerik konvertieren
                    df[sig_col] = df[sig_col].apply(_to_float)
                    if time_col is not None:
                        df[time_col] = df[time_col].apply(_to_float)

                    # NaNs im Signal droppen (Zeit wird danach passend mitgefiltert)
                    before = len(df)
                    df = df.dropna(subset=[sig_col])
                    after = len(df)
                    if after < before:
                        print(f"[INFO] Dropped {before - after} NaNs (signal={sig_col}) in {file_path}")

                    if len(df) == 0:
                        print(f"[INFO] Empty after cleaning: {file_path}")
                        continue

                    signal = df[sig_col].astype(float).values

                    # Zeiten vorbereiten
                    if time_col is not None:
                        times = df[time_col].astype(float).values
                        # Nur endliche Werte behalten (Signal & Zeit kongruent)
                        finite = np.isfinite(times)
                        if finite.sum() != len(times):
                            times = times[finite]
                            signal = signal[finite]
                        if len(signal) < 3:
                            print(f"[INFO] Not enough points after time alignment in {file_path}")
                            continue

                        # Pro-Datei Hz schätzen
                        sr_used = _detect_sampling_rate_from_time(times)
                        sr_note = "detected"
                        if sr_used is None:
                            sr_used = nominal_sr
                            sr_note = "fallback-nominal"

                        # Für NeuroKit IMMER int Hz
                        sr_used_int = int(round(sr_used))

                        print(f"[FILE] {file_path} | n={len(signal)} | sr({sr_note})={sr_used:.6f} Hz "
                              f"(nk={sr_used_int}) | interval={interval_sec}s step={step_sec}s | time-based")

                        # Zeitbasierte Fenster
                        windows = _make_time_based_windows(times, interval_sec, step_sec)
                    else:
                        # Kein TimeStamp → Fallback: sample-basiert mit nominaler Hz
                        sr_used = nominal_sr
                        sr_used_int = int(round(sr_used))
                        win_len = int(round(interval_sec * sr_used_int))
                        step = int(round(step_sec * sr_used_int))
                        if win_len <= 0 or step <= 0:
                            print(f"[ERR ] Non-positive window/step (len={win_len}, step={step}) in {file_path}")
                            continue

                        expected = int((len(signal) - win_len) // step + 1) if len(signal) >= win_len else 0
                        print(f"[FILE] {file_path} | n={len(signal)} | sr(nominal)={sr_used:.6f} Hz "
                              f"(nk={sr_used_int}) | win={win_len} step={step} expected={expected} | sample-based")

                        windows, start_idx = [], 0
                        while start_idx + win_len <= len(signal):
                            end_idx = start_idx + win_len - 1
                            windows.append((int(start_idx), int(end_idx)))
                            start_idx += step
                        times = None  # keine echte Zeit

                    if len(windows) == 0:
                        print(f"[INFO] No windows in {file_path}")
                        continue

                    # >>> SAMPLING_RATE für die Fensteranalyse (int!) – wird unten EXAKT verwendet
                    SAMPLING_RATE = sr_used_int

                    slice_num = 1
                    for (i0, i1) in windows:
                        # i0/i1 sind ints; i1 ist inklusiv → +1 beim Slicen
                        window_signal = signal[i0:i1 + 1]
                        if window_signal.size < 10:
                            slice_num += 1
                            continue
                        try:
                            if method == "eda":
                                # >>> EXAKT DEINE DREI ZEILEN (NICHT ÄNDERN)
                                processed_window, _ = nk.eda_process(window_signal, sampling_rate=SAMPLING_RATE)
                                analyzed_window = nk.eda_analyze(processed_window, sampling_rate=SAMPLING_RATE, method="interval-related")
                                window_features = extract_scalar_features(analyzed_window)

                                # --- 72s: ungeeignete EDA-Features entfernen ---
                                for key in list(window_features.keys()):
                                    if key in EDA_DROP_72S:
                                        window_features.pop(key, None)

                            elif method == "ecg":
                                processed_window, _ = nk.ecg_process(window_signal, sampling_rate=SAMPLING_RATE)
                                analyzed_window = nk.ecg_analyze(processed_window, sampling_rate=SAMPLING_RATE, method="interval-related")
                                window_features = extract_scalar_features(analyzed_window)

                                # --- 72s: ungeeignete ECG/HRV-Features entfernen ---
                                for key in list(window_features.keys()):
                                    if key in ECG_DROP_72S:
                                        window_features.pop(key, None)

                            elif method == "ppg":
                                processed_window, _ = nk.ppg_process(window_signal, sampling_rate=SAMPLING_RATE)
                                analyzed_window = nk.ppg_analyze(processed_window, sampling_rate=SAMPLING_RATE, method="interval-related")
                                window_features = extract_scalar_features(analyzed_window)

                            else:
                                raise ValueError(f"Unknown method '{method}'")

                            row = {
                                "Participant": participant,
                                "Task": (speed - 1) * 3 + robot,
                                "Slice": slice_num,
                                "Speed": speed,
                                "Robots": robot,
                            }
                            row.update(window_features)
                            results.append(row)

                        except Exception as e_slice:
                            print(f"[WARN] Slice error | file={file_path} slice={slice_num} "
                                  f"start={i0} end={i1} | {e_slice}")
                        finally:
                            slice_num += 1

                except Exception as e_file:
                    print(f"[ERR ] File-level error {file_path}: {e_file}")

    return pd.DataFrame(results)

# ----------------------------------- Run -----------------------------------

if __name__ == "__main__":
    for signal_name, config in SIGNALS.items():
        print(f"\n[RUN ] Processing {signal_name} …")
        df = process_signal(signal_name, config)

        if not df.empty:
            if signal_name == "BVP":
                output_name = "ppg_features.csv"   # BVP -> PPG-Features
            else:
                output_name = f"{signal_name.lower()}_features.csv"
            output_path = os.path.join(output_folder, output_name)
            df.to_csv(output_path, index=False)
            print(f"[SAVE] {output_path} | rows={len(df)}")
        else:
            print(f"[INFO] No valid data for {signal_name}")
