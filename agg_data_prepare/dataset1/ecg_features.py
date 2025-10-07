# process_ecg_with_paper.py
import os
import sys
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Projekt-Imports & Pfade wie bei dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# ----------------------------
# Verzeichnisse & Parameter
# ----------------------------
BASE_DIR = os.path.join(const.BASE_DIR, "dataset1_SenseCobot/ECG_Shimmer3_Signals")
OUTPUT_FOLDER = os.path.join(const.OUTPUT_DIR, "dataset1")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SAMPLING_RATE = 512
WINDOW_LEN = int(const.INTERVAL * SAMPLING_RATE)
STEP = int(const.STEP * SAMPLING_RATE)

PARTICIPANTS = range(1, 26)
TASKS = range(1, 6)

ECG_COLS = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]

# ----------------------------
# Hilfsfunktionen (Paper-Features)
# ----------------------------
def shannon_entropy(x, bins=10):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log(hist)).sum())

def higuchi_fd(x, kmax=20):
    x = np.asarray(x)
    N = x.size
    if N < kmax + 1:
        return np.nan
    Lk = []
    for k in range(1, kmax + 1):
        Lmk = []
        for m in range(k):
            idx = np.arange(m, N - k, k)
            if idx.size == 0:
                continue
            Lm = np.abs(np.diff(x[idx.tolist() + [idx[-1] + k]]).astype(float)).sum()
            Lm *= (N - 1) / ((idx.size) * k)
            Lmk.append(Lm)
        if len(Lmk) == 0:
            return np.nan
        Lk.append(np.mean(Lmk) / k)
    Lk = np.asarray(Lk)
    if np.any(Lk <= 0):
        return np.nan
    log_k = np.log(np.arange(1, kmax + 1))
    log_Lk = np.log(Lk)
    # Robust: ignoriere k=1 zur Stabilisierung (wie bei dir)
    coeffs = np.polyfit(log_k[1:], log_Lk[1:], 1)
    return float(abs(coeffs[0]))

def freq_band_powers_from_hr(hr_series, rr_ms):
    """
    Bandpower aus Herzfrequenz-Zeitreihe (wie in deinem Code):
    - fs_HR aus mittlerem RR-Intervall (ms) -> Hz
    """
    hr = np.asarray(hr_series, dtype=float)
    hr = hr[~np.isnan(hr)]
    if hr.size < 4:
        return np.nan, np.nan, np.nan
    mean_rr_ms = np.nanmean(rr_ms)
    if not np.isfinite(mean_rr_ms) or mean_rr_ms <= 0:
        return np.nan, np.nan, np.nan
    fs_HR = 1000.0 / mean_rr_ms  # Hz

    f, Pxx = signal.periodogram(hr, fs=fs_HR, scaling='spectrum')
    # Bänder
    VLF = (0.01, 0.04)
    LF = (0.04, 0.15)
    HF = (0.15, 0.40)

    def band_mean(a, f, band):
        mask = (f >= band[0]) & (f <= band[1])
        if not np.any(mask):
            return np.nan
        vals = a[mask]
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if vals.size else np.nan

    vlf = band_mean(Pxx, f, VLF)
    lf = band_mean(Pxx, f, LF)
    hf = band_mean(Pxx, f, HF)
    return vlf, lf, hf

def summarize_rr_hr(rr_ms):
    """
    Herzrate aus RR (ms): HR[bpm] = 60000 / RR
    Liefert dict mit Statistiken RR/HR & normalisierte Varianten.
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[~np.isnan(rr)]
    if rr.size < 2:
        return {
            "RR_Mean": np.nan, "RR_Min": np.nan, "RR_Max": np.nan, "RR_Median": np.nan,
            "HR_Mean": np.nan, "HR_Min": np.nan, "HR_Max": np.nan, "HR_Std": np.nan, "HR_Var": np.nan, "HR_Median": np.nan,
            "RR_norm_Mean": np.nan, "RR_norm_Min": np.nan, "RR_norm_Max": np.nan, "RR_norm_Median": np.nan,
            "HR_norm_Mean": np.nan, "HR_norm_Min": np.nan, "HR_norm_Max": np.nan, "HR_norm_Median": np.nan,
            "RMSSD": np.nan, "SDNN": np.nan, "PNN25": np.nan, "PNN50": np.nan
        }

    hr = 60000.0 / rr

    # RR Stats
    rr_mean = float(np.round(np.mean(rr), 2))
    rr_min = float(np.round(np.min(rr), 2))
    rr_max = float(np.round(np.max(rr), 2))
    rr_median = float(np.round(np.median(rr), 2))

    # HR Stats
    hr_mean = float(np.round(np.mean(hr), 2))
    hr_std = float(np.round(np.std(hr, ddof=0), 2))
    hr_min = float(np.round(np.min(hr), 2))
    hr_max = float(np.round(np.max(hr), 2))
    hr_var = float(np.round(np.var(hr, ddof=0), 2))
    hr_median = float(np.round(np.median(hr), 2))

    # Normalisierungen
    rr_norm = rr / (np.max(rr) if np.max(rr) else np.nan)
    hr_norm = hr / (np.max(hr) if np.max(hr) else np.nan)

    # HRV (wie bei dir über RR-Differenzen in ms)
    diff_rr = np.diff(rr)
    diff_sq = np.abs(diff_rr) ** 2
    if diff_rr.size > 0:
        rmssd = float(np.round(np.sqrt(np.sum(diff_sq) / diff_rr.size), 3))
        nn50 = int(np.sum(np.abs(diff_rr) > 50))
        nn25 = int(np.sum(np.abs(diff_rr) > 25))
        pnn50 = float(np.round(100.0 * nn50 / diff_rr.size, 3))
        pnn25 = float(np.round(100.0 * nn25 / diff_rr.size, 3))
    else:
        rmssd = np.nan
        pnn25 = np.nan
        pnn50 = np.nan

    sdnn = float(np.round(np.std(rr, ddof=1), 3)) if rr.size > 1 else np.nan

    return {
        # RR
        "RR_Mean": rr_mean, "RR_Min": rr_min, "RR_Max": rr_max, "RR_Median": rr_median,
        # HR
        "HR_Mean": hr_mean, "HR_Min": hr_min, "HR_Max": hr_max, "HR_Std": hr_std, "HR_Var": hr_var, "HR_Median": hr_median,
        # Norm
        "RR_norm_Mean": float(np.round(np.nanmean(rr_norm), 2)) if np.isfinite(np.nanmean(rr_norm)) else np.nan,
        "RR_norm_Min": float(np.round(np.nanmin(rr_norm), 2)) if rr_norm.size else np.nan,
        "RR_norm_Max": float(np.round(np.nanmax(rr_norm), 2)) if rr_norm.size else np.nan,
        "RR_norm_Median": float(np.round(np.nanmedian(rr_norm), 2)) if rr_norm.size else np.nan,
        "HR_norm_Mean": float(np.round(np.nanmean(hr_norm), 2)) if np.isfinite(np.nanmean(hr_norm)) else np.nan,
        "HR_norm_Min": float(np.round(np.nanmin(hr_norm), 2)) if hr_norm.size else np.nan,
        "HR_norm_Max": float(np.round(np.nanmax(hr_norm), 2)) if hr_norm.size else np.nan,
        "HR_norm_Median": float(np.round(np.nanmedian(hr_norm), 2)) if hr_norm.size else np.nan,
        # HRV
        "RMSSD": rmssd, "SDNN": sdnn, "PNN25": pnn25, "PNN50": pnn50
    }

def compute_paper_features_from_segment(ecg_segment, sampling_rate=SAMPLING_RATE):
    """
    Nutzt nk.ecg_process zum R-Peak-Finding, leitet RR (ms) & HR ab
    und berechnet Paper-Features für diesen Segment-Slice.
    """
    if ecg_segment is None or len(ecg_segment) < int(0.5 * sampling_rate):
        # zu kurz
        return None

    try:
        # NeuroKit Processing (robustes R-Peak-Detekt.)
        ecg_signals, info = nk.ecg_process(ecg_segment, sampling_rate=sampling_rate)
        # rpeaks in Samples -> ms
        rpeaks_samples = info.get("ECG_R_Peaks", None)
        if rpeaks_samples is None or len(rpeaks_samples) < 3:
            return None
        rr_ms = np.diff(rpeaks_samples) * 1000.0 / sampling_rate

        stats = summarize_rr_hr(rr_ms)

        # HR-Zeitreihe auf RR-Basis (mittlere Zeitstempel, nicht zwingend gleichmäßig)
        hr_series = 60000.0 / (rr_ms + 1e-9)
        vlf, lf, hf = freq_band_powers_from_hr(hr_series, rr_ms)

        # Entropie & HFD auf gefiltertem ECG-Signal
        # (hier: die von NK vorgefilterte Signalspalte verwenden, falls vorhanden)
        if "ECG_Clean" in ecg_signals.columns:
            x = ecg_signals["ECG_Clean"].values
        else:
            x = ecg_segment

        features = {
            **stats,
            "Higuchi_FD": float(np.round(higuchi_fd(x, kmax=20), 6)),
            "Shannon_Entropy": float(np.round(shannon_entropy(x, bins=10), 6)),
            "VLF": vlf, "LF": lf, "HF": hf,
        }
        return features
    except Exception:
        return None

# ----------------------------
# Hauptverarbeitung
# ----------------------------
def process_ecg_with_paper():
    # Ergebnisse getrennt nach Ableitung (wie bei dir) UND eine „Paper“-Gesamtdatei
    per_lead_results = {col: [] for col in ECG_COLS}
    paper_rows = []

    for participant in PARTICIPANTS:
        pid = f"{participant:02}"

        # ---- Baseline laden & Paper-/Baseline-Features vorbereiten ----
        baseline_file = f"{BASE_DIR}/ECG_Baseline_P_{pid}.csv"
        baseline_features_by_col = {col: {} for col in ECG_COLS}
        baseline_paper_by_col = {col: {} for col in ECG_COLS}

        if os.path.exists(baseline_file):
            try:
                print(f"[P{pid}] Baseline…")
                baseline_df = pd.read_csv(baseline_file)
                for col in ECG_COLS:
                    if col in baseline_df.columns:
                        # NK Analyse (wie in deinem ersten Skript)
                        ecg_signals, _ = nk.ecg_process(baseline_df[col].dropna().values, sampling_rate=SAMPLING_RATE)
                        analyzed = nk.ecg_analyze(ecg_signals, sampling_rate=SAMPLING_RATE, method="interval-related")
                        baseline_features_by_col[col] = {
                            f"Baseline_{k}": v for k, v in extract_scalar_features(analyzed).items()
                        }
                        # Paper-Features für Baseline (als zusätzliche Referenz)
                        paper = compute_paper_features_from_segment(baseline_df[col].dropna().values, SAMPLING_RATE)
                        if paper:
                            baseline_paper_by_col[col] = {f"BaselinePaper_{k}": v for k, v in paper.items()}
                    else:
                        print(f"  ⚠ Spalte '{col}' fehlt in {baseline_file}")
            except Exception as e:
                print(f"  ⚠ Fehler Baseline P{pid}: {e}")
        else:
            print(f"[P{pid}] ⚠ Baseline-Datei fehlt")

        # ---- Tasks ----
        for task in TASKS:
            task_file = f"{BASE_DIR}/ECG_Task {task}_P_{pid}.csv"
            if not os.path.exists(task_file):
                print(f"[P{pid}] ⚠ Task-Datei fehlt: Task {task}")
                continue

            try:
                print(f"[P{pid}] Task {task}…")
                df_task = pd.read_csv(task_file)

                for col in ECG_COLS:
                    if col not in df_task.columns:
                        print(f"  ⚠ Spalte '{col}' fehlt in {task_file}")
                        continue

                    sig = df_task[col].dropna().values
                    total = len(sig)
                    start = 0
                    slice_id = 1

                    while start + WINDOW_LEN <= total:
                        seg = sig[start:start + WINDOW_LEN]

                        # NK-Analyse für deinen bisherigen Output (lead-spezifische CSVs)
                        ecg_signals, _ = nk.ecg_process(seg, sampling_rate=SAMPLING_RATE)
                        analyzed = nk.ecg_analyze(ecg_signals, sampling_rate=SAMPLING_RATE, method="interval-related")

                        row_common = {
                            "Participant": participant,
                            "Task": task,
                            "Slice": slice_id,
                            "ECG_Lead": col,
                        }

                        # (1) wie bisher pro Ableitung:
                        comb = dict(row_common)
                        comb.update(extract_scalar_features(analyzed))
                        comb.update(baseline_features_by_col.get(col, {}))
                        per_lead_results[col].append(comb)

                        # (2) Paper-Features in EINE gemeinsame Datei:
                        paper = compute_paper_features_from_segment(seg, SAMPLING_RATE)
                        if paper:
                            paper_row = dict(row_common)
                            paper_row.update(paper)
                            # optional Baseline-Bezug
                            paper_row.update(baseline_paper_by_col.get(col, {}))
                            # zusätzlich: auch deine NK-Baseline-Features anhängen, falls gewünscht
                            paper_row.update(baseline_features_by_col.get(col, {}))
                            paper_rows.append(paper_row)

                        start += STEP
                        slice_id += 1

            except Exception as e:
                print(f"[P{pid}] ⚠ Fehler Task {task}: {e}")

    # ---- Dateien speichern ----
    # a) pro Ableitung (wie in deinem ersten Code)
    for col, data in per_lead_results.items():
        if data:
            df = pd.DataFrame(data)
            out = os.path.join(OUTPUT_FOLDER, f"ecg_features_{col.replace(' ', '_')}.csv")
            df.to_csv(out, index=False)
            print(f" gespeichert: {out}")

    # b) EINE „Paper“-Datei über alles
    if paper_rows:
        paper_df = pd.DataFrame(paper_rows)
        paper_out = os.path.join(OUTPUT_FOLDER, "ecg_features_paper.csv")
        paper_df.to_csv(paper_out, index=False)
        print(f" gespeichert: {paper_out}")
    else:
        print("⚠ Keine Paper-Features erzeugt (zu wenig/fehlerhafte Daten?).")

# ----------------------------
# CLI
# ----------------------------
def main():
    process_ecg_with_paper()

if __name__ == "__main__":
    main()
