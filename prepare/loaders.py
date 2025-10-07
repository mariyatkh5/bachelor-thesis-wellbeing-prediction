# loaders.py
import os, re
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from utils import rng, windowize_by_index, infer_fs_from_timestamp, _fill_numeric_series
from filters import resample_any, zscore_norm, eda_lowpass_5hz, eda_wavelet_daub5, eda_tonic_phasic, ecg_bandpass_10_75
from paths import DATASET1_DIR, DATASET2_DIR, DATASET3_DIR, DATASET8_DIR, D1_SIGNALS, D2_SIGNALS, D3_SIGNALS, D8_SIGNALS

SINGLE_EDA_SR, SINGLE_ECG_SR, COMBO_SR = 4.0, 128.0, 128.0


def _resample_raw_linear(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Simple linear resampling (no filter)."""
    x = np.asarray(x, float)
    if fs_in == fs_out or len(x) <= 1:
        return x.astype(np.float32)
    N, new_N = len(x), int(round(len(x) * fs_out / fs_in))
    if new_N <= 1:
        return x.astype(np.float32)
    xi, xo = np.linspace(0, N - 1, N), np.linspace(0, N - 1, new_N)
    return np.interp(xo, xi, x).astype(np.float32)


def _make_arrays(XN_list, XF_list, XN128_list, XF128_list, rows):
    XN = np.stack(XN_list, 0) if XN_list else np.empty((0,1,1), np.float32)
    XF = np.stack(XF_list, 0) if XF_list else np.empty((0,1,1), np.float32)
    XN128 = np.stack(XN128_list, 0) if XN128_list else np.empty((0,1,1), np.float32)
    XF128 = np.stack(XF128_list, 0) if XF128_list else np.empty((0,1,1), np.float32)
    return XN, XF, XN128, XF128, pd.DataFrame(rows)


def _eda_pipeline(x, fs):
    x = zscore_norm(x)
    x = eda_lowpass_5hz(x, fs)
    x = eda_wavelet_daub5(x)
    tonic, _ = eda_tonic_phasic(x, fs)
    return tonic


def _ecg_pipeline(x, fs):
    return ecg_bandpass_10_75(zscore_norm(x), fs)


# ===================================================================
# Dataset 1
# ===================================================================

def load_d1_signal(kind: str, args):
    parts = rng(args.participants) or range(1, 23)
    tasks = rng(args.tasks) or range(1, 7)
    cfg = D1_SIGNALS[kind]
    XN_list, XF_list, XN128_list, XF128_list, rows = [], [], [], [], []

    for p in parts:
        for tsk in tasks:
            fp = os.path.join(DATASET1_DIR, cfg["folder"], cfg["pattern"].format(task=tsk, p2=f"{p:02d}"))
            if not os.path.isfile(fp):
                continue
            try:
                df = pd.read_csv(fp, sep=None, engine="python", on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(fp, sep=";", decimal=",", on_bad_lines="skip")

            col = next((c for c in cfg["col_candidates"] if c in df.columns), None)
            if not col:
                continue
            ts = next((df[c] for c in df.columns if "time" in c.lower()), None)
            fs = infer_fs_from_timestamp(ts) or 512.0
            x = _fill_numeric_series(df[col])
            if len(x) < args.min_points:
                continue

            if kind == "EDA":
                tonic = _eda_pipeline(x, fs)
                x4_f, x128_f = resample_any(tonic, fs, 4), resample_any(tonic, fs, 128)
                x4_r, x128_r = _resample_raw_linear(x, fs, 4), _resample_raw_linear(x, fs, 128)
                sr = 4
            else:
                filt = _ecg_pipeline(x, fs)
                x4_f, x128_f = None, resample_any(filt, fs, 128)
                x4_r, x128_r, sr = None, _resample_raw_linear(x, fs, 128), 128

            mp = int(args.min_points if not args.auto_min_points else round(args.interval * sr))
            step_sr = sr
            for s, e, t0, t1 in windowize_by_index(len(x4_r or x128_r), step_sr, args.interval, args.step, mp):
                if sr == 4:
                    s128, e128 = int(round(t0 * 128)), int(round(t0 * 128 + args.interval * 128))
                    if e128 > len(x128_r):
                        continue
                    XN_list.append(x4_r[s:e, None])
                    XF_list.append(x4_f[s:e, None])
                    XN128_list.append(x128_r[s128:e128, None])
                    XF128_list.append(x128_f[s128:e128, None])
                else:
                    XN_list.append(x4_r[s:e, None])
                    XF_list.append(x128_f[s:e, None])
                    XN128_list.append(x4_r[s:e, None])
                    XF128_list.append(x128_f[s:e, None])

                rows.append(dict(Participant=p, Task=tsk, Start_sec=t0, End_sec=t1,
                                 file=fp, signal=kind.upper(), sampling_rate=int(sr)))

    return _make_arrays(XN_list, XF_list, XN128_list, XF128_list, rows)


# ===================================================================
# Dataset 2
# ===================================================================

def load_d2_signal(kind: str, args):
    parts = rng(args.participants) or range(1, 26)
    speeds = rng(args.speeds) or [1, 2]
    robots = rng(args.robots) or [1, 2, 3]
    cfg = D2_SIGNALS[kind]
    XN_list, XF_list, XN128_list, XF128_list, rows = [], [], [], [], []

    for p in parts:
        for s in speeds:
            for r in robots:
                fp = os.path.join(DATASET2_DIR, f"p_{p}", f"{s}_{r}", f"{cfg['ext']}.csv")
                if not os.path.isfile(fp):
                    continue
                try:
                    df = pd.read_csv(fp, sep=";", decimal=",", on_bad_lines="skip")
                except Exception:
                    continue

                if cfg["ext"] not in df.columns:
                    continue
                x = pd.to_numeric(df[cfg["ext"]], errors="coerce").dropna().to_numpy(np.float32)
                if len(x) < args.min_points:
                    continue
                t = pd.to_numeric(df.get("TimeStamp"), errors="coerce")
                fs = infer_fs_from_timestamp(t) or (4.0 if kind == "EDA" else 130.0)

                if kind == "EDA":
                    tonic = _eda_pipeline(x, fs)
                    x4_f, x128_f = resample_any(tonic, fs, 4), resample_any(tonic, fs, 128)
                    x4_r, x128_r = _resample_raw_linear(x, fs, 4), _resample_raw_linear(x, fs, 128)
                    sr = 4
                else:
                    filt = _ecg_pipeline(x, fs)
                    x128_f, x128_r, sr = resample_any(filt, fs, 128), _resample_raw_linear(x, fs, 128), 128

                mp = int(args.min_points if not args.auto_min_points else round(args.interval * sr))
                for sidx, eidx, t0, t1 in windowize_by_index(len(x128_r if sr == 128 else x4_r),
                                                             sr, args.interval, args.step, mp):
                    if sr == 4:
                        s128 = int(round(t0 * 128))
                        e128 = s128 + int(round(args.interval * 128))
                        if e128 > len(x128_r): continue
                        XN_list.append(x4_r[sidx:eidx, None])
                        XF_list.append(x4_f[sidx:eidx, None])
                        XN128_list.append(x128_r[s128:e128, None])
                        XF128_list.append(x128_f[s128:e128, None])
                    else:
                        seg_n, seg_f = x128_r[sidx:eidx, None], x128_f[sidx:eidx, None]
                        XN_list.append(seg_n)
                        XF_list.append(seg_f)
                        XN128_list.append(seg_n)
                        XF128_list.append(seg_f)

                    rows.append(dict(Participant=p, Speed=s, Robots=r, Start_sec=t0, End_sec=t1,
                                     file=fp, signal=kind.upper(), sampling_rate=int(sr)))

    return _make_arrays(XN_list, XF_list, XN128_list, XF128_list, rows)


# ===================================================================
# Dataset 3
# ===================================================================

def _pick_signal_column(df: pd.DataFrame, prefer: List[str]) -> Optional[str]:
    for c in prefer:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower() not in {"time", "timestamp"}:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(10, int(0.5 * len(s))):
                return c
    return None


def load_d3_signal(kind: str, args):
    parts = rng(args.participants)
    if parts is None and os.path.isdir(DATASET3_DIR):
        parts = [int(d) for d in os.listdir(DATASET3_DIR) if re.fullmatch(r"\d{3}", d)]
    if not parts:
        return (np.empty((0,1,1), np.float32),)*4 + (pd.DataFrame(),)

    XN_list, XF_list, XN128_list, XF128_list, rows = [], [], [], [], []
    for p in parts:
        p3 = f"{p:03d}"
        part_dir = os.path.join(DATASET3_DIR, p3)
        fname = "inf_gsr.csv" if kind.upper() == "EDA" else "inf_ecg.csv"
        fp = os.path.join(part_dir, fname)
        if not os.path.isfile(fp): continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            df = pd.read_csv(fp, sep=";", decimal=",")
        fs = 256.0
        col = _pick_signal_column(df, ["eda", "ecg", "value", "Value"])
        if not col: continue
        x = _fill_numeric_series(df[col])
        if len(x) < args.min_points: continue

        if kind.upper() == "EDA":
            tonic = _eda_pipeline(x, fs)
            x4_f, x128_f = resample_any(tonic, fs, 4), resample_any(tonic, fs, 128)
            x4_r, x128_r = _resample_raw_linear(x, fs, 4), _resample_raw_linear(x, fs, 128)
            sr = 4
        else:
            filt = _ecg_pipeline(x, fs)
            x128_f, x128_r, sr = resample_any(filt, fs, 128), _resample_raw_linear(x, fs, 128), 128

        mp = int(args.min_points if not args.auto_min_points else round(args.interval * sr))
        for s, e, t0, t1 in windowize_by_index(len(x128_r if sr == 128 else x4_r),
                                               sr, args.interval, args.step, mp):
            if sr == 4:
                s128 = int(round(t0 * 128))
                e128 = s128 + int(round(args.interval * 128))
                if e128 > len(x128_r): continue
                XN_list.append(x4_r[s:e, None]); XF_list.append(x4_f[s:e, None])
                XN128_list.append(x128_r[s128:e128, None]); XF128_list.append(x128_f[s128:e128, None])
            else:
                seg_n, seg_f = x128_r[s:e, None], x128_f[s:e, None]
                XN_list.append(seg_n); XF_list.append(seg_f)
                XN128_list.append(seg_n); XF128_list.append(seg_f)

            rows.append(dict(Participant=p, Task=None, Start_sec=t0, End_sec=t1,
                             file=fp, signal=kind.upper(), sampling_rate=int(sr)))

    return _make_arrays(XN_list, XF_list, XN128_list, XF128_list, rows)


# ===================================================================
# Dataset 8
# ===================================================================

def load_d8_signal(kind: str, args):
    cfg = D8_SIGNALS[kind]
    XN_list, XF_list, XN128_list, XF128_list, rows = [], [], [], [], []
    if not os.path.isdir(DATASET8_DIR):
        print(f"[WARN] dataset8 dir not found: {DATASET8_DIR}")
        return (np.empty((0,1,1), np.float32),)*4 + (pd.DataFrame(),)

    for marker_folder in os.listdir(DATASET8_DIR):
        folder = os.path.join(DATASET8_DIR, marker_folder)
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if not fname.endswith(".csv"): continue
            fp = os.path.join(folder, fname)
            try:
                df = pd.read_csv(fp, comment="#", on_bad_lines="skip")
            except Exception:
                continue
            if "timestamp" not in df.columns or cfg["col"] not in df.columns:
                continue
            fs = 1000.0
            x = pd.to_numeric(df[cfg["col"]], errors="coerce").dropna().to_numpy(np.float32)
            if len(x) < args.min_points:
                continue

            if kind.upper() == "EDA":
                tonic = _eda_pipeline(x, fs)
                x4_f, x128_f = resample_any(tonic, fs, 4), resample_any(tonic, fs, 128)
                x4_r, x128_r = _resample_raw_linear(x, fs, 4), _resample_raw_linear(x, fs, 128)
                sr = 4
            else:
                filt = _ecg_pipeline(x, fs)
                x128_f, x128_r, sr = resample_any(filt, fs, 128), _resample_raw_linear(x, fs, 128), 128

            mp = int(args.min_points if not args.auto_min_points else round(args.interval * sr))
            for s, e, t0, t1 in windowize_by_index(len(x128_r if sr == 128 else x4_r),
                                                   sr, args.interval, args.step, mp):
                if sr == 4:
                    s128 = int(round(t0 * 128))
                    e128 = s128 + int(round(args.interval * 128))
                    if e128 > len(x128_r): continue
                    XN_list.append(x4_r[s:e,None]); XF_list.append(x4_f[s:e,None])
                    XN128_list.append(x128_r[s128:e128,None]); XF128_list.append(x128_f[s128:e128,None])
                else:
                    seg_n, seg_f = x128_r[s:e,None], x128_f[s:e,None]
                    XN_list.append(seg_n); XF_list.append(seg_f)
                    XN128_list.append(seg_n); XF128_list.append(seg_f)

                part = int(re.search(r"P(\d+)", fname, re.I).group(1)) if re.search(r"P(\d+)", fname, re.I) else -1
                task = int(re.search(r"S(\d+)", fname, re.I).group(1)) if re.search(r"S(\d+)", fname, re.I) else -1
                rows.append(dict(Participant=part, Task=task, Stimuli=marker_folder,
                                 Start_sec=t0, End_sec=t1, file=fp, signal=kind.upper(), sampling_rate=int(sr)))
    return _make_arrays(XN_list, XF_list, XN128_list, XF128_list, rows)
