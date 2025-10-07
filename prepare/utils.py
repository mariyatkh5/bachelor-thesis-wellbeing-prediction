# utils.py
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Iterable, Tuple

# ---------------- Range parser ----------------
def rng(text_or_list: Optional[Iterable[int] | str]) -> Optional[List[int]]:
    """Accepts list[int] or '1,2,5-7' → [1,2,5,6,7]."""
    if text_or_list is None:
        return None
    if isinstance(text_or_list, (list, tuple, set)):
        return sorted({int(x) for x in text_or_list})
    out: List[int] = []
    for part in str(text_or_list).split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


# ---------------- Windowing ----------------
def windowize_by_index(n: int, fs: float, interval: float, step: float, min_points: int):
    """Return list of (start, end, t0, t1) for full windows with ≥ min_points."""
    if n <= 0 or fs <= 0 or interval <= 0 or step <= 0:
        return []
    win = int(round(interval * fs))
    hop = int(round(step * fs))
    if win <= 0 or hop <= 0:
        return []
    out, s = [], 0
    while s + win <= n:
        e = s + win
        if (e - s) >= min_points:
            out.append((s, e, s / fs, e / fs))
        s += hop
    return out


# ---------------- Sampling rate inference ----------------
def infer_fs_from_timestamp(t_like) -> Optional[float]:
    """Estimate sampling rate (Hz) from time column."""
    try:
        t = pd.to_numeric(t_like, errors="coerce").to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        if t.size >= 3:
            if np.nanmedian(t) > 1e10:
                t = t / 1000.0
            dt = np.diff(t)
            dt = dt[(dt > 0) & np.isfinite(dt)]
            if dt.size:
                fs = 1.0 / np.median(dt)
                if 0.5 < fs < 10000:
                    return float(fs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            tdt = pd.to_datetime(t_like, errors="coerce")
        tdt = tdt.dropna()
        if tdt.size >= 3:
            dt = tdt.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
            dt = dt[(dt > 0) & np.isfinite(dt)]
            if dt.size:
                fs = 1.0 / np.median(dt)
                if 0.5 < fs < 10000:
                    return float(fs)
    except Exception:
        pass
    return None


# ---------------- Numeric filler ----------------
def _fill_numeric_series(series: pd.Series) -> np.ndarray:
    """Convert to float, replace inf/NaN, interpolate, bfill/ffill."""
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(limit_direction="both")
    s = s.bfill().ffill()
    return s.to_numpy(dtype=np.float32)


# ---------------- Label type inference ----------------
def infer_y(y: np.ndarray) -> Tuple[np.ndarray, str]:
    """Infer label type: binary, multiclass, or continuous."""
    y_num = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
    if np.all(~np.isnan(y_num)):
        u = np.unique(y_num.astype(float))
        if set(u).issubset({0.0, 1.0}):
            return y_num.astype(np.int64), "binary"
        if u.size <= 10 and np.allclose(u, np.round(u)):
            return y_num.astype(np.int64), "multiclass"
        return y_num.astype(np.float32), "continuous"
    _, inv = np.unique(y.astype(str), return_inverse=True)
    return inv.astype(np.int64), "multiclass"


# ---------------- Z-score per window ----------------
def zscore_window(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization per window (axis=1)."""
    if X.size == 0:
        return X
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32, copy=False)
