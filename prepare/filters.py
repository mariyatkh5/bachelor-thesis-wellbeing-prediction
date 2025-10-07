import numpy as np

# Optional SciPy (filter/resample)
try:
    from scipy.signal import butter, sosfiltfilt, sosfilt, resample_poly
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Optional PyWavelets
try:
    import pywt
    PYWAVELETS_OK = True
except Exception:
    PYWAVELETS_OK = False


def zscore_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization of 1D signal."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return ((x - mu) / (sd + eps)).astype(np.float32, copy=False)


def eda_lowpass_5hz(x: np.ndarray, fs: float) -> np.ndarray:
    """2nd order lowpass at 5 Hz for EDA."""
    if (not SCIPY_OK) or fs <= 0:
        return x.astype(np.float32, copy=False)
    wn = min(max(5.0 / (0.5 * fs), 1e-4), 0.999)
    sos = butter(2, wn, btype="low", output="sos")
    try:
        y = sosfiltfilt(sos, x)
    except ValueError:
        y = sosfilt(sos, x)
    return y.astype(np.float32, copy=False)


def eda_wavelet_daub5(x: np.ndarray, level: int | None = None) -> np.ndarray:
    """Wavelet denoising with Daubechies-5, soft threshold."""
    if not PYWAVELETS_OK:
        return x.astype(np.float32, copy=False)
    x = np.asarray(x, dtype=np.float64)
    w = "db5"
    if level is None:
        level = pywt.dwt_max_level(len(x), pywt.Wavelet(w).dec_len)
        level = max(1, min(level, 6))
    coeffs = pywt.wavedec(x, w, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    thr = sigma * np.sqrt(2.0 * np.log(len(x))) if sigma > 0 else 0.0
    coeffs_th = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    y = pywt.waverec(coeffs_th, w)
    if len(y) != len(x):
        y = y[:len(x)]
    return y.astype(np.float32, copy=False)


def eda_tonic_phasic(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Split EDA into tonic (low freq) and phasic (residual)."""
    if (not SCIPY_OK) or fs <= 0:
        tonic = x.astype(np.float32, copy=False)
        phasic = (x - tonic).astype(np.float32, copy=False)
        return tonic, phasic
    wc = min(max(0.05 / (0.5 * fs), 1e-5), 0.999)
    sos = butter(2, wc, btype="low", output="sos")
    try:
        tonic = sosfiltfilt(sos, x)
    except ValueError:
        tonic = sosfilt(sos, x)
    phasic = x - tonic
    return tonic.astype(np.float32, copy=False), phasic.astype(np.float32, copy=False)


def ecg_bandpass_10_75(x: np.ndarray, fs: float) -> np.ndarray:
    """ECG bandpass 10–75 Hz."""
    if (not SCIPY_OK) or fs <= 0 or (0.5 * fs) <= 75.0:
        return x.astype(np.float32, copy=False)
    lo = 10.0 / (0.5 * fs)
    hi = 75.0 / (0.5 * fs)
    lo = min(max(lo, 1e-5), 0.999)
    hi = min(max(hi, lo + 1e-5), 0.999)
    sos = butter(4, [lo, hi], btype="band", output="sos")
    try:
        y = sosfiltfilt(sos, x)
    except ValueError:
        y = sosfilt(sos, x)
    return y.astype(np.float32, copy=False)


def resample_any(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling, fallback: linear interpolation."""
    x = np.asarray(x, dtype=np.float64)
    if fs_in == fs_out or len(x) <= 1:
        return x.astype(np.float32, copy=False)
    if SCIPY_OK:
        from fractions import Fraction
        frac = Fraction(fs_out / fs_in).limit_denominator(2000)
        y = resample_poly(x, up=frac.numerator, down=frac.denominator)
        return y.astype(np.float32, copy=False)

    N = len(x)
    new_N = int(round(N * fs_out / fs_in))
    if new_N <= 1:
        return x.astype(np.float32, copy=False)
    xi = np.linspace(0, N - 1, N, dtype=np.float64)
    xo = np.linspace(0, N - 1, new_N, dtype=np.float64)
    y = np.interp(xo, xi, x)
    return y.astype(np.float32, copy=False)
