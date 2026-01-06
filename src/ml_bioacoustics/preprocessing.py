from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    nyq = 0.5 * sr
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if high <= low:
        return y.astype(np.float32, copy=False)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y).astype(np.float32)


def zscore(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    mu = float(np.mean(y))
    sd = float(np.std(y) + 1e-8)
    return ((y - mu) / sd).astype(np.float32)
