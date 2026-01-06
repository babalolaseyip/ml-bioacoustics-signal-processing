from __future__ import annotations

from typing import List, Optional

import librosa
import numpy as np

# Optional dependency for WT path
try:
    import pywt  # type: ignore
except Exception:
    pywt = None


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Simple Sample Entropy (O(N^2)) — OK for demo/portfolio, not optimized for huge audio.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < (m + 2):
        return 0.0
    sd = float(np.std(x) + 1e-8)
    tol = r * sd

    def _phi(mm: int) -> int:
        count = 0
        for i in range(n - mm):
            xi = x[i : i + mm]
            for j in range(i + 1, n - mm):
                xj = x[j : j + mm]
                if np.max(np.abs(xi - xj)) <= tol:
                    count += 1
        return count

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def multiscale_sample_entropy(x: np.ndarray, max_scale: int = 5, m: int = 2, r: float = 0.2) -> np.ndarray:
    feats: List[float] = []
    x = np.asarray(x, dtype=np.float32)
    for tau in range(1, max_scale + 1):
        L = (len(x) // tau) * tau
        if L <= 0:
            feats.append(0.0)
            continue
        cg = x[:L].reshape(-1, tau).mean(axis=1)
        feats.append(sample_entropy(cg, m=m, r=r))
    return np.asarray(feats, dtype=np.float32)


def wavelet_frame_features(
    x: np.ndarray,
    sr: int,
    wavelet: str = "morl",
    scales: Optional[np.ndarray] = None,
    hop_length: int = 128,
    frame_length: int = 512,
) -> np.ndarray:
    """
    Per-frame wavelet features for WT-HMM:
      [energy_mean, spectral_centroid_hz, scale_entropy]
    """
    if pywt is None:
        raise ImportError("pywavelets (pywt) is required for WT-HMM. Install: pip install pywavelets")

    if scales is None:
        scales = np.logspace(1, 4, 32)

    x = np.asarray(x, dtype=np.float32)
    if len(x) < frame_length:
        return np.zeros((0, 3), dtype=np.float32)

    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).T

    feats = []
    for fr in frames:
        coeffs, freqs = pywt.cwt(fr, scales, wavelet, sampling_period=1.0 / sr)
        E = (np.abs(coeffs) ** 2).astype(np.float32)

        # Energy per scale
        E_scale = E.sum(axis=1) + 1e-10

        # Mean energy over time
        energy_mean = float(E.sum(axis=0).mean())

        # Centroid: freq-weighted
        centroid = float((freqs * E_scale).sum() / E_scale.sum())

        # Entropy across scales
        p = (E_scale / E_scale.sum()).astype(np.float32)
        ent = float(-(p * np.log2(p + 1e-10)).sum())

        feats.append([energy_mean, centroid, ent])

    if not feats:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)
