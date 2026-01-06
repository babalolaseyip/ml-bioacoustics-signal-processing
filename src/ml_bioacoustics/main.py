"""
ml_bioacoustics.pipeline

Run as:
  python -m ml_bioacoustics.pipeline --mode synthetic --out results
  python -m ml_bioacoustics.pipeline --mode real --audio data/blue_whale_sample.wav --annotations data/blue_whale_sample.tsv --method mse-gmm --segment-length 15 --out results
  python -m ml_bioacoustics.pipeline --mode real --audio data/blue_whale_sample.wav --annotations data/blue_whale_sample.tsv --method wt-hmm --segment-length 15 --out results
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# deps
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Optional deps (WT-HMM path)
try:
    import pywt  # type: ignore
except Exception:
    pywt = None

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:
    GaussianHMM = None


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class RunConfig:
    mode: str = 'synthetic' # "synthetic" | "real"
    method: str = 'mse-gmm' # "mse-gmm" | "wt-hmm"
    audio_path: Optional[str] = None  # Fixed: added default value
    annotations_path: Optional[str] = None  # Fixed: added default value
    sr: int = 1000  # Fixed: added default value
    segment_length_s: int = 15  # Fixed: added default value
    out_dir: str = 'results'  # Fixed: added default value
    seed: int = 42  # Fixed: added default value
    # Preprocessing defaults (tuned for low-frequency baleen calls; adjust as needed)
    bp_low_hz: float = 10.0
    bp_high_hz: float = 40.0


# ----------------------------
# Utilities: I/O, segmentation
# ----------------------------
def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def segment_audio(y: np.ndarray, sr: int, segment_length_s: int) -> np.ndarray:
    seg_len = int(segment_length_s * sr)
    n_full = len(y) // seg_len
    if n_full <= 0:
        return np.empty((0, seg_len), dtype=np.float32)
    y = y[: n_full * seg_len]
    return y.reshape(n_full, seg_len)


# ----------------------------
# Preprocessing
# ----------------------------
def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if high <= low:
        return y.astype(np.float32, copy=False)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y).astype(np.float32)


def zscore(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    mu = float(np.mean(y))
    sd = float(np.std(y) + 1e-8)
    return ((y - mu) / sd).astype(np.float32)


# ----------------------------
# Annotation parsing (binary labels)
# Expect TSV with start/end in seconds (columns containing "start" and "end")
# ----------------------------
def parse_annotations_binary(
    tsv_path: str,
    audio_duration_s: float,
    segment_length_s: int,
) -> np.ndarray:
    """
    Returns y_true per segment: 1 if any annotation overlaps segment, else 0.

    TSV expected columns (flexible):
      - start_time / Start / start (seconds)
      - end_time / End / end (seconds)
      - call_type optional (ignored here)
    """
    import pandas as pd  # local import to keep startup lean

    n_segments = int(audio_duration_s // segment_length_s)
    if n_segments <= 0:
        return np.zeros((0,), dtype=int)

    y_true = np.zeros((n_segments,), dtype=int)

    if not tsv_path or not os.path.exists(tsv_path):
        return y_true

    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        return y_true

    # Find start/end cols flexibly
    cols = list(df.columns)
    start_col = next((c for c in cols if "start" in c.lower()), cols[0])
    end_col = next((c for c in cols if "end" in c.lower()), cols[1] if len(cols) > 1 else cols[0])

    # numeric conversion
    df = df.dropna(subset=[start_col, end_col]).copy()
    df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
    df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
    df = df.dropna(subset=[start_col, end_col])
    df = df[df[end_col] > df[start_col]]
    if df.empty:
        return y_true

    seg_starts = np.arange(n_segments) * segment_length_s
    seg_ends = seg_starts + segment_length_s

    for _, r in df.iterrows():
        a = float(r[start_col])
        b = float(r[end_col])
        # overlap test
        overlap = (seg_starts < b) & (seg_ends > a)
        y_true[overlap] = 1

    return y_true


# ----------------------------
# Features: MSE (Sample Entropy) and Wavelet
# ----------------------------
def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Simple sample entropy implementation (O(N^2)). Suitable for demo/portfolio.
    For speed on large datasets, replace with optimized library or vectorized version.
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
        # coarse-graining
        L = (len(x) // tau) * tau
        if L <= 0:
            feats.append(0.0)
            continue
        cg = x[:L].reshape(-1, tau).mean(axis=1)
        feats.append(sample_entropy(cg, m=m, r=r))
    return np.array(feats, dtype=np.float32)


def wavelet_frame_features(
    x: np.ndarray,
    sr: int,
    wavelet: str = "morl",
    scales: Optional[np.ndarray] = None,
    hop_length: int = 128,
    frame_length: int = 512,
) -> np.ndarray:
    """
    Returns per-frame features: [energy_sum, centroid, entropy] for each frame.
    Uses CWT magnitude energy across scales for each frame.
    """
    if pywt is None:
        raise ImportError("pywavelets (pywt) is required for WT-HMM method. Install: pip install pywavelets")

    if scales is None:
        scales = np.logspace(1, 4, 32)

    x = np.asarray(x, dtype=np.float32)
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).T  # (n_frames, frame_length)

    feats = []
    for fr in frames:
        coeffs, freqs = pywt.cwt(fr, scales, wavelet, sampling_period=1.0 / sr)
        E = (np.abs(coeffs) ** 2).astype(np.float32)  # (n_scales, frame_length)
        # Sum energy per time sample within frame -> collapse to 1 vector then summarize
        e_time = E.sum(axis=0) + 1e-10
        # Centroid: freq-weighted energy across scales, averaged over time within frame
        E_scale = E.sum(axis=1) + 1e-10  # energy per scale
        centroid = float((freqs * E_scale).sum() / E_scale.sum())
        # Entropy across scales
        p = (E_scale / E_scale.sum()).astype(np.float32)
        ent = float(-(p * np.log2(p + 1e-10)).sum())
        feats.append([float(e_time.mean()), centroid, ent])

    if not feats:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(feats, dtype=np.float32)


# ----------------------------
# Models / prediction
# ----------------------------
def gmm_unsupervised_predict(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0,), dtype=int)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    gmm.fit(X)
    return gmm.predict(X).astype(int)


def map_clusters_to_binary(y_cluster: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Map cluster IDs to binary labels (0/1) by majority vote in each cluster.
    Safe for demo: prevents "label inversion" when using unsupervised GMM.
    """
    y_out = np.zeros_like(y_cluster, dtype=int)
    for k in np.unique(y_cluster):
        idx = np.where(y_cluster == k)[0]
        if len(idx) == 0:
            continue
        # majority label among true labels
        maj = int(np.round(np.mean(y_true[idx]) >= 0.5))
        y_out[idx] = maj
    return y_out


def hmm_supervised_binary(
    sequences: List[np.ndarray],
    y_true: np.ndarray,
    seed: int,
    n_states: int = 6,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Supervised WT-HMM for binary classification:
      - Fit one HMM on "noise" sequences
      - Fit one HMM on "call" sequences
      - Predict segment by which model gives higher log-likelihood

    Requires hmmlearn.
    """
    if GaussianHMM is None:
        raise ImportError("hmmlearn is required for WT-HMM method. Install: pip install hmmlearn")

    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]

    # If one class missing, fall back to unsupervised
    if len(idx0) < 2 or len(idx1) < 2:
        # fallback: cluster by segment-level summary
        summaries = np.array([seq.mean(axis=0) if len(seq) else np.zeros((3,)) for seq in sequences], dtype=np.float32)
        cl = gmm_unsupervised_predict(summaries, n_components=2, seed=seed)
        # map clusters to "call" as the cluster with higher mean energy
        if summaries.shape[0] > 0:
            e_means = []
            for k in np.unique(cl):
                e_means.append((k, summaries[cl == k, 0].mean() if np.any(cl == k) else -np.inf))
            call_cluster = max(e_means, key=lambda x: x[1])[0]
            return (cl == call_cluster).astype(int)
        return np.zeros((len(sequences),), dtype=int)

    def _stack(idxs: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        seqs = [sequences[i] for i in idxs if len(sequences[i]) > 0]
        if not seqs:
            return np.zeros((0, 3), dtype=np.float32), []
        lengths = [len(s) for s in seqs]
        X = np.vstack(seqs).astype(np.float32)
        return X, lengths

    X0, L0 = _stack(idx0)
    X1, L1 = _stack(idx1)

    # Fit two models
    m0 = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, random_state=seed)
    m1 = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, random_state=seed)
    m0.fit(X0, lengths=L0)
    m1.fit(X1, lengths=L1)

    # Predict by comparing log-likelihoods
    y_pred = np.zeros((len(sequences),), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            y_pred[i] = 0
            continue
        s0 = m0.score(seq)
        s1 = m1.score(seq)
        y_pred[i] = 1 if s1 > s0 else 0
    return y_pred


# ----------------------------
# Visualizations
# ----------------------------
def plot_spectrogram(y: np.ndarray, sr: int, out_path: str, fmax: float = 100.0) -> None:
    plt.figure(figsize=(11, 4))
    plt.specgram(y, NFFT=2048, Fs=sr, noverlap=1024)
    plt.ylim(0, fmax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram (0–100 Hz)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_detection_timeline(
    y_pred: np.ndarray,
    segment_length_s: int,
    out_path: str,
    y_true: Optional[np.ndarray] = None,
) -> None:
    t = np.arange(len(y_pred)) * segment_length_s
    plt.figure(figsize=(11, 2.8))
    plt.step(t, y_pred, where="post", linewidth=2, label="Pred")
    if y_true is not None and len(y_true) == len(y_pred):
        plt.step(t, y_true, where="post", alpha=0.7, label="True")
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Call")
    plt.title(f"Detection timeline ({segment_length_s}s segments)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Synthetic data generator
# ----------------------------
def generate_synthetic_audio(sr: int, duration_s: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produces a low-frequency call-like signal + noise and binary segment labels.
    Labels are segment-level (15s by default in CLI).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, sr * duration_s, endpoint=False)
    y = (0.15 * rng.standard_normal(len(t))).astype(np.float32)

    call_windows = [(10, 18), (28, 36), (44, 52)]
    for (a, b) in call_windows:
        i0, i1 = int(a * sr), int(b * sr)
        tt = t[i0:i1]
        f0, f1 = 18.0, 28.0
        k = (f1 - f0) / max((b - a), 1e-6)
        phase = 2 * np.pi * (f0 * (tt - a) + 0.5 * k * (tt - a) ** 2)
        y[i0:i1] += (0.6 * np.sin(phase)).astype(np.float32)

    # No per-sample truth here; return empty placeholder; segment labels derived by overlap in main
    return y, np.array(call_windows, dtype=np.float32)


def derive_synthetic_segment_labels(call_windows: np.ndarray, duration_s: int, segment_length_s: int) -> np.ndarray:
    n_segments = duration_s // segment_length_s
    y_true = np.zeros((n_segments,), dtype=int)
    for i in range(n_segments):
        a = i * segment_length_s
        b = a + segment_length_s
        if any((a < w[1] and b > w[0]) for w in call_windows):
            y_true[i] = 1
    return y_true


# ----------------------------
# Evaluation
# ----------------------------
def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ----------------------------
# Main pipeline runners
# ----------------------------
def run_mse_gmm(
    segments: np.ndarray,
    sr: int,
    seed: int,
    max_scale: int = 5,
    m: int = 2,
    r: float = 0.2,
) -> np.ndarray:
    # Feature per segment: MSE vector
    X = []
    for seg in segments:
        feat = multiscale_sample_entropy(seg, max_scale=max_scale, m=m, r=r)
        X.append(feat)
    X = np.asarray(X, dtype=np.float32)
    # Unsupervised GMM clusters (2 for binary)
    y_cluster = gmm_unsupervised_predict(X, n_components=2, seed=seed)
    return y_cluster


def run_wt_hmm(
    segments: np.ndarray,
    sr: int,
    seed: int,
    y_true: Optional[np.ndarray],
) -> np.ndarray:
    # Per segment: sequence of frame features (energy, centroid, entropy)
    sequences: List[np.ndarray] = []
    for seg in segments:
        seq = wavelet_frame_features(seg, sr=sr)
        sequences.append(seq)

    # If labels exist -> supervised HMM (two models) gives meaningful predictions
    if y_true is not None and len(y_true) == len(sequences) and np.unique(y_true).size >= 2:
        return hmm_supervised_binary(sequences, y_true=y_true, seed=seed)

    # Otherwise: unsupervised fallback using segment-level summaries + GMM
    summaries = np.array([seq.mean(axis=0) if len(seq) else np.zeros((3,)) for seq in sequences], dtype=np.float32)
    cl = gmm_unsupervised_predict(summaries, n_components=2, seed=seed)

    # Choose "call" cluster as the one with higher mean energy summary (feature 0)
    if len(summaries) == 0:
        return np.zeros((0,), dtype=int)
    e0 = summaries[cl == 0, 0].mean() if np.any(cl == 0) else -np.inf
    e1 = summaries[cl == 1, 0].mean() if np.any(cl == 1) else -np.inf  # Fixed: npinf -> np.inf
    call_cluster = 0 if e0 >= e1 else 1
    return (cl == call_cluster).astype(int)


def load_real_audio(audio_path: str, sr: int) -> np.ndarray:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y.astype(np.float32)


def pipeline(cfg: RunConfig) -> Dict:
    np.random.seed(cfg.seed)

    ensure_out_dir(cfg.out_dir)

    # Load/generate audio
    if cfg.mode == "synthetic":
        y, call_windows = generate_synthetic_audio(sr=cfg.sr, duration_s=60, seed=cfg.seed)
        audio_duration_s = 60.0
        y_true = derive_synthetic_segment_labels(call_windows, duration_s=60, segment_length_s=cfg.segment_length_s)
    else:
        if not cfg.audio_path:
            raise ValueError("--audio is required when --mode real")
        y = load_real_audio(cfg.audio_path, sr=cfg.sr)
        audio_duration_s = float(len(y) / cfg.sr)
        y_true = None
        if cfg.annotations_path:
            y_true = parse_annotations_binary(cfg.annotations_path, audio_duration_s, cfg.segment_length_s)

    # Preprocess: bandpass + zscore
    y_f = bandpass_filter(y, sr=cfg.sr, low_hz=cfg.bp_low_hz, high_hz=cfg.bp_high_hz)
    y_n = zscore(y_f)

    # Save spectrogram of full audio (preprocessed)
    spec_path = os.path.join(cfg.out_dir, "spectrogram.png")
    plot_spectrogram(y_n, sr=cfg.sr, out_path=spec_path, fmax=100.0)

    # Segment
    segments = segment_audio(y_n, sr=cfg.sr, segment_length_s=cfg.segment_length_s)

    # Align y_true to segments if present
    if y_true is not None:
        y_true = y_true[: len(segments)]

    # Run chosen method
    if cfg.method == "mse-gmm":
        y_cluster = run_mse_gmm(segments, sr=cfg.sr, seed=cfg.seed)
        if y_true is not None and len(y_true) == len(y_cluster):
            y_pred = map_clusters_to_binary(y_cluster, y_true)
        else:
            # heuristic: cluster with higher mean "entropy at scale 1" -> call
            # (safe but not a performance claim; purely for visualization)
            # Recompute just for the heuristic mapping
            X = np.asarray([multiscale_sample_entropy(seg, max_scale=5)[0] for seg in segments], dtype=np.float32)
            means = []
            for k in np.unique(y_cluster):
                means.append((k, float(X[y_cluster == k].mean()) if np.any(y_cluster == k) else -np.inf))
            call_cluster = max(means, key=lambda x: x[1])[0] if means else 1
            y_pred = (y_cluster == call_cluster).astype(int)
    elif cfg.method == "wt-hmm":
        if pywt is None:
            raise ImportError("WT-HMM requires pywavelets. Install: pip install pywavelets")
        y_pred = run_wt_hmm(segments, sr=cfg.sr, seed=cfg.seed, y_true=y_true)
    else:
        raise ValueError("--method must be one of: mse-gmm, wt-hmm")

    # Plots
    det_path = os.path.join(cfg.out_dir, "detection_timeline.png")
    plot_detection_timeline(y_pred, cfg.segment_length_s, det_path, y_true=y_true)

    # Metrics (only if we have annotations)
    metrics = compute_metrics_binary(y_true, y_pred) if y_true is not None else {}

    # Confusion matrix (if labels exist)
    cm_path = None
    if y_true is not None and len(y_true) == len(y_pred) and len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(4.2, 3.6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion matrix")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(int(v)), ha="center", va="center")
        plt.tight_layout()
        cm_path = os.path.join(cfg.out_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()

    # Write summary JSON
    summary = {
        "mode": cfg.mode,
        "method": cfg.method,
        "sr": cfg.sr,
        "segment_length_s": cfg.segment_length_s,
        "audio_path": cfg.audio_path,
        "annotations_path": cfg.annotations_path,
        "out_dir": cfg.out_dir,
        "artifacts": {
            "spectrogram": spec_path,
            "detection_timeline": det_path,
            "confusion_matrix": cm_path,
        },
        "metrics": metrics,
        "notes": (
            "Metrics are computed only when annotations are supplied. "
            "Unsupervised methods may produce cluster IDs that are mapped for visualization; "
            "for rigorous reporting, provide annotations and use the printed metrics."
        ),
    }
    save_json(os.path.join(cfg.out_dir, "run_summary.json"), summary)
    return summary


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml_bioacoustics.pipeline",
        description="Run bioacoustics pipelines (MSE–GMM or WT–HMM) on synthetic or real data.",
    )
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic", help="Run mode (default: synthetic).")
    p.add_argument("--method", choices=["mse-gmm", "wt-hmm"], default="mse-gmm", help="Detection method (default: mse-gmm).")
    p.add_argument("--audio", dest="audio_path", default=None, help="Path to WAV (required for real mode).")
    p.add_argument("--annotations", dest="annotations_path", default=None, help="Path to TSV annotations (optional).")
    p.add_argument("--sr", type=int, default=1000, help="Target sampling rate (default: 1000 Hz).")
    p.add_argument("--segment-length", type=int, default=15, help="Segment length in seconds (default: 15).")
    p.add_argument("--out", dest="out_dir", default="results", help="Output directory for artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    # optional preprocessing tweaks
    p.add_argument("--bp-low", type=float, default=10.0, help="Bandpass low cutoff Hz (default: 10).")
    p.add_argument("--bp-high", type=float, default=40.0, help="Bandpass high cutoff Hz (default: 40).")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    cfg = RunConfig(
        mode=args.mode,
        method=args.method,
        audio_path=args.audio_path,
        annotations_path=args.annotations_path,
        sr=args.sr,
        segment_length_s=args.segment_length,
        out_dir=args.out_dir,
        seed=args.seed,
        bp_low_hz=float(args.bp_low),
        bp_high_hz=float(args.bp_high),
    )

    summary = pipeline(cfg)
    # Print a compact console summary for CI logs / users
    print(json.dumps({"metrics": summary.get("metrics", {}), "artifacts": summary.get("artifacts", {})}, indent=2))


if __name__ == "__main__":
    main()