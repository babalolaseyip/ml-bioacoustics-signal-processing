from __future__ import annotations

"""
ml_bioacoustics.pipeline

Run as:
  python -m ml_bioacoustics.pipeline --mode synthetic --out results
  python -m ml_bioacoustics.pipeline --mode real --audio data/blue_whale_sample.wav --annotations data/blue_whale_sample.tsv --method mse-gmm --segment-length 15 --out results
  python -m ml_bioacoustics.pipeline --mode real --audio data/blue_whale_sample.wav --annotations data/blue_whale_sample.tsv --method wt-hmm --segment-length 15 --out results
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Optional deps for WT path
try:
    import pywt  # type: ignore
except Exception:
    pywt = None

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:
    GaussianHMM = None

from .annotation_parser import parse_annotations_binary
from .feature_extraction import multiscale_sample_entropy, wavelet_frame_features
from .io import ensure_out_dir, load_audio_mono, save_json, segment_audio
from .models import gmm_unsupervised_predict, hmm_supervised_binary, map_clusters_to_binary
from .preprocessing import bandpass_filter, zscore
from .visualizations import plot_confusion_matrix_binary, plot_detection_timeline, plot_spectrogram


@dataclass
class RunConfig:
    mode: str = "synthetic"  # "synthetic" | "real"
    method: str = "mse-gmm"  # "mse-gmm" | "wt-hmm"
    audio_path: Optional[str] = None
    annotations_path: Optional[str] = None
    sr: int = 1000
    segment_length_s: int = 15
    out_dir: str = "results"
    seed: int = 42
    bp_low_hz: float = 10.0
    bp_high_hz: float = 40.0


def generate_synthetic_audio(sr: int, duration_s: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_mse_gmm(segments: np.ndarray, seed: int) -> np.ndarray:
    X = np.asarray([multiscale_sample_entropy(seg, max_scale=5) for seg in segments], dtype=np.float32)
    y_cluster = gmm_unsupervised_predict(X, n_components=2, seed=seed)
    return y_cluster


def run_wt_hmm(segments: np.ndarray, sr: int, seed: int, y_true: Optional[np.ndarray]) -> np.ndarray:
    # sequences per segment
    sequences: List[np.ndarray] = [wavelet_frame_features(seg, sr=sr) for seg in segments]

    # supervised if labels present
    if y_true is not None and len(y_true) == len(sequences) and np.unique(y_true).size >= 2:
        return hmm_supervised_binary(sequences, y_true=y_true, seed=seed)

    # fallback: unsupervised (energy-based)
    summaries = np.array([seq.mean(axis=0) if len(seq) else np.zeros((3,)) for seq in sequences], dtype=np.float32)
    cl = gmm_unsupervised_predict(summaries, n_components=2, seed=seed)

    if len(summaries) == 0:
        return np.zeros((0,), dtype=int)

    e0 = summaries[cl == 0, 0].mean() if np.any(cl == 0) else -np.inf
    e1 = summaries[cl == 1, 0].mean() if np.any(cl == 1) else -np.inf
    call_cluster = 0 if e0 >= e1 else 1
    return (cl == call_cluster).astype(int)


def pipeline(cfg: RunConfig) -> Dict:
    np.random.seed(cfg.seed)
    ensure_out_dir(cfg.out_dir)

    # Load/generate
    if cfg.mode == "synthetic":
        y, call_windows = generate_synthetic_audio(sr=cfg.sr, duration_s=60, seed=cfg.seed)
        audio_duration_s = 60.0
        y_true: Optional[np.ndarray] = derive_synthetic_segment_labels(call_windows, 60, cfg.segment_length_s)
    else:
        if not cfg.audio_path:
            raise ValueError("--audio is required when --mode real")
        y = load_audio_mono(cfg.audio_path, sr=cfg.sr)
        audio_duration_s = float(len(y) / cfg.sr)
        y_true = None
        if cfg.annotations_path:
            y_true = parse_annotations_binary(cfg.annotations_path, audio_duration_s, cfg.segment_length_s)

    # Preprocess
    y_f = bandpass_filter(y, sr=cfg.sr, low_hz=cfg.bp_low_hz, high_hz=cfg.bp_high_hz)
    y_n = zscore(y_f)

    # Spectrogram artifact
    spec_path = os.path.join(cfg.out_dir, "spectrogram.png")
    plot_spectrogram(y_n, sr=cfg.sr, out_path=spec_path, fmax=100.0)

    # Segment
    segments = segment_audio(y_n, sr=cfg.sr, segment_length_s=cfg.segment_length_s)

    # Align labels
    if y_true is not None:
        y_true = y_true[: len(segments)]

    # Run method
    if cfg.method == "mse-gmm":
        y_cluster = run_mse_gmm(segments, seed=cfg.seed)
        if y_true is not None and len(y_true) == len(y_cluster):
            y_pred = map_clusters_to_binary(y_cluster, y_true)
        else:
            # heuristic for visualization (no performance claims)
            x0 = np.asarray([multiscale_sample_entropy(seg, max_scale=5)[0] for seg in segments], dtype=np.float32)
            means = []
            for k in np.unique(y_cluster):
                means.append((k, float(x0[y_cluster == k].mean()) if np.any(y_cluster == k) else -np.inf))
            call_cluster = max(means, key=lambda z: z[1])[0] if means else 1
            y_pred = (y_cluster == call_cluster).astype(int)

    elif cfg.method == "wt-hmm":
        if pywt is None:
            raise ImportError("WT-HMM requires pywavelets. Install: pip install pywavelets")
        if GaussianHMM is None:
            raise ImportError("WT-HMM requires hmmlearn. Install: pip install hmmlearn")
        y_pred = run_wt_hmm(segments, sr=cfg.sr, seed=cfg.seed, y_true=y_true)

    else:
        raise ValueError("--method must be one of: mse-gmm, wt-hmm")

    # Detection plot
    det_path = os.path.join(cfg.out_dir, "detection_timeline.png")
    plot_detection_timeline(y_pred, cfg.segment_length_s, det_path, y_true=y_true)

    # Metrics + confusion matrix if labels exist
    metrics = compute_metrics_binary(y_true, y_pred) if y_true is not None else {}
    cm_path = None
    if y_true is not None and len(y_true) == len(y_pred) and len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_path = os.path.join(cfg.out_dir, "confusion_matrix.png")
        plot_confusion_matrix_binary(cm, cm_path)

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
            "Unsupervised outputs may be heuristically mapped for visualization. "
            "For rigorous reporting, provide TSV annotations and use printed metrics."
        ),
    }
    save_json(os.path.join(cfg.out_dir, "run_summary.json"), summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml_bioacoustics.pipeline",
        description="Run bioacoustics pipelines (MSE–GMM or WT–HMM) on synthetic or real data.",
    )
    p.add_argument("--mode", choices=["synthetic", "real"], required=True, help="Run mode.")
    p.add_argument("--method", choices=["mse-gmm", "wt-hmm"], default="mse-gmm", help="Detection method.")
    p.add_argument("--audio", dest="audio_path", default=None, help="Path to WAV (required for real mode).")
    p.add_argument("--annotations", dest="annotations_path", default=None, help="Path to TSV annotations (optional).")
    p.add_argument("--sr", type=int, default=1000, help="Target sampling rate (default: 1000 Hz).")
    p.add_argument("--segment-length", type=int, default=15, help="Segment length in seconds (default: 15).")
    p.add_argument("--out", dest="out_dir", default="results", help="Output directory for artifacts.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
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
    print(json.dumps({"metrics": summary.get("metrics", {}), "artifacts": summary.get("artifacts", {})}, indent=2))


if __name__ == "__main__":
    main()
