from __future__ import annotations

import json
import os
from typing import Any, Dict

import librosa
import numpy as np


def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_audio_mono(audio_path: str, sr: int) -> np.ndarray:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32)


def segment_audio(y: np.ndarray, sr: int, segment_length_s: int) -> np.ndarray:
    seg_len = int(segment_length_s * sr)
    if seg_len <= 0:
        raise ValueError("segment_length_s must be positive.")
    n_full = len(y) // seg_len
    if n_full <= 0:
        return np.empty((0, seg_len), dtype=np.float32)
    y = y[: n_full * seg_len]
    return y.reshape(n_full, seg_len).astype(np.float32, copy=False)
