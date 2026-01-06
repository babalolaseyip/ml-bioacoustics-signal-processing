from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


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


def plot_confusion_matrix_binary(cm: np.ndarray, out_path: str) -> None:
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
    plt.savefig(out_path, dpi=200)
    plt.close()
