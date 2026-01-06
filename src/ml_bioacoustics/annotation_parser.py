from __future__ import annotations

import os
from typing import Optional

import numpy as np


def parse_annotations_binary(
    tsv_path: Optional[str],
    audio_duration_s: float,
    segment_length_s: int,
) -> np.ndarray:
    """
    Segment-level binary labels:
      1 if any annotation overlaps the segment, else 0

    Expects a TSV with flexible column names containing:
      start* and end* (in seconds)

    If no TSV (or missing file) => returns zeros.
    """
    import pandas as pd  # local import

    n_segments = int(audio_duration_s // segment_length_s)
    if n_segments <= 0:
        return np.zeros((0,), dtype=int)

    y_true = np.zeros((n_segments,), dtype=int)

    if not tsv_path or not os.path.exists(tsv_path):
        return y_true

    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        return y_true

    cols = list(df.columns)
    start_col = next((c for c in cols if "start" in c.lower()), cols[0])
    end_col = next((c for c in cols if "end" in c.lower()), cols[1] if len(cols) > 1 else cols[0])

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
        overlap = (seg_starts < b) & (seg_ends > a)
        y_true[overlap] = 1

    return y_true
