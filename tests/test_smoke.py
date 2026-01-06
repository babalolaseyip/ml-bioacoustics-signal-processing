import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_smoke_creates_artifacts(tmp_path):
    """
    Smoke test: run synthetic pipeline and ensure key artifacts are created.
    CI should fail loudly if any artifact is missing.
    """

    out_dir = tmp_path / "results"

    cmd = [
        sys.executable,
        "-m",
        "ml_bioacoustics.pipeline",
        "--mode",
        "synthetic",
        "--out",
        str(out_dir),
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        "Pipeline failed to run.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )

    spectrogram = out_dir / "spectrogram.png"
    detection = out_dir / "detection_timeline.png"
    summary = out_dir / "run_summary.json"

    assert spectrogram.exists(), "Missing spectrogram.png"
    assert detection.exists(), "Missing detection_timeline.png"
    assert summary.exists(), "Missing run_summary.json"

    with open(summary, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("mode") == "synthetic"
    assert "artifacts" in data
