"""Image loading and CSV output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return an RGB uint8 ndarray."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def build_panel_metadata(panels: list[dict]) -> pd.DataFrame:
    """Build a metadata DataFrame from the final panel list."""
    rows: list[dict] = []
    for p in panels:
        cal = p.get("y_calibration", {})
        xcal = p.get("x_calibration", {})
        rows.append(
            {
                "panel_id": p["panel_id"],
                "x0": p["x0"],
                "x1": p["x1"],
                "y_top": p["y_top"],
                "y_bottom": p["y_bottom"],
                "source": p["source"],
                "score": p["score"],
                "gridline_y": str(p["gridline_y"]),
                "y_calibration_a": cal.get("a", ""),
                "y_calibration_b": cal.get("b", ""),
                "scale_type": cal.get("scale_type", ""),
                "x_calibration_a": xcal.get("a", ""),
                "x_calibration_b": xcal.get("b", ""),
                "x_scale_type": xcal.get("scale_type", ""),
                "x_labels": str([lab["text"] for lab in p.get("x_labels", [])]),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    trace_data: pd.DataFrame,
    panel_metadata: pd.DataFrame,
    summary_by_label: pd.DataFrame,
    output_prefix: str,
    output_dir: Optional[str | Path] = None,
) -> tuple[Path, Path, Optional[Path]]:
    """Write CSV outputs and return their paths.

    Returns ``(trace_path, metadata_path, summary_path_or_None)``.
    """
    base = Path(output_dir) if output_dir else Path(".")
    base.mkdir(parents=True, exist_ok=True)

    trace_path = base / f"{output_prefix}_automated_digitised_trace_by_column.csv"
    meta_path = base / f"{output_prefix}_automated_panel_metadata.csv"
    summary_path = base / f"{output_prefix}_automated_digitised_summary_by_label.csv"

    trace_data.to_csv(trace_path, index=False)
    panel_metadata.to_csv(meta_path, index=False)

    if not summary_by_label.empty:
        summary_by_label.to_csv(summary_path, index=False)
        return trace_path, meta_path, summary_path

    return trace_path, meta_path, None
