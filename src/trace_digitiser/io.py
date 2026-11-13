"""Image loading and CSV output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return an RGB uint8 ndarray.

    Transparent areas are placed on white (a figure saved with a transparent
    background would otherwise load as black), and a rotation recorded by a
    scanner or camera in the file's EXIF data is applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        hint = " A PDF page must be exported as an image (PNG) first." if path.suffix.lower() == ".pdf" else ""
        raise ValueError(f"Cannot read {path} as an image.{hint}") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return np.array(img.convert("RGB"))


def build_panel_metadata(panels: list[dict]) -> pd.DataFrame:
    """Build a metadata DataFrame from the final panel list: one row per panel."""
    rows: list[dict] = []
    for p in panels:
        cal = p.get("y_calibration", {})
        xcal = p.get("x_calibration") or {}
        edges = p.get("edges") or {}
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
                # Sub-pixel plot-area edges (what the calibrations refer to).
                "plot_left": edges.get("left", p["x0"]),
                "plot_right": edges.get("right", p["x1"]),
                "plot_top": edges.get("top", p["y_top"]),
                "plot_bottom": edges.get("bottom", p["y_bottom"]),
                "axes": p.get("kind", "box"),
                # How much the y calibration can be trusted, and what it rests on.
                "y_calibration_method": cal.get("method", ""),
                "y_calibration_confidence": cal.get("confidence", ""),
                "y_labels_agreeing": len(cal.get("inliers", [])) if cal.get("method", "").startswith("tick") else "",
                "y_labels_found": cal.get("n_labels") or "",
                "y_label_height_px": cal.get("label_px") or "",
                "x_calibration_method": xcal.get("method", ""),
                "x_calibration_confidence": xcal.get("confidence", ""),
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
