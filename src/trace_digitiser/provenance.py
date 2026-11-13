"""A record of how a digitisation was made.

Numbers pulled out of a picture are only as good as the account of how
they were obtained.  :func:`build_provenance` gathers everything needed to
reproduce or audit a digitisation - the image (by SHA-256), every setting,
the plot areas and axis calibrations that were used, how each trace was
found, any warnings, and the software versions - and
:func:`write_provenance` saves it as JSON beside the CSV output.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def package_version() -> str:
    """Installed version of trace-digitiser."""
    try:
        return metadata.version("trace-digitiser")
    except metadata.PackageNotFoundError:  # running from a source tree that is not installed
        return "0+unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {"python": platform.python_version()}
    for label, dist in (
        ("numpy", "numpy"), ("opencv", "opencv-python-headless"), ("pandas", "pandas"),
        ("pillow", "Pillow"), ("matplotlib", "matplotlib"), ("pytesseract", "pytesseract"),
    ):
        try:
            out[label] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[label] = None
    if out["opencv"] is None:
        try:
            import cv2

            out["opencv"] = cv2.__version__
        except Exception:
            pass
    try:
        from .axis_labels import tesseract_available

        if tesseract_available():
            import pytesseract

            out["tesseract"] = str(pytesseract.get_tesseract_version())
        else:
            out["tesseract"] = None
    except Exception:
        out["tesseract"] = None
    return out


def _plain(obj: Any) -> Any:
    """Convert numpy / pandas / Path values into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_provenance(
    image_path: Optional[Path],
    rgb: np.ndarray,
    parameters: dict,
    panels: list[dict],
    traces: list[tuple[dict, dict]],
    trace_data: pd.DataFrame,
    warnings: list[str],
) -> dict:
    """Assemble the provenance record for one digitisation.

    *panels* are the pipeline's panel dicts and *traces* the
    ``(panel, trace)`` pairs that were digitised.
    """
    image: dict[str, Any] = {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])}
    if image_path is not None:
        image["file"] = Path(image_path).name
        image["sha256"] = sha256_file(image_path)

    panel_records = []
    for p in panels:
        ycal = dict(p.get("y_calibration", {}))
        ycal.pop("ocr_candidates", None)
        xcal = dict(p.get("x_calibration") or {})
        xcal.pop("ocr_candidates", None)
        trace_records = []
        for q, tr in traces:
            if q is not p:
                continue
            record = {
                "trace_id": tr["trace_id"],
                "kind": tr.get("kind", "colour"),
                "median_rgb": tr["median_rgb"],
                "pixel_count": tr["pixel_count"],
            }
            if not trace_data.empty:
                sub = trace_data[(trace_data["panel_id"] == p["panel_id"]) & (trace_data["trace_id"] == tr["trace_id"])]
                record["columns"] = int(len(sub))
                record["columns_interpolated"] = int(sub["interpolated"].sum()) if "interpolated" in sub else 0
            trace_records.append(record)
        panel_records.append(
            {
                "panel_id": p["panel_id"],
                "found_by": p.get("source", ""),
                "axes": p.get("kind", "box"),
                "plot_area": p.get("edges", {}),
                "y_calibration": ycal,
                "x_calibration": xcal or None,
                "x_labels": [lab["text"] for lab in p.get("x_labels", [])],
                "traces": trace_records,
            }
        )

    return _plain(
        {
            "trace_digitiser_version": package_version(),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image": image,
            "parameters": parameters,
            "panels": panel_records,
            "warnings": list(warnings),
            "software": _versions(),
            "processing": "local; no data left this computer",
        }
    )


def write_provenance(path: str | Path, record: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(_plain(record), indent=2) + "\n", encoding="utf-8")
    return path
