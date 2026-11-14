"""What the readers of points, bars, rows, heat maps and polar plots give back, and how they save it.

Each reader returns a :class:`FigureResult`: a table (``result.data``) with a row for each thing it found (a point,
a bar, a bin of a row, a cell of a heat map), in pixels and, where the axes are calibrated, in the figure's own
units, with the warnings, a provenance record and the paths of the files written.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

from .io import build_panel_metadata
from .models import Calibration
from .provenance import build_provenance, write_provenance
from .x_calibration import apply_x


@dataclass
class FigureResult:
    """The result of reading points, bars, rows, a heat map or a polar plot from a figure."""

    kind: str  # "points", "bars", "rows", "heatmap" or "polar"
    image_path: Optional[Path]
    rgb: np.ndarray
    panels: list[dict]
    data: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    csv_path: Optional[Path] = None
    provenance_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    straightened_path: Optional[Path] = None
    _draw: Optional[Callable[[np.ndarray], np.ndarray]] = field(default=None, repr=False)

    @property
    def panel_metadata(self) -> pd.DataFrame:
        return build_panel_metadata(self.panels) if self.panels else pd.DataFrame()

    def overlay_image(self) -> np.ndarray:
        """The figure with what was found drawn over it, as an RGB array."""
        canvas = self.rgb.copy()
        for p in self.panels:
            e = p["edges"]
            cv2.rectangle(
                canvas,
                (int(round(e["left"])), int(round(e["top"]))),
                (int(round(e["right"])), int(round(e["bottom"]))),
                (255, 0, 255),
                1,
            )
        return self._draw(canvas) if self._draw else canvas

    def overlay(self, path=None, show: bool = False) -> np.ndarray:
        """Draw what was found over the figure; save it to *path* (PNG) if given, show it if *show*."""
        img = self.overlay_image()
        if path is not None:
            from PIL import Image

            Image.fromarray(img).save(path)
        if show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10 * img.shape[0] / max(1, img.shape[1])))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        return img

    def __repr__(self) -> str:
        return (
            f"FigureResult(kind={self.kind!r}, panels={len(self.panels)}, rows={len(self.data)}, "
            f"warnings={len(self.warnings)})"
        )


# --------------------------------------------------------------------- axes


def y_values(panel: dict, y_px) -> np.ndarray:
    """Pixel rows -> values on the panel's y axis (0 to 1 up the plot area if it is not calibrated)."""
    cal = panel.get("y_calibration")
    if not cal:
        e = panel["edges"]
        return (e["bottom"] - np.asarray(y_px, dtype=float)) / max(1e-9, e["bottom"] - e["top"])
    return Calibration(**cal).y_to_value(y_px)


def y_pixels(panel: dict, values) -> np.ndarray:
    """Values on the panel's y axis -> pixel rows."""
    cal = panel.get("y_calibration")
    if not cal:
        e = panel["edges"]
        return e["bottom"] - np.asarray(values, dtype=float) * (e["bottom"] - e["top"])
    c = Calibration(**cal)
    v = np.asarray(values, dtype=float)
    if c.is_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(v)
    return (v - c.b) / c.a


def y_calibrated(panel: dict) -> bool:
    cal = panel.get("y_calibration")
    return bool(cal) and cal.get("scale_type") != "normalised_0_to_1"


def x_values(panel: dict, x_px) -> np.ndarray:
    """Pixel columns -> values on the panel's x axis (NaN if it is not calibrated)."""
    cal = panel.get("x_calibration")
    x = np.asarray(x_px, dtype=float)
    if not cal:
        return np.full(x.shape, np.nan)
    return apply_x(cal, x)


def x_norm(panel: dict, x_px) -> np.ndarray:
    e = panel["edges"]
    return (np.asarray(x_px, dtype=float) - e["left"]) / max(1e-9, e["right"] - e["left"])


def x_pixels(panel: dict, values) -> np.ndarray:
    cal = panel.get("x_calibration")
    if not cal:
        raise ValueError("the x axis is not calibrated")
    v = np.asarray(values, dtype=float)
    if str(cal.get("scale_type", "")).startswith("log"):
        v = np.log10(v)
    return (v - cal["b"]) / cal["a"]


# --------------------------------------------------------------------- colours


def colour_distance(rgb: np.ndarray, colour) -> np.ndarray:
    """Euclidean distance in RGB of every pixel of *rgb* from *colour*."""
    c = np.asarray(colour, dtype=np.float32).reshape(1, 1, 3)
    return np.sqrt(((rgb.astype(np.float32) - c) ** 2).sum(axis=2))


def background_colour(rgb: np.ndarray, box: Optional[tuple[int, int, int, int]] = None) -> tuple[int, int, int]:
    """The commonest colour (the paper) in *box* (x0, y0, x1, y1) of the image, or the whole image."""
    roi = rgb if box is None else rgb[box[1] : box[3] + 1, box[0] : box[2] + 1]
    flat = (roi.reshape(-1, 3) // 4).astype(np.int32)
    keys = flat[:, 0] * 4096 + flat[:, 1] * 64 + flat[:, 2]
    k = int(np.bincount(keys).argmax())
    return (int((k // 4096) * 4 + 2), int(((k // 64) % 64) * 4 + 2), int((k % 64) * 4 + 2))


def hex_colour(rgb_triplet) -> str:
    r, g, b = (int(round(float(v))) for v in rgb_triplet)
    return f"#{r:02x}{g:02x}{b:02x}"


def interior_box(panel: dict, inset: int = 2) -> tuple[int, int, int, int]:
    """The inside of a panel's plot area, clear of its axis lines: (x0, y0, x1, y1), inclusive."""
    e = panel["edges"]
    half = int(panel.get("line_thickness", 1)) // 2 + inset
    return (
        int(np.ceil(e["left"])) + half,
        int(np.ceil(e["top"])) + half,
        int(np.floor(e["right"])) - half,
        int(np.floor(e["bottom"])) - half,
    )


def exclusion_mask(shape: tuple[int, int], exclude, origin=(0, 0)) -> np.ndarray:
    """True where a pixel of an array of *shape*, whose top-left pixel is *origin* (x, y) in the image, falls in one
    of the *exclude* boxes (x0, y0, x1, y1)."""
    out = np.zeros(shape, dtype=bool)
    for box in exclude or []:
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        ox, oy = origin
        a0, a1 = max(0, y0 - oy), min(shape[0], y1 - oy + 1)
        b0, b1 = max(0, x0 - ox), min(shape[1], x1 - ox + 1)
        if a0 < a1 and b0 < b1:
            out[a0:a1, b0:b1] = True
    return out


# --------------------------------------------------------------------- saving


def finish(
    fig,
    kind: str,
    data: pd.DataFrame,
    draw: Optional[Callable[[np.ndarray], np.ndarray]],
    *,
    output_dir=None,
    write_outputs: bool = True,
    save_overlay: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> FigureResult:
    """Wrap up a reader: the provenance record, the files, the result."""
    provenance = build_provenance(fig.source_path, fig.rgb, fig.parameters, fig.panels, [], pd.DataFrame(), fig.notes)
    provenance["kind"] = kind
    provenance["rows"] = int(len(data))
    provenance["straightening"] = {
        "tilt_measured_degrees": round(float(fig.tilt_found), 3)
        if fig.parameters.get("straighten") == "auto"
        else None,
        "turned_clockwise_by_degrees": round(float(fig.turned_by), 3),
    }
    if extra:
        provenance.update(extra)
    result = FigureResult(
        kind=kind,
        image_path=fig.source_path,
        rgb=fig.rgb,
        panels=fig.panels,
        data=data,
        warnings=list(fig.notes),
        provenance=provenance,
        _draw=draw,
    )
    if write_outputs:
        base = Path(output_dir) if output_dir else Path(".")
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{fig.output_prefix}_{kind}"
        result.csv_path = base / f"{stem}.csv"
        data.to_csv(result.csv_path, index=False)
        result.provenance_path = write_provenance(base / f"{stem}_provenance.json", provenance)
        if fig.turned_by != 0:
            from PIL import Image

            result.straightened_path = base / f"{fig.output_prefix}_straightened.png"
            Image.fromarray(fig.rgb).save(result.straightened_path)
        if save_overlay:
            result.overlay_path = base / f"{stem}_overlay.png"
            result.overlay(result.overlay_path)
    return result
