"""Looking at a figure before digitising it, and turning positions clicked on it into values.

:func:`describe_figure` says what the automatic steps find in a figure - its tilt, its plot areas, each axis's tick
labels with what was read from them and the scale they give, the colours drawn in each plot, a colour bar, polar
axes - as plain values, so that a program with a graphical interface (the digitiser workbench) can show them and
let you correct them before digitising. :func:`pixels_to_values` turns points picked by hand into the figure's
units.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ._results import hex_colour
from .axis_labels import calibrate_axis, tesseract_available
from .frames import darkness_map
from .io import load_image
from .panel_detection import find_plot_panels
from .straighten import MIN_TILT_DEGREES, estimate_tilt
from .straighten import straighten as _turn_upright
from .trace_detection import detect_trace_masks


def _axis(rgb, dark, panel, axis: str, limit=None) -> Optional[dict[str, Any]]:
    fit, labels = calibrate_axis(rgb, dark, panel, axis, limit=limit)
    out: dict[str, Any] = {
        "labels": [
            {
                "box": [int(v) for v in lab.box],
                "position": round(float(lab.where), 2),
                "text": lab.text,
                "readings": list(dict.fromkeys(lab.readings)),
                "values": [float(v) for v in lab.values],
            }
            for lab in labels
        ]
    }
    if fit is not None:
        out.update(
            scale=fit.scale,
            confidence=fit.confidence,
            usable=bool(fit.usable),
            ticks=[[round(float(px), 2), float(v)] for px, v in sorted(fit.inliers)],
            a=float(fit.a),
            b=float(fit.b),
        )
    return out if (labels or fit is not None) else None


def describe_figure(
    image: Union[str, Path, np.ndarray],
    *,
    straighten: Union[str, bool, float] = "auto",
    plot_area=None,
    ocr: bool = True,
) -> dict[str, Any]:
    """What the automatic steps find in a figure, as plain values (for a graphical interface, or to look at).

    Returns a dict with ``width``, ``height``, ``tilt_degrees`` (anticlockwise; the figure is turned upright by that
    much if *straighten* is "auto" and it is 0.15 degrees or more: ``straightened`` says whether it was), and
    ``panels``: for each plot area its ``plot_area`` (left, top, right, bottom), ``kind`` ("box", "L" ...), its
    ``tick_marks`` (the rows of the ticks on the left-hand axis, the columns of those on the bottom one), and for
    each of its axes (``y_axis``, ``y2_axis`` on the right, ``x_axis``) the tick labels found (their boxes, what
    was read, the values those readings could stand for) and, if they agree on one, the scale (``ticks``: the
    (pixel, value) pairs it rests on, with its ``confidence``); and ``colours``, the colours drawn in it. Then
    ``colourbar`` and ``polar``, if the figure has them.
    """
    rgb = load_image(Path(image)) if not isinstance(image, np.ndarray) else np.ascontiguousarray(image)
    tilt = float(estimate_tilt(rgb)) if straighten == "auto" else 0.0
    turned = 0.0
    if straighten == "auto" and abs(tilt) >= MIN_TILT_DEGREES and plot_area is None:
        turned = tilt
    elif straighten not in ("auto", False, None, "never"):
        turned = float(straighten)
    if turned:
        rgb = _turn_upright(rgb, turned)
    dark = darkness_map(rgb)
    boxes = None
    if plot_area is not None:
        boxes = [tuple(float(v) for v in b) for b in (plot_area if hasattr(plot_area[0], "__len__") else [plot_area])]
    panels, _, _ = find_plot_panels(rgb, plot_area=boxes)
    use_ocr = ocr and tesseract_available()
    out_panels = []
    for p in panels:
        e = p["edges"]
        item: dict[str, Any] = {
            "panel_id": p["panel_id"],
            "plot_area": [round(e["left"], 2), round(e["top"], 2), round(e["right"], 2), round(e["bottom"], 2)],
            "kind": p.get("kind", "box"),
            # the tick marks found on the axis lines: the rows of those on the left-hand axis and the columns of
            # those on the bottom axis (for a program that lets you click on a tick)
            "tick_marks": {
                "y": [round(float(v), 2) for v in p.get("tick_rows", [])],
                "x": [round(float(v), 2) for v in p.get("tick_cols", [])],
            },
        }
        if use_ocr:
            item["y_axis"] = _axis(rgb, dark, p, "y")
            item["y2_axis"] = _axis(rgb, dark, p, "y2")
            item["x_axis"] = _axis(rgb, dark, p, "x")
        try:
            found = detect_trace_masks(rgb, p, darkness=dark)
        except Exception:  # noqa: BLE001 - a plot with nothing in it
            found = []
        item["colours"] = [
            {
                "hex": hex_colour(t["median_rgb"]),
                "pixels": int(t["pixel_count"]),
                "kind": t.get("kind", "colour"),
            }
            for t in found
        ]
        out_panels.append(item)
    from .heatmap import find_colourbar
    from .polar import find_polar_axes

    bar = find_colourbar(rgb, panels)
    polar = find_polar_axes(rgb) if not panels or len(panels) == 1 and panels[0].get("kind") != "box" else None
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "tilt_degrees": round(tilt, 3),
        "straightened": bool(turned),
        "turned_by_degrees": round(turned, 3),
        "ocr": bool(use_ocr),
        "panels": out_panels,
        "colourbar": None
        if bar is None
        else {
            "box": [round(bar["edges"][k], 2) for k in ("left", "top", "right", "bottom")],
            "vertical": bool(bar.get("vertical", True)),
        },
        "polar": None
        if polar is None
        else {
            "centre": [round(v, 2) for v in polar["centre"]],
            "radius": round(polar["radius"], 2),
            "rings": [round(v, 2) for v in polar["rings"]],
        },
    }


def _fit(ref: Sequence[Sequence[float]], scale: str) -> tuple[float, float]:
    pts = [(float(a), float(b)) for a, b in ref]
    if len(pts) < 2:
        raise ValueError("an axis needs at least two (pixel, value) reference points")
    px = np.array([q[0] for q in pts])
    vals = np.array([q[1] for q in pts])
    if scale == "log":
        if np.any(vals <= 0):
            raise ValueError("values on a log axis must be positive")
        vals = np.log10(vals)
    if np.ptp(px) == 0 or np.ptp(vals) == 0:
        raise ValueError("reference points must differ in pixel position and in value")
    a, b = np.polyfit(px, vals, 1)
    return float(a), float(b)


def pixels_to_values(
    points: Sequence[Sequence[float]],
    *,
    x_ref: Optional[Sequence[Sequence[float]]] = None,
    y_ref: Optional[Sequence[Sequence[float]]] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> list[tuple[Optional[float], Optional[float]]]:
    """Turn positions in a figure (pixel ``(x, y)``, from the top-left corner) into values on its axes.

    *x_ref* and *y_ref* are each two or more ``(pixel, value)`` pairs - a tick's column (for x) or row (for y) and
    the value it marks; with more than two, a straight line is fitted through them. An axis without references gives
    None. This is what to use for points picked by hand::

        pixels_to_values([(120, 80), (240, 60)], x_ref=[(41, 0), (388, 7)], y_ref=[(159, 0), (23, 300)])
    """
    xa = _fit(x_ref, x_scale) if x_ref else None
    ya = _fit(y_ref, y_scale) if y_ref else None
    out = []
    for x, y in points:
        xv = yv = None
        if xa:
            xv = xa[0] * float(x) + xa[1]
            if x_scale == "log":
                xv = 10.0**xv
        if ya:
            yv = ya[0] * float(y) + ya[1]
            if y_scale == "log":
                yv = 10.0**yv
        out.append((xv, yv))
    return out
