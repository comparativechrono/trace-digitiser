"""Reading the points of a scatter plot, a dot plot or a line drawn with markers, with their error bars."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd

from ._figure import prepare_figure
from ._results import FigureResult, finish, hex_colour, interior_box, x_norm, x_values, y_values
from .markers import find_error_bars, find_markers
from .trace_detection import detect_trace_masks, parse_colour


def _auto_colours(rgb, panel, exclude, dark) -> list[tuple[int, int, int]]:
    """The colours of what is drawn in a panel (from the trace detector), as candidates for markers."""
    found = detect_trace_masks(rgb, panel, exclude=exclude, achromatic="auto", darkness=dark)
    return [tuple(int(v) for v in t["median_rgb"]) for t in found]


def digitise_points(
    image_path: Union[str, Path, np.ndarray],
    *,
    colours: Optional[Sequence] = None,
    colour_tolerance: float = 60.0,
    marker_size: Optional[float] = None,
    error_bars: str = "auto",
    output_dir=None,
    output_prefix: Optional[str] = None,
    write_outputs: bool = True,
    save_overlay: bool = False,
    **figure,
) -> FigureResult:
    """Read the points (markers) of a figure: their positions, in pixels and in the axes' units, and their error bars.

    Parameters
    ----------
    image_path : str, Path or ndarray
        The figure.
    colours : list of colours, optional
        The colours of the markers (``"#rrggbb"`` or ``(r, g, b)``), one series each. Without them, every colour
        drawn in the plot is tried, and those that hold markers are kept.
    colour_tolerance : float
        How far (in RGB) a pixel may be from a colour in *colours* and still belong to it.
    marker_size : float, optional
        The markers' diameter in pixels. Found from the markers themselves if not given.
    error_bars : {"auto", "vertical", "horizontal", "both", "none"}
        Which error bars to look for.
    **figure
        The arguments of :func:`trace_digitiser.digitise` that concern the figure and its axes: *plot_area*,
        *y_min*, *y_max*, *x_min*, *x_max*, *y_ref*, *x_ref*, *y_scale*, *x_scale*, *exclude*, *ocr*,
        *straighten*, *layout_mode*, *expected_panels* ...

    Returns
    -------
    FigureResult
        ``data`` has a row for each point: ``panel_id``, ``series``, ``colour``, ``point``, ``x_pixel``,
        ``y_pixel``, ``x_value``, ``y_value``, ``x_norm_0_to_1``, the marker's ``area_px`` and ``radius_px``,
        ``shared`` (how many markers were in the blob it came from: more than 1 where markers overlap) and, where
        error bars were found, ``x_error_low``, ``x_error_high``, ``y_error_low``, ``y_error_high`` (values) with
        the same in pixels.
    """
    if error_bars not in ("auto", "vertical", "horizontal", "both", "none"):
        raise ValueError("error_bars must be 'auto', 'vertical', 'horizontal', 'both' or 'none'")
    wanted = [parse_colour(c) for c in colours] if colours else None
    fig = prepare_figure(
        image_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        write_outputs=write_outputs,
        extra_parameters={
            "kind": "points",
            "colours": [list(c) for c in wanted] if wanted else None,
            "colour_tolerance": colour_tolerance,
            "marker_size": marker_size,
            "error_bars": error_bars,
        },
        **figure,
    )
    rows = []
    drawn = []
    for p in fig.panels:
        box = interior_box(p, inset=1)
        palette = wanted or _auto_colours(fig.rgb, p, fig.exclude_boxes, fig.dark)
        radius = 0.5 * marker_size if marker_size else None
        markers = find_markers(
            fig.rgb,
            box,
            palette,
            tolerance=colour_tolerance,
            exclude=fig.exclude_boxes,
            radius=radius,
            min_coverage=0.0 if wanted else 0.3,
        )
        if not markers:
            fig.note(
                f"Panel {p['panel_id']}: no markers were found. Give colours (and marker_size) if the points are "
                "an unusual colour or very small, or exclude a legend that hides them."
            )
            continue
        series_used = sorted({m.series for m in markers})
        renumber = {s: i + 1 for i, s in enumerate(series_used)}
        find_error_bars(
            fig.rgb,
            box,
            markers,
            directions=error_bars,
            marker_colours=[palette[s - 1] for s in series_used],
            tolerance=colour_tolerance,
        )
        if any(m.shared > 1 for m in markers):
            n = sum(1 for m in markers if m.shared > 1)
            fig.note(
                f"Panel {p['panel_id']}: {n} points were in blobs of overlapping markers and were separated by "
                "their size; check them on the overlay."
            )
        count = {}
        for m in markers:
            s = renumber[m.series]
            count[s] = count.get(s, 0) + 1
            row = {
                "panel_id": p["panel_id"],
                "series": s,
                "colour": hex_colour(palette[m.series - 1]),
                "point": count[s],
                "x_pixel": round(m.x, 2),
                "y_pixel": round(m.y, 2),
                "x_value": float(x_values(p, m.x)),
                "y_value": float(y_values(p, m.y)),
                "x_norm_0_to_1": float(x_norm(p, m.x)),
                "area_px": round(m.area, 1),
                "radius_px": round(m.radius, 2),
                "shared": m.shared,
                "clipped": m.clipped,
            }
            e = m.errors
            if "left" in e or "right" in e:
                ends = [e.get("left", m.x), e.get("right", m.x)]
                vals = sorted(float(x_values(p, v)) for v in ends)
                row.update(
                    x_error_low=vals[0],
                    x_error_high=vals[1],
                    x_error_low_pixel=min(ends),
                    x_error_high_pixel=max(ends),
                    x_error_capped=e.get("capped_h"),
                )
            if "up" in e or "down" in e:
                ends = [e.get("up", m.y), e.get("down", m.y)]
                vals = sorted(float(y_values(p, v)) for v in ends)
                row.update(
                    y_error_low=vals[0],
                    y_error_high=vals[1],
                    y_error_top_pixel=min(ends),
                    y_error_bottom_pixel=max(ends),
                    y_error_capped=e.get("capped_v"),
                )
            rows.append(row)
            drawn.append(m)
    data = pd.DataFrame(rows)
    if not data.empty:
        data = data.sort_values(["panel_id", "series", "x_pixel"]).reset_index(drop=True)
        data["point"] = data.groupby(["panel_id", "series"]).cumcount() + 1

    def draw(canvas: np.ndarray) -> np.ndarray:
        for m in drawn:
            c = (255, 0, 255)
            centre = (int(round(m.x)), int(round(m.y)))
            cv2.circle(canvas, centre, max(2, int(round(m.radius)) + 2), c, 1, lineType=cv2.LINE_AA)
            e = m.errors
            if "left" in e or "right" in e:
                cv2.line(
                    canvas,
                    (int(round(e.get("left", m.x))), centre[1]),
                    (int(round(e.get("right", m.x))), centre[1]),
                    (0, 160, 0),
                    1,
                )
            if "up" in e or "down" in e:
                cv2.line(
                    canvas,
                    (centre[0], int(round(e.get("up", m.y)))),
                    (centre[0], int(round(e.get("down", m.y)))),
                    (0, 160, 0),
                    1,
                )
        return canvas

    return finish(
        fig, "points", data, draw, output_dir=output_dir, write_outputs=write_outputs, save_overlay=save_overlay
    )
