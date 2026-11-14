"""Reading figures drawn as a stack of rows: actograms, double-plotted records, "birthograms" and the like.

Each row has a baseline; what is drawn in it rises from the baseline (or hangs from it), as filled bars, steps or
a filled curve. The reader measures, column by column, how far the fill reaches from each row's baseline, and
averages that over bins (the hours of a day, the months of a year) if asked.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd

from ._figure import prepare_figure
from ._results import (
    FigureResult,
    background_colour,
    exclusion_mask,
    finish,
    hex_colour,
    interior_box,
    x_norm,
    x_values,
    y_pixels,
)
from .markers import find_markers
from .trace_detection import detect_trace_masks, parse_colour


def coverage(roi: np.ndarray, colour, paper) -> tuple[np.ndarray, np.ndarray]:
    """How much of each pixel the fill covers (0 to 1), and where other ink (a dot, a line of another colour) hides
    what is under it.

    A pixel between the paper and the fill colour is that far covered; one darker than the fill along the same
    line (where its outline overlaps it, or two fills overlap) is fully covered; one of another colour altogether
    is neither, but other ink.
    """
    paper = np.asarray(paper, dtype=np.float32)
    toward = paper - np.asarray(colour, dtype=np.float32)
    norm2 = max(1.0, float((toward**2).sum()))
    diff = paper[None, None, :] - roi.astype(np.float32)
    t = (diff * toward).sum(axis=2) / norm2
    off_line = np.sqrt(np.maximum(0.0, (diff**2).sum(axis=2) - t * t * norm2))
    other = off_line > 0.5 * np.sqrt(norm2) + 12.0
    return np.where(other, 0.0, np.clip(t, 0.0, 1.0)), other


def _extents(member: np.ndarray, start: float, step: int, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """For every column of *member* (how much of each pixel is the fill), where the fill of the row whose baseline is
    at row *start* begins and ends, in image rows of *member* (sub-pixel), going in direction *step* (+1 down, -1
    up) no further than *limit* pixels. The fill must begin at the baseline's pixel or one of the two after it
    (the baseline the axis gives may be a pixel off), and may reach back one pixel before it - but no further, which
    would be the fill of the row before. Columns with no fill there give NaN."""
    n, w = member.shape
    s0 = 2
    first = int(np.floor(start + 0.5)) - s0 * step
    idx = first + step * np.arange(limit + 6)
    ok = (idx >= 0) & (idx < n)
    band = np.zeros((limit + 6, w))
    band[ok] = member[idx[ok]]
    on = band >= 0.5
    k = band.shape[0] - 1
    cols = np.arange(w)
    begin = np.where(on[s0], s0, np.where(on[s0 + 1], s0 + 1, np.where(on[s0 + 2], s0 + 2, -1)))
    has = begin >= 0
    begin = np.where(has, begin, s0)
    back = has & (begin == s0) & on[s0 - 1]
    begin = np.where(back, s0 - 1, begin)
    after = np.arange(band.shape[0])[:, None] >= begin[None, :]
    off = (~on) & after
    stops = np.where(off.any(axis=0), np.argmax(off, axis=0), band.shape[0])
    last = np.clip(stops - 1, 0, k)

    def partial(i):
        v = np.where((i >= 0) & (i <= k), band[np.clip(i, 0, k), cols], 0.0)
        return np.where(v < 0.5, v, 0.0)

    # in rows counted outward from `first`: the fill covers [near, far]
    near = begin + 0.5 - band[begin, cols] - np.where(begin - 1 >= s0 - 1, partial(begin - 1), 0.0)
    far = last - 0.5 + band[last, cols] + partial(stops)
    to_image = (lambda t: first + t) if step > 0 else (lambda t: first - t)
    return np.where(has, to_image(near), np.nan), np.where(has, to_image(far), np.nan)


def digitise_rows(
    image_path: Union[str, Path, np.ndarray],
    *,
    row_values: Optional[Sequence[float]] = None,
    row_positions: Optional[Sequence[float]] = None,
    rows: Optional[int] = None,
    row_range: Optional[tuple[float, float]] = None,
    row_spacing_value: Optional[float] = None,
    direction: str = "auto",
    bins: Optional[int] = None,
    colour=None,
    colour_tolerance: float = 60.0,
    dots=None,
    output_dir=None,
    output_prefix: Optional[str] = None,
    write_outputs: bool = True,
    save_overlay: bool = False,
    **figure,
) -> FigureResult:
    """Read a figure drawn as a stack of rows (an actogram, a double-plotted record, a birthogram).

    Where each row's baseline is, in order of preference:

    * *row_positions*: the pixel row of each baseline;
    * *row_values*: the value of each baseline on the y axis (the years 1955, 1956 ... of a figure whose y axis is
      labelled with them), placed by the y-axis calibration;
    * *rows* evenly spaced baselines from the first to the last of *row_range* (pixel rows);
    * otherwise the tick labels the y-axis calibration was read from, one row each.

    Parameters
    ----------
    row_spacing_value : float, optional
        What the distance between two baselines stands for, in the units of what is drawn: 1.1 if each row's
        values run from 0 to 1 and rows are drawn 1.1 apart. Without it, heights are given as a fraction of the
        spacing.
    direction : {"auto", "up", "down"}
        Whether the fill rises above its baseline (up, as in most actograms) or hangs below it (down, as on an axis
        that runs downwards).
    bins : int, optional
        Average each row over this many equal bins across the plot (24 for a double-plotted year of months, 48 for
        two days of half-hours); otherwise every pixel column is reported.
    colour : colour, optional
        The colour of the fill. The commonest colour drawn in the plot otherwise.
    dots : colour, optional
        Also find markers of this colour (the black dots that mark a peak, say) and give each the row and bin it is
        in.
    **figure
        The arguments of :func:`trace_digitiser.digitise` that concern the figure and its axes.

    Returns
    -------
    FigureResult
        ``data`` has a row for each bin (or column) of each row: ``panel_id``, ``row`` (from the top), its
        ``row_value`` if known, ``bin`` (or ``column``), ``x_pixel``, ``x_value`` (if the x axis is calibrated),
        ``x_norm_0_to_1``, ``height_px``, ``fraction`` (of the row spacing) and ``value`` (if *row_spacing_value* is
        given), and ``dot`` (whether a dot is in that bin) if *dots* was given.
    """
    if direction not in ("auto", "up", "down"):
        raise ValueError("direction must be 'auto', 'up' or 'down'")
    fill = parse_colour(colour) if colour is not None else None
    dot_colour = parse_colour(dots) if dots is not None else None
    fig = prepare_figure(
        image_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        write_outputs=write_outputs,
        extra_parameters={
            "kind": "rows",
            "row_values": list(row_values) if row_values is not None else None,
            "row_positions": list(row_positions) if row_positions is not None else None,
            "rows": rows,
            "row_range": list(row_range) if row_range else None,
            "row_spacing_value": row_spacing_value,
            "direction": direction,
            "bins": bins,
            "colour": list(fill) if fill else None,
            "dots": list(dot_colour) if dot_colour else None,
        },
        **figure,
    )
    out_rows, drawn = [], []
    for p in fig.panels:
        e = p["edges"]
        # ---- the baselines
        labels = None
        if row_positions is not None:
            base = [float(v) for v in row_positions]
        elif row_values is not None:
            base = [float(v) for v in y_pixels(p, list(row_values))]
            labels = [float(v) for v in row_values]
        elif rows is not None and row_range is not None:
            base = list(np.linspace(float(row_range[0]), float(row_range[1]), int(rows)))
        else:
            inl = (p.get("y_calibration") or {}).get("inliers") or []
            if len(inl) < 2:
                fig.note(
                    f"Panel {p['panel_id']}: where its rows are is not known. Give row_values, row_positions, or "
                    "rows and row_range."
                )
                continue
            base = [float(y_pixels(p, v)) for _, v in sorted(inl)]
            labels = [float(v) for _, v in sorted(inl)]
        order = np.argsort(base)
        base = [base[i] for i in order]
        if labels is not None:
            labels = [labels[i] for i in order]
        if len(base) < 2 and not row_spacing_value:
            fig.note(f"Panel {p['panel_id']}: only one row was given, so the row spacing is not known.")
            continue
        spacing = float(np.median(np.diff(base))) if len(base) > 1 else float(e["bottom"] - e["top"])
        # ---- the fill
        x0, y0, x1, y1 = interior_box(p, inset=1)
        roi = fig.rgb[y0 : y1 + 1, x0 : x1 + 1]
        paper = background_colour(fig.rgb, (x0, y0, x1, y1))
        if fill is None:
            found = detect_trace_masks(fig.rgb, p, exclude=fig.exclude_boxes, achromatic="never", darkness=fig.dark)
            if not found:
                fig.note(f"Panel {p['panel_id']}: no fill was found; give colour.")
                continue
            colour_here = tuple(int(v) for v in max(found, key=lambda t: t["pixel_count"])["median_rgb"])
        else:
            colour_here = fill
        member, other = coverage(roi, colour_here, paper)
        member[exclusion_mask(member.shape, fig.exclude_boxes, origin=(x0, y0))] = 0.0
        # ---- up or down
        way = direction
        if way == "auto":
            above = below = 0.0
            for b in base:
                r = int(round(b)) - y0
                above += float(member[max(0, r - 4) : max(0, r - 1), :].sum())
                below += float(member[min(member.shape[0], r + 2) : min(member.shape[0], r + 5), :].sum())
            way = "down" if below > above else "up"
        step = 1 if way == "down" else -1
        limit = int(np.ceil(spacing))
        # ---- heights, column by column
        heights = np.zeros((len(base), member.shape[1]))
        observed = []
        for j, b in enumerate(base):
            near, far = _extents(member, b - y0, step, limit)
            extent = np.abs(far - near)
            # A fill drawn with an outline of its own colour (matplotlib's fill_between, a bar with an edge) has a
            # line along its baseline even where its height is nought: that line's thickness is the thinnest the
            # fill gets, and the fill's height is its extent less that.
            filled = np.isfinite(extent)
            thin = extent[filled & (extent < 3.0)]
            stroke = float(np.median(thin)) if thin.size >= max(3, 0.03 * filled.sum()) else 0.0
            heights[j] = np.where(filled, np.clip(extent - stroke, 0.0, None), 0.0)
            # where other ink (a dot) lies on the baseline, what is under it cannot be seen: leave those columns out
            r = int(round(b)) - y0
            hidden = other[max(0, r - 2) : max(0, r + 3), :].any(axis=0) if 0 <= r < other.shape[0] else None
            if hidden is not None:
                heights[j] = np.where(hidden, np.nan, heights[j])
            # where the baseline really is: the near edge of the fill, half the outline in
            nears = near[filled]
            observed.append(float(np.median(nears)) + y0 + step * 0.5 * stroke if nears.size else b)
        base = observed
        cols = np.arange(roi.shape[1]) + x0
        if bins:
            edges = np.linspace(e["left"], e["right"], int(bins) + 1)
            per = []
            for k in range(int(bins)):
                a, z = edges[k], edges[k + 1]
                w = z - a
                inside = (cols >= a + 0.2 * w) & (cols <= z - 0.2 * w)
                per.append((0.5 * (a + z), inside))
        else:
            per = [(float(c), cols == c) for c in cols]
        dot_at = {}
        if dot_colour is not None:
            markers = find_markers(
                fig.rgb, (x0, y0, x1, y1), [dot_colour], tolerance=colour_tolerance, exclude=fig.exclude_boxes
            )
            for m in markers:
                # the row whose band holds the dot
                j = int(np.argmin([abs((m.y - b) * step - 0.5 * spacing) for b in base]))
                if bins:
                    k = int(np.clip(np.floor((m.x - e["left"]) / (e["right"] - e["left"]) * bins), 0, bins - 1))
                else:
                    k = int(round(m.x)) - x0
                dot_at[(j, k)] = m
        for j, b in enumerate(base):
            line = []
            for k, (xc, inside) in enumerate(per):
                vals = heights[j, inside]
                h = float(np.nanmedian(vals)) if inside.any() and np.isfinite(vals).any() else float("nan")
                row = {"panel_id": p["panel_id"], "row": j + 1}
                if labels is not None:
                    row["row_value"] = labels[j]
                row["bin" if bins else "column"] = k + 1 if bins else int(round(xc))
                row.update(
                    x_pixel=round(float(xc), 2),
                    x_value=float(x_values(p, xc)),
                    x_norm_0_to_1=float(x_norm(p, xc)),
                    baseline_pixel=round(b, 2),
                    height_px=round(h, 3),
                    fraction=h / spacing if spacing else float("nan"),
                )
                if row_spacing_value:
                    row["value"] = row["fraction"] * float(row_spacing_value)
                if dot_colour is not None:
                    row["dot"] = (j, k) in dot_at
                out_rows.append(row)
                line.append((xc, b + step * h))
            drawn.append((b, line))
        fig.parameters.setdefault("found", {})[p["panel_id"]] = {
            "rows": len(base),
            "spacing_px": spacing,
            "direction": way,
            "colour": hex_colour(colour_here),
        }
    data = pd.DataFrame(out_rows)

    def draw(canvas: np.ndarray) -> np.ndarray:
        for b, line in drawn:
            xs = [x for x, _ in line]
            cv2.line(canvas, (int(min(xs)), int(round(b))), (int(max(xs)), int(round(b))), (0, 170, 0), 1)
            pts = np.array([[int(round(x)), int(round(y))] for x, y in line if np.isfinite(y)], np.int32)
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, (255, 0, 255), 1, lineType=cv2.LINE_AA)
        return canvas

    return finish(
        fig, "rows", data, draw, output_dir=output_dir, write_outputs=write_outputs, save_overlay=save_overlay
    )
