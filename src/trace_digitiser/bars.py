"""Reading bar charts: the value at the end of each bar, and its error bar."""

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
    colour_distance,
    exclusion_mask,
    finish,
    hex_colour,
    interior_box,
    x_norm,
    x_values,
    y_calibrated,
    y_pixels,
    y_values,
)
from .trace_detection import detect_trace_masks, parse_colour


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _bar_components(member: np.ndarray, min_area: int) -> list[list[int]]:
    """Rectangles of the bar colour: [x0, y0, x1, y1] (in the region), with pieces cut apart by an error bar drawn
    down the middle joined again."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(member.astype(np.uint8), connectivity=4)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w < 2 or h < 2:
            continue
        if area < 0.75 * w * h:
            continue  # not a rectangle: a line, text, a patch of something else
        boxes.append([x, y, x + w - 1, y + h - 1])
    boxes.sort()
    merged: list[list[int]] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            if b[0] - m[2] <= 4 and abs(b[1] - m[1]) <= 2 and abs(b[3] - m[3]) <= 2:
                m[2] = max(m[2], b[2])
                m[1], m[3] = min(m[1], b[1]), max(m[3], b[3])
                continue
        merged.append(list(b))
    return merged


def _far_edge(m: np.ndarray, start: int, step: int) -> float:
    """The edge of a bar in one column of its membership *m*, walking from row *start* (inside the bar) in direction
    *step*: the last pixel at least half covered, plus whatever part of the next is covered."""
    n = len(m)
    i = start
    while 0 <= i + step < n and m[i + step] >= 0.5:
        i += step
    nxt = m[i + step] if 0 <= i + step < n and m[i + step] < 0.5 else 0.0
    return (i - 0.5 + m[i] + nxt) if step > 0 else (i + 0.5 - m[i] - nxt)


def _membership(roi: np.ndarray, colour, paper) -> np.ndarray:
    """How much of each pixel is the bar colour (1) rather than the paper (0): antialiased edges come out between."""
    d_bar = colour_distance(roi, colour)
    gap = max(1.0, float(np.sqrt(((np.asarray(colour, float) - np.asarray(paper, float)) ** 2).sum())))
    return np.clip(1.0 - d_bar / gap, 0.0, 1.0)


def digitise_bars(
    image_path: Union[str, Path, np.ndarray],
    *,
    colours: Optional[Sequence] = None,
    colour_tolerance: float = 40.0,
    baseline: Optional[float] = None,
    orientation: str = "vertical",
    error_bars: bool = True,
    output_dir=None,
    output_prefix: Optional[str] = None,
    write_outputs: bool = True,
    save_overlay: bool = False,
    **figure,
) -> FigureResult:
    """Read the bars of a bar chart: where each one ends, in the axis's units, and its error bar.

    Parameters
    ----------
    image_path : str, Path or ndarray
        The figure.
    colours : list of colours, optional
        The colours of the bars, one series each. Without them, the colours drawn in the plot whose patches are
        rectangles are used.
    colour_tolerance : float
        How far (in RGB) a pixel may be from a bar colour and still belong to the bar.
    baseline : float, optional
        The value the bars start from. By default 0 if it is on the axis, else the edge the bars share.
    orientation : {"vertical", "horizontal"}
        Bars that rise (or fall) from the x axis, or that run across from the y axis.
    error_bars : bool
        Look for an error bar (a thin dark line) through the end of each bar.
    **figure
        The arguments of :func:`trace_digitiser.digitise` that concern the figure and its axes.

    Returns
    -------
    FigureResult
        ``data`` has a row for each bar: ``panel_id``, ``series``, ``colour``, ``bar`` (counting from the left, or
        from the top), its centre and edges in pixels, its position on the category axis (``x_value`` if that axis
        is calibrated, else ``x_norm_0_to_1``), ``value`` (where it ends), ``baseline`` and, where an error bar was
        found, ``error_low`` and ``error_high``.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    wanted = [parse_colour(c) for c in colours] if colours else None
    fig = prepare_figure(
        image_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        write_outputs=write_outputs,
        extra_parameters={
            "kind": "bars",
            "colours": [list(c) for c in wanted] if wanted else None,
            "colour_tolerance": colour_tolerance,
            "baseline": baseline,
            "orientation": orientation,
            "error_bars": error_bars,
        },
        **figure,
    )
    rows, drawn = [], []
    for p in fig.panels:
        x0, y0, x1, y1 = interior_box(p, inset=1)
        rgb = fig.rgb if orientation == "vertical" else np.ascontiguousarray(np.transpose(fig.rgb, (1, 0, 2)))
        if orientation == "horizontal":
            x0, y0, x1, y1 = y0, x0, y1, x1
        roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
        paper = background_colour(fig.rgb, interior_box(p, inset=1))
        away = exclusion_mask(
            roi.shape[:2],
            [(b[1], b[0], b[3], b[2]) for b in fig.exclude_boxes or []]
            if orientation == "horizontal"
            else fig.exclude_boxes,
            origin=(x0, y0),
        )
        if wanted:
            palette = wanted
        else:
            found = detect_trace_masks(fig.rgb, p, exclude=fig.exclude_boxes, achromatic="never", darkness=fig.dark)
            palette = [tuple(int(v) for v in t["median_rgb"]) for t in found]
        # the baseline, in pixels along the value axis (rows of the turned image for horizontal bars)
        e = p["edges"]
        base_px = None
        if orientation == "vertical":
            if baseline is not None or y_calibrated(p):
                b = float(y_pixels(p, 0.0 if baseline is None else baseline))
                if e["top"] - 3 <= b <= e["bottom"] + 3 or baseline is not None:
                    base_px = b
        elif p.get("x_calibration"):
            from ._results import x_pixels

            b = float(x_pixels(p, 0.0 if baseline is None else baseline))
            if e["left"] - 3 <= b <= e["right"] + 3 or baseline is not None:
                base_px = b
        series = 0
        for colour in palette:
            member = _membership(roi, colour, paper)
            member[away] = 0.0
            solid = colour_distance(roi, colour) <= colour_tolerance
            solid &= ~away
            boxes = _bar_components(solid, min_area=max(12, int(0.0004 * roi.size / 3)))
            if not boxes:
                continue
            # the edge the bars share: the baseline
            if base_px is None:
                tops = np.array([b[1] for b in boxes]) + y0
                bottoms = np.array([b[3] for b in boxes]) + y0
                vals, counts = np.unique(np.concatenate([np.round(tops), np.round(bottoms)]), return_counts=True)
                base_here = float(vals[np.argmax(counts)])
                if counts.max() < 2 and len(boxes) > 1:
                    continue
            else:
                base_here = base_px
            bars = [b for b in boxes if min(abs(b[1] + y0 - base_here), abs(b[3] + y0 - base_here)) <= 3.0]
            if not bars:
                continue
            series += 1
            if len(bars) < len(boxes):
                fig.note(
                    f"Panel {p['panel_id']}: {len(boxes) - len(bars)} patch(es) of the bar colour do not reach the "
                    "baseline and were left out (a legend, perhaps)."
                )
            for k, (bx0, by0, bx1, by1) in enumerate(sorted(bars), start=1):
                cols = np.arange(bx0, bx1 + 1)
                width = len(cols)
                inner = cols[int(0.2 * width) : max(int(0.2 * width) + 1, int(np.ceil(0.8 * width)))]
                downward = abs(by0 + y0 - base_here) <= abs(by1 + y0 - base_here)  # the bar hangs below its baseline
                # where the bar ends, column by column, counting partly covered pixels at its edge; the median
                # over the middle of the bar ignores an error bar drawn down it
                mid = int(round(0.5 * (by0 + by1)))
                ends = [_far_edge(member[:, c], mid, +1 if downward else -1) for c in inner]
                end_px = float(np.median(ends)) + y0
                centre = 0.5 * (bx0 + bx1) + x0
                row = {"panel_id": p["panel_id"], "series": series, "colour": hex_colour(colour), "bar": k}
                err = None
                if error_bars:
                    err = _error_bar(fig.rgb if orientation == "vertical" else rgb, member, x0, y0, bx0, bx1, end_px)
                if orientation == "vertical":
                    row.update(
                        x_pixel=round(centre, 2),
                        left_pixel=bx0 + x0,
                        right_pixel=bx1 + x0,
                        x_value=float(x_values(p, centre)),
                        x_norm_0_to_1=float(x_norm(p, centre)),
                        end_pixel=round(end_px, 2),
                        baseline_pixel=round(base_here, 2),
                        value=float(y_values(p, end_px)),
                        baseline=float(y_values(p, base_here)),
                    )
                    if err is not None:
                        lo_v, hi_v = sorted(float(y_values(p, v)) for v in err)
                        row.update(
                            error_low=lo_v, error_high=hi_v, error_top_pixel=min(err), error_bottom_pixel=max(err)
                        )
                    drawn.append(("v", bx0 + x0, bx1 + x0, base_here, end_px, err))
                else:
                    row.update(
                        y_pixel=round(centre, 2),
                        top_pixel=bx0 + x0,
                        bottom_pixel=bx1 + x0,
                        end_pixel=round(end_px, 2),
                        baseline_pixel=round(base_here, 2),
                        value=float(x_values(p, end_px)),
                        baseline=float(x_values(p, base_here)),
                        y_value=float(y_values(p, centre)),
                    )
                    if err is not None:
                        lo_v, hi_v = sorted(float(x_values(p, v)) for v in err)
                        row.update(
                            error_low=lo_v, error_high=hi_v, error_left_pixel=min(err), error_right_pixel=max(err)
                        )
                    drawn.append(("h", bx0 + x0, bx1 + x0, base_here, end_px, err))
                rows.append(row)
        if series == 0:
            fig.note(
                f"Panel {p['panel_id']}: no bars were found. Give colours if the bars are an unusual colour, or "
                "plot_area if the plot area was not found."
            )
        elif orientation == "vertical" and not p.get("x_calibration"):
            pass  # bars on a category axis: their order (bar) and position (x_norm_0_to_1) say which is which
    data = pd.DataFrame(rows)

    def draw(canvas: np.ndarray) -> np.ndarray:
        for kind, a, b, base, end, err in drawn:
            if kind == "v":
                cv2.rectangle(
                    canvas, (int(a), int(round(min(base, end)))), (int(b), int(round(max(base, end)))), (255, 0, 255), 1
                )
                cv2.line(canvas, (int(a) - 2, int(round(end))), (int(b) + 2, int(round(end))), (255, 0, 255), 2)
                if err is not None:
                    c = int(round(0.5 * (a + b)))
                    cv2.line(canvas, (c + 2, int(round(err[0]))), (c + 2, int(round(err[1]))), (0, 170, 0), 1)
            else:
                cv2.rectangle(
                    canvas, (int(round(min(base, end))), int(a)), (int(round(max(base, end))), int(b)), (255, 0, 255), 1
                )
                if err is not None:
                    c = int(round(0.5 * (a + b)))
                    cv2.line(canvas, (int(round(err[0])), c + 2), (int(round(err[1])), c + 2), (0, 170, 0), 1)
        return canvas

    return finish(
        fig, "bars", data, draw, output_dir=output_dir, write_outputs=write_outputs, save_overlay=save_overlay
    )


def _error_bar(rgb, member, x0, y0, bx0, bx1, end_px) -> Optional[tuple[float, float]]:
    """The ends (rows, in the image) of a thin dark line drawn through the end of a bar, near its middle."""
    h = member.shape[0]
    centre = 0.5 * (bx0 + bx1)
    best = None
    roi = rgb[y0 : y0 + h, :]
    gray = cv2.cvtColor(np.ascontiguousarray(roi), cv2.COLOR_RGB2GRAY).astype(np.float32)
    for dc in range(-3, 4):
        c = int(round(centre + dc))
        if not 0 <= c < member.shape[1]:
            continue
        col = gray[:, c + x0]
        # dark, and not the bar colour: the ink of an error bar
        dark = (col < 110) & (member[:, c] < 0.5)
        r = int(round(end_px)) - y0
        found = [i for i in range(max(0, r - 3), min(h, r + 4)) if dark[i]]
        if not found:
            continue
        i0 = i1 = found[0]
        while i0 - 1 >= 0 and dark[i0 - 1]:
            i0 -= 1
        while i1 + 1 < h and dark[i1 + 1]:
            i1 += 1
        if i1 - i0 + 1 < 4:
            continue
        span = (i0 + y0, i1 + y0)
        if best is None or (span[1] - span[0]) > (best[1] - best[0]):
            best = span
    if best is None:
        return None
    # the ends of the line are its caps (or its last pixels): take the middle of the end pixel
    return (float(best[0]) + 0.5 - 0.5, float(best[1]) - 0.5 + 0.5)
