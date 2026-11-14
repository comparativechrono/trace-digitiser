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


def _fill_enclosed(mask: np.ndarray, closed_side: str = "bottom") -> np.ndarray:
    """*mask* with the pockets of paper it encloses filled in: the inside of a hatched or outlined bar becomes part of
    the bar. The side *closed_side* ("bottom" or "top") of the region counts as closed - the axis the bars stand on is
    outside the region, and closes them there. A pocket as big as a quarter of the region is not filled (it is not the
    inside of a bar)."""
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    pad[-1 if closed_side == "bottom" else 0, :] = 1
    flood = (1 - pad).astype(np.uint8)
    seed = (0, 0) if closed_side == "bottom" else (0, h + 1)
    ffmask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ffmask, seed, 2, flags=4)
    pockets = (flood[1:-1, 1:-1] == 1) & ~mask
    if not pockets.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pockets.astype(np.uint8), connectivity=4)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= 0.25 * h * w
    return mask | keep[labels]


def _achromatic_colours(roi: np.ndarray, paper, away: np.ndarray) -> list[tuple[int, int, int]]:
    """Black and the greys that cover a fair part of the region: the colours of black and grey bars, which the search
    for the colours of lines leaves out (it takes black and grey for axes and text)."""
    rgb = roi.astype(np.int16)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    gray = rgb.mean(axis=2)
    paper_gray = float(np.mean(paper))
    ink = (sat < 25) & (gray < paper_gray - 45) & ~away
    if ink.sum() < 0.005 * ink.size:
        return []
    hist = np.bincount(np.clip(gray[ink], 0, 255).astype(int) // 16, minlength=16).astype(float)
    out = []
    for b in np.argsort(-hist):
        if hist[b] < 0.004 * ink.size or len(out) >= 3:
            break
        level = int(np.median(gray[ink & (np.clip(gray, 0, 255).astype(int) // 16 == b)]))
        if all(abs(level - o[0]) > 40 for o in out):
            out.append((level, level, level))
    return out


def _whisker(ink: np.ndarray, centre: float, end: float, width: float, step: int) -> Optional[float]:
    """The far end of an error bar drawn from the end of a bar, beyond it (*step* -1: upwards in the region, +1:
    downwards), whatever its colour - the bar's own, as often as not, so that the half inside the bar is hidden: a thin
    line of ink down the middle of the bar, running on from its end, usually ending in a cap (a short line across it,
    narrower than the bar). Returns the middle of the cap (or the last pixel of an uncapped line), in the region's
    rows, or None."""
    h, w = ink.shape
    thin = max(3.0, 0.35 * width)

    def across(rr: int, c: int) -> int:
        a = b = c
        while a - 1 >= 0 and ink[rr, a - 1]:
            a -= 1
        while b + 1 < w and ink[rr, b + 1]:
            b += 1
        return b - a + 1

    best = None
    for dc in (0, -1, 1, -2, 2):
        c = int(round(centre + dc))
        if not 0 <= c < w:
            continue
        r = int(round(end)) + (step if step < 0 else 0)  # the first row beyond the end of the bar
        start = next((r + step * k for k in range(0, 3) if 0 <= r + step * k < h and ink[r + step * k, c]), None)
        if start is None:
            continue
        i = start
        while 0 <= i + step < h and ink[i + step, c]:
            i += step
        rows = list(range(start, i + step, step))  # from the bar outwards
        widths = [across(rr, c) for rr in rows]
        k = 0
        while k < len(rows) and widths[k] <= thin:
            k += 1
        line_w = float(np.median(widths[:k])) if k else None
        cap = []
        while k < len(rows) and (line_w is None or widths[k] >= 2.0 * line_w) and widths[k] <= 0.9 * width + 2:
            cap.append(rows[k])
            k += 1
        if cap and line_w is None and len(cap) > max(6, 0.05 * h):
            continue  # a block, not a cap sitting on the bar
        if cap:
            far = 0.5 * (cap[0] + cap[-1])
        elif line_w is not None and k == len(rows) and k >= 3:
            far = float(rows[k - 1])  # a line with no cap
        else:
            continue
        if best is None or abs(far - end) > abs(best - end):
            best = far
    return best


def _heights_from(mask: np.ndarray, base_row: float, step: int, slack: int) -> np.ndarray:
    """For each column of *mask*, how far it is filled from the baseline (row *base_row* of the region, which may lie
    just outside it) in direction *step* (-1: upwards, +1: downwards): from a filled pixel within *slack* rows of the
    baseline, as far as the column runs on unbroken, counted in rows from the baseline to its last pixel. 0 where the
    column is not filled near the baseline."""
    h, w = mask.shape
    b = int(round(base_row))
    if step < 0:
        first = min(h - 1, b + slack)
        if first < 0:
            return np.zeros(w)
        sub = mask[: first + 1][::-1]  # index 0: row `first`; going up
    else:
        first = max(0, b - slack)
        if first > h - 1:
            return np.zeros(w)
        sub = mask[first:]
    n = sub.shape[0]
    idx = np.arange(n)[:, None]
    window = min(n, 2 * slack + 1)
    has = sub[:window].any(axis=0)
    start = np.argmax(sub[:window], axis=0)
    after = (idx > start[None, :]) & ~sub
    stop = np.where(after.any(axis=0), np.argmax(after, axis=0), n)  # the first gap after the start
    last = stop - 1  # the index of the run's last pixel
    rows = first - last if step < 0 else first + last
    height = (b - rows) if step < 0 else (rows - b)
    return np.where(has, np.maximum(height, 0), 0).astype(float)


def _border(ink: np.ndarray, member: np.ndarray, cols, end: float, step: int, thick: int) -> int:
    """How many rows of a border of another colour (a black outline round a grey bar) lie beyond the end *end* of a
    bar's fill (rows of the region, *step* outwards): rows where the ink, not of the fill's colour, runs right across
    the middle of the bar; at most a few line widths. 0 if there is none."""
    h = ink.shape[0]
    cols = np.asarray(cols)
    r = int(np.floor(end)) if step > 0 else int(np.ceil(end)) - 1
    if step > 0:
        r = int(np.ceil(end))
    n = 0
    limit = max(4, 3 * thick + 2, int(0.01 * h) + 2)
    while 0 <= r < h and n <= limit:
        across = ink[r, cols] & (member[r, cols] < 0.5)
        if across.mean() < 0.8:
            break
        n += 1
        r += step
    return n if 0 < n <= limit else 0


def _far_edge(m: np.ndarray, start: int, step: int) -> float:
    """The edge of a bar in one column of its membership *m*, walking from row *start* (inside the bar) in direction
    *step*: the last pixel at least half covered, plus whatever part of the next is covered."""
    n = len(m)
    i = start
    while 0 <= i + step < n and m[i + step] >= 0.5:
        i += step
    nxt = m[i + step] if 0 <= i + step < n and m[i + step] < 0.5 else 0.0
    return (i - 0.5 + m[i] + nxt) if step > 0 else (i + 0.5 - m[i] - nxt)


def _paper(rgb: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """The colour of the paper in a plot area: the commonest colour of the whole area, or of a strip along its top or
    its bottom, whichever is the lightest - black bars can cover more of a plot than the paper does."""
    x0, y0, x1, y1 = box
    strip = max(1, int(0.15 * (y1 - y0 + 1)))
    candidates = [
        background_colour(rgb, box),
        background_colour(rgb, (x0, y0, x1, min(y1, y0 + strip))),
        background_colour(rgb, (x0, max(y0, y1 - strip), x1, y1)),
    ]
    return max(candidates, key=lambda c: sum(c))


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
        paper = _paper(fig.rgb, interior_box(p, inset=1))
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
            palette += [c for c in _achromatic_colours(roi, paper, away) if c not in palette]
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
        one_sided = 0
        ink = colour_distance(roi, paper) > 60
        ink &= ~away
        axis_edge = e["bottom"] if orientation == "vertical" else e["left"]  # the axis bars stand on (image px)
        # an axis line the region takes in some of (where it is thicker than it was measured): rows of ink right
        # across, at the top or bottom of the region; they are not bars, but they close the bars standing on them
        barrier = np.zeros(roi.shape[:2], bool)
        ink_rows = ink.mean(axis=1)
        r = roi.shape[0] - 1
        while r >= 0 and ink_rows[r] >= 0.85:
            barrier[r] = True
            r -= 1
        r = 0
        while r < roi.shape[0] and ink_rows[r] >= 0.85:
            barrier[r] = True
            r += 1
        thick = int(p.get("line_thickness", 1))
        slack = max(thick // 2 + 3, int(barrier.any(axis=1).sum()) + 3)
        # a bar is wider than the lines of the figure (the sides of an outlined bar are not bars)
        min_w = max(3, 2 * thick + 1, int(round(0.01 * roi.shape[1])))
        k3 = np.ones((3, 3), np.uint8)
        candidates = []
        left_out = 0
        for colour in palette:
            member = _membership(roi, colour, paper)
            member[away] = 0.0
            solid = colour_distance(roi, colour) <= colour_tolerance
            solid &= ~away
            # an axis line the region takes in some of (where it is thicker than it was measured): rows right across,
            # at the top or bottom of the region, are not bars
            solid[barrier] = False
            if solid.sum() < 12:
                continue
            # hatched and outlined bars: the paper inside them is part of them (a small gap in an outline closed;
            # the axis line closes them at the bottom, where the bar's own outline is hidden under it)
            closed = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, k3).astype(bool)
            filled = _fill_enclosed(closed | barrier, "bottom" if orientation == "vertical" else "top") & ~barrier
            boxes = _bar_components(filled, min_area=max(12, int(0.0004 * roi.size / 3)))
            boxes = [b for b in boxes if (b[2] - b[0] + 1) < 0.9 * roi.shape[1]]  # a band right across is not a bar
            # the baseline: 0 if it is on the axis (or the value given); else the edge the bars share - of the edges
            # shared by most bars, the nearest the axis they stand on (a zero line in the middle is shared by more);
            # else the axis itself
            if base_px is not None:
                base_here = base_px
            elif boxes:
                tops = np.array([b[1] for b in boxes]) + y0
                bottoms = np.array([b[3] for b in boxes]) + y0
                vals, counts = np.unique(np.concatenate([np.round(tops), np.round(bottoms)]), return_counts=True)
                if counts.max() >= 2:
                    shared = vals[counts == counts.max()]
                    base_here = float(min(shared, key=lambda v: abs(v - axis_edge)))
                else:
                    near = [v for v in vals if abs(v - axis_edge) <= slack + 2]
                    base_here = float(min(near, key=lambda v: abs(v - axis_edge))) if near else float(axis_edge)
            else:
                base_here = float(axis_edge)
            left_out += sum(1 for b in boxes if min(abs(b[1] + y0 - base_here), abs(b[3] + y0 - base_here)) > slack)
            # the bars: columns filled from the baseline, side by side; the median height over the middle of a bar
            # ignores an error bar (or a bracket) joined to it, and its top must be flat
            for step in (-1, +1):
                heights = _heights_from(filled, base_here - y0, step, slack)
                for c0, c1 in _runs(heights >= 3):
                    if c1 - c0 + 1 < min_w or c1 - c0 + 1 >= 0.9 * roi.shape[1]:
                        continue
                    width = c1 - c0 + 1
                    lo_c = c0 + int(0.2 * width)
                    inner = np.arange(lo_c, max(lo_c + 1, c0 + int(np.ceil(0.8 * width))))
                    hs = heights[inner]
                    med = float(np.median(hs))
                    q1, q3 = np.percentile(hs, [25, 75])
                    if med < 3 or q3 - q1 > 0.1 * med + 3:
                        continue  # not a bar: its top is not flat (a scatter of lines, text)
                    b_row = base_here - y0
                    far_row = b_row - med if step < 0 else b_row + med
                    r0, r1 = int(round(min(b_row, far_row))), int(round(max(b_row, far_row)))
                    r0, r1 = max(0, r0), min(roi.shape[0] - 1, r1)
                    cover = float(solid[r0 : r1 + 1, c0 : c1 + 1].mean()) if r1 >= r0 else 0.0
                    candidates.append({
                        "colour": colour, "c0": int(c0), "c1": int(c1), "base": base_here, "downward": step > 0,
                        "inner": inner, "cover": cover,
                        "mid": int(min(max(round(0.5 * (b_row + far_row)), 0), roi.shape[0] - 1)),
                        "member": member, "filled": filled,
                    })
        # the same bar found in two colours (a grey bar with a black outline): keep the colour that fills it
        candidates.sort(key=lambda c: -c["cover"])
        kept = []
        for c in candidates:
            clash = False
            for k in kept:
                overlap = min(c["c1"], k["c1"]) - max(c["c0"], k["c0"]) + 1
                if overlap > 0.5 * min(c["c1"] - c["c0"] + 1, k["c1"] - k["c0"] + 1) and c["downward"] == k["downward"]:
                    clash = True
                    break
            if not clash:
                kept.append(c)
        colours_used = []
        for c in sorted(kept, key=lambda c: c["c0"]):
            key = tuple(int(v) for v in c["colour"])
            if key not in colours_used:
                colours_used.append(key)
        rank = {k: (palette.index(k) if k in palette else 99) for k in colours_used}
        order = {k: i + 1 for i, k in enumerate(sorted(colours_used, key=rank.get))}
        count = {}
        if left_out and kept:
            fig.note(
                f"Panel {p['panel_id']}: {left_out} patch(es) of a bar colour do not reach the baseline and were left "
                "out (a legend, perhaps)."
            )
        series = len(order)
        for c in sorted(kept, key=lambda c: (order[tuple(int(v) for v in c["colour"])], c["c0"])):
            colour, bx0, bx1, base_here, downward = c["colour"], c["c0"], c["c1"], c["base"], c["downward"]
            s_id = order[tuple(int(v) for v in colour)]
            count[s_id] = count.get(s_id, 0) + 1
            k = count[s_id]
            member = c["member"]
            # where the bar ends, column by column, counting partly covered pixels at its edge
            m_end = member if c["cover"] >= 0.6 else c["filled"].astype(float)
            ends = [_far_edge(m_end[:, col], c["mid"], +1 if downward else -1) for col in c["inner"]]
            end_px = float(np.median(ends)) + y0
            if c["cover"] >= 0.6:
                end_px += _border(ink, member, c["inner"], end_px - y0, +1 if downward else -1, thick) * (
                    1 if downward else -1)
            centre = 0.5 * (bx0 + bx1) + x0
            row = {"panel_id": p["panel_id"], "series": s_id, "colour": hex_colour(colour), "bar": k}
            err = None
            if error_bars:
                # an error bar of another colour, crossing the end of the bar (it cannot be told apart from a dark bar)
                dark_bar = float(np.mean(colour)) < 110
                err = None if dark_bar else _error_bar(
                    fig.rgb if orientation == "vertical" else rgb, member, x0, y0, bx0, bx1, end_px)
                if err is None:
                    # drawn in the bar's own colour (or hidden inside the bar): only the half beyond its end shows
                    far = _whisker(ink, 0.5 * (bx0 + bx1), end_px - y0, bx1 - bx0 + 1, +1 if downward else -1)
                    if far is not None:
                        err = (end_px, far + y0)
                        one_sided += 1
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
                    row.update(error_low=lo_v, error_high=hi_v, error_top_pixel=min(err), error_bottom_pixel=max(err))
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
                    row.update(error_low=lo_v, error_high=hi_v, error_left_pixel=min(err), error_right_pixel=max(err))
                drawn.append(("h", bx0 + x0, bx1 + x0, base_here, end_px, err))
            rows.append(row)
        if one_sided:
            fig.note(
                f"Panel {p['panel_id']}: {one_sided} error bar(s) show only beyond the end of the bar (the other half "
                "is hidden by the bar, or not drawn); error_high is where they end, and error_low is the bar's end."
            )
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
