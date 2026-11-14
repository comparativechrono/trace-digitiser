"""Panel detection: finding the plot area(s) in a figure.

The main entry point is :func:`find_plot_panels`.  It tries, in order:

1. **User-supplied plot areas** (``plot_area=``) - used exactly as given.
2. **Axis lines** (:mod:`trace_digitiser.frames`) - each plot area is
   located from its axis lines, to sub-pixel precision.  Works for closed
   frames and for open "L" axes, with or without gridlines.
3. **Gridline clustering** (the original route) - for charts that have
   horizontal gridlines but no axis lines to anchor on.

Every panel is a dict.  ``x0``, ``x1``, ``y_top`` and ``y_bottom`` are the
integer pixel bounds of the plot area; ``edges`` holds the same bounds as
sub-pixel floats, which is what axis calibration uses.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .frames import (
    AxisLine,
    Frame,
    _add_gridlines_and_ticks,
    _reading_order,
    _saturation,
    darkness_map,
    find_frames_and_lines,
)
from .geometry import box_area, box_iou, contains_box
from .line_detection import detect_axis_line_segments, detect_long_horizontal_lines

# ===================================================================
# Clustering helpers
# ===================================================================

def cluster_by_x_extent(
    h_lines: list[dict],
    width: int,
    tol_frac: float = 0.04,
) -> list[dict]:
    """Group horizontal lines with similar left/right x extents."""
    tol = tol_frac * width
    groups: list[dict] = []

    for line in h_lines:
        if line["x0"] < 0.015 * width and line["x1"] > 0.985 * width:
            continue

        placed = False
        for g in groups:
            if abs(line["x0"] - g["median_x0"]) < tol and abs(line["x1"] - g["median_x1"]) < tol:
                g["lines"].append(line)
                g["median_x0"] = float(np.median([d["x0"] for d in g["lines"]]))
                g["median_x1"] = float(np.median([d["x1"] for d in g["lines"]]))
                placed = True
                break

        if not placed:
            groups.append(
                {
                    "median_x0": float(line["x0"]),
                    "median_x1": float(line["x1"]),
                    "lines": [line],
                }
            )

    return groups


def split_y_clusters(
    y_values: list[int],
    image_height: int,
    expected_splits: Optional[int] = None,
) -> list[list[int]]:
    """Split horizontal gridline y-positions into vertically separated clusters.

    Parameters
    ----------
    y_values : list of int
        Sorted y-pixel positions of detected horizontal lines.
    image_height : int
        Full image height (used for adaptive threshold).
    expected_splits : int, optional
        If provided, the number of splits to make (= expected_rows - 1).
        The function picks the *expected_splits* largest gaps, provided
        each is at least 10 % wider than the smallest gap.  This allows
        stacked panels with nearly uniform gridline spacing to be
        separated when the user supplies a row count.
    """
    y_values = sorted(int(y) for y in y_values)

    if len(y_values) <= 1:
        return [y_values]

    gaps = np.diff(y_values)

    # ------------------------------------------------------------------
    # Hint-aware splitting: pick the N largest gaps
    # ------------------------------------------------------------------
    if expected_splits is not None and expected_splits > 0 and len(gaps) >= expected_splits:
        n_splits = int(expected_splits)
        sorted_gap_indices = np.argsort(gaps)[::-1]  # largest-first

        min_gap = float(gaps.min())
        # Accept a gap as a split point if it's at least 10 % wider than
        # the smallest gap, OR if it's the largest gap regardless.
        valid_indices: list[int] = []
        for rank, idx in enumerate(sorted_gap_indices):
            if len(valid_indices) >= n_splits:
                break
            if rank == 0 or gaps[idx] > 1.10 * min_gap:
                valid_indices.append(int(idx))

        if valid_indices:
            valid_indices.sort()
            clusters: list[list[int]] = []
            prev = 0
            for i in valid_indices:
                clusters.append(y_values[prev : i + 1])
                prev = i + 1
            clusters.append(y_values[prev:])
            clusters = [c for c in clusters if c]
            # Only accept if we got a reasonable number of clusters
            if len(clusters) >= 2:
                return clusters

    # ------------------------------------------------------------------
    # Adaptive threshold (original logic)
    # ------------------------------------------------------------------
    small_gaps = gaps[gaps < 0.25 * image_height]
    typical_gap = float(np.median(small_gaps)) if len(small_gaps) else float(np.median(gaps))

    split_threshold = max(0.08 * image_height, 2.4 * typical_gap)

    clusters = []
    current = [y_values[0]]

    for y_next, gap in zip(y_values[1:], gaps):
        if gap > split_threshold:
            clusters.append(current)
            current = [y_next]
        else:
            current.append(y_next)

    clusters.append(current)
    return clusters


# ===================================================================
# Candidate generation
# ===================================================================

def candidates_from_horizontal_extents(
    h_lines: list[dict],
    image_width: int,
    image_height: int,
    source_name: str = "horizontal_extents",
    score_bonus: float = 2.0,
    expected_rows: Optional[int] = None,
) -> list[dict]:
    """Build panel candidate rectangles from grouped horizontal lines.

    When *expected_rows* is provided it is forwarded to
    :func:`split_y_clusters` as a splitting hint.
    """
    candidates: list[dict] = []
    x_groups = cluster_by_x_extent(h_lines, image_width)

    expected_splits = (expected_rows - 1) if expected_rows is not None and expected_rows > 1 else None

    for g in x_groups:
        if len(g["lines"]) < 2:
            continue

        y_clusters = split_y_clusters(
            [d["y"] for d in g["lines"]],
            image_height,
            expected_splits=expected_splits,
        )

        for ys in y_clusters:
            if len(ys) < 2:
                continue

            y_top = int(min(ys))
            y_bottom = int(max(ys))
            panel_height = y_bottom - y_top

            if panel_height < 0.07 * image_height:
                continue

            relevant = [d for d in g["lines"] if d["y"] in ys]
            x0 = int(np.median([d["x0"] for d in relevant]))
            x1 = int(np.median([d["x1"] for d in relevant]))

            if x1 - x0 < 0.12 * image_width:
                continue

            candidates.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "y_top": y_top,
                    "y_bottom": y_bottom,
                    "gridline_y": sorted(ys),
                    "score": len(ys) + score_bonus,
                    "source": source_name,
                }
            )

    return candidates


def horizontal_support_for_box(
    h_lines: list[dict],
    x0: int,
    x1: int,
    y_top: int,
    y_bottom: int,
    image_width: int,
) -> list[dict]:
    """Find horizontal lines that support a proposed rectangular box."""
    tol = max(8, int(0.01 * image_width))
    support: list[dict] = []
    box_width = x1 - x0

    for line in h_lines:
        if not (y_top - tol <= line["y"] <= y_bottom + tol):
            continue
        overlap = max(0, min(x1, line["x1"]) - max(x0, line["x0"]))
        if overlap >= 0.70 * box_width:
            support.append(line)

    return support


def candidates_from_vertical_pairs(
    h_lines: list[dict],
    v_lines: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Create panel candidates from pairs of vertical line segments.

    This is a supplemental route — candidates are only accepted when
    they have sufficient horizontal gridline support.
    """
    candidates: list[dict] = []
    if len(v_lines) < 2:
        return candidates

    tol_y = max(12, int(0.018 * image_height))

    for i in range(len(v_lines)):
        for j in range(i + 1, len(v_lines)):
            vl = v_lines[i]
            vr = v_lines[j]
            x0, x1 = sorted([vl["x"], vr["x"]])
            panel_width = x1 - x0

            if panel_width < 0.14 * image_width or panel_width > 0.95 * image_width:
                continue

            y_top = max(vl["y0"], vr["y0"])
            y_bottom = min(vl["y1"], vr["y1"])
            panel_height = y_bottom - y_top

            if panel_height < 0.10 * image_height:
                continue

            if abs(vl["y0"] - vr["y0"]) < tol_y and abs(vl["y1"] - vr["y1"]) < tol_y:
                y_top = int(np.median([vl["y0"], vr["y0"]]))
                y_bottom = int(np.median([vl["y1"], vr["y1"]]))

            support = horizontal_support_for_box(h_lines, x0, x1, y_top, y_bottom, image_width)
            if len(support) < 3:
                continue

            grid_ys = sorted(set(int(s["y"]) for s in support))

            candidates.append(
                {
                    "x0": int(x0),
                    "x1": int(x1),
                    "y_top": int(min(grid_ys + [y_top])),
                    "y_bottom": int(max(grid_ys + [y_bottom])),
                    "gridline_y": grid_ys,
                    "score": len(support) + 1.0,
                    "source": "vertical_pairs_low_saturation",
                }
            )

    return candidates


# ===================================================================
# Deduplication
# ===================================================================

def dedupe_panel_candidates(
    candidates: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Remove duplicate, nested, tiny, and overlapping panel candidates."""
    cleaned = []
    for c in candidates:
        if c["x1"] <= c["x0"] or c["y_bottom"] <= c["y_top"]:
            continue
        if box_area(c) < 0.01 * image_width * image_height:
            continue
        cleaned.append(c)

    def priority(c):
        source_priority = {
            "legacy_horizontal_primary": 0,
            "axis_horizontal_extents": 1,
            "vertical_pairs_low_saturation": 2,
        }.get(c["source"], 3)
        return (source_priority, -c["score"], box_area(c))

    cleaned = sorted(cleaned, key=priority)

    kept: list[dict] = []
    for cand in cleaned:
        reject = False
        for prev in kept:
            if box_iou(cand, prev) >= 0.55:
                reject = True
                break
            if contains_box(prev, cand) and box_area(cand) < 0.85 * box_area(prev):
                reject = True
                break
        if not reject:
            kept.append(cand)

    final: list[dict] = []
    for p in kept:
        contained = [
            q for q in kept if q is not p and box_area(q) < 0.75 * box_area(p) and contains_box(p, q)
        ]
        if len(contained) >= 2:
            continue
        final.append(p)

    final = sorted(final, key=lambda d: (d["y_top"], d["x0"]))
    for i, p in enumerate(final, start=1):
        p["panel_id"] = i

    return final


# ===================================================================
# Post-hoc panel subdivision
# ===================================================================

def _subdivide_panel_vertically(panel: dict, n_parts: int) -> list[dict]:
    """Split one panel into *n_parts* vertically at its gridlines.

    Uses the *n_parts - 1* largest gaps among the panel's gridlines as
    split points.  Each resulting sub-panel inherits x-extent and
    source from the parent.
    """
    gridlines = sorted(panel["gridline_y"])
    if len(gridlines) < 2 * n_parts - 1:
        # Not enough gridlines to form n_parts panels with ≥ 2 lines each.
        return [panel]

    gaps = np.diff(gridlines)
    if len(gaps) < n_parts - 1:
        return [panel]

    split_indices = list(np.argsort(gaps)[::-1][: n_parts - 1])
    split_indices.sort()

    parts: list[dict] = []
    prev = 0
    for si in split_indices:
        ys = gridlines[prev : si + 1]
        prev = si + 1
        if len(ys) >= 2:
            parts.append(
                {
                    "x0": panel["x0"],
                    "x1": panel["x1"],
                    "y_top": int(min(ys)),
                    "y_bottom": int(max(ys)),
                    "gridline_y": ys,
                    "score": len(ys) + 2.0,
                    "source": panel["source"] + "_subdivided",
                }
            )
    # Last segment
    ys = gridlines[prev:]
    if len(ys) >= 2:
        parts.append(
            {
                "x0": panel["x0"],
                "x1": panel["x1"],
                "y_top": int(min(ys)),
                "y_bottom": int(max(ys)),
                "gridline_y": ys,
                "score": len(ys) + 2.0,
                "source": panel["source"] + "_subdivided",
            }
        )

    return parts if len(parts) == n_parts else [panel]


# ===================================================================
# Layout selection helpers
# ===================================================================

def infer_expected_panels(
    layout_mode: str = "auto",
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
) -> Optional[int]:
    """Determine the expected number of panels from user settings."""
    if expected_panels is not None:
        return int(expected_panels)
    if expected_rows is not None and expected_cols is not None:
        return int(expected_rows) * int(expected_cols)
    if layout_mode == "single":
        return 1
    return None


def cluster_panel_rows(panels: list[dict], tol: int = 18) -> list[list[dict]]:
    """Group panel candidates into rows based on similar y extents."""
    rows: list[list[dict]] = []
    for p in sorted(panels, key=lambda d: (d["y_top"], d["x0"])):
        placed = False
        for row in rows:
            ref = row[0]
            if abs(p["y_top"] - ref["y_top"]) <= tol or abs(p["y_bottom"] - ref["y_bottom"]) <= tol:
                row.append(p)
                placed = True
                break
        if not placed:
            rows.append([p])
    for row in rows:
        row.sort(key=lambda d: d["x0"])
    return rows


def cluster_panel_cols(panels: list[dict], tol: int = 18) -> list[list[dict]]:
    """Group panel candidates into columns based on similar x extents."""
    cols: list[list[dict]] = []
    for p in sorted(panels, key=lambda d: (d["x0"], d["y_top"])):
        placed = False
        for col in cols:
            ref = col[0]
            if abs(p["x0"] - ref["x0"]) <= tol or abs(p["x1"] - ref["x1"]) <= tol:
                col.append(p)
                placed = True
                break
        if not placed:
            cols.append([p])
    for col in cols:
        col.sort(key=lambda d: d["y_top"])
    return cols


def panel_quality(p: dict) -> float:
    """Score a panel candidate (higher is better)."""
    return float(p.get("score", 0)) + 0.00001 * box_area(p)


def choose_best_same_x_stack(
    panels: list[dict],
    expected_rows: Optional[int] = None,
    expected_panels: Optional[int] = None,
) -> list[dict]:
    """Pick vertically stacked panels with similar x extents.

    If the result has fewer panels than *expected_rows*, oversized
    panels are subdivided at their gridlines.
    """
    if not panels:
        return []

    target_n = expected_panels or expected_rows

    candidates = [p for p in panels if p["source"] in ["legacy_horizontal_primary", "axis_horizontal_extents"]]
    if not candidates:
        candidates = panels[:]

    groups: list[list[dict]] = []
    for p in sorted(candidates, key=lambda d: d["x0"]):
        placed = False
        for g in groups:
            ref = g[0]
            tol = max(12, 0.05 * max(1, ref["x1"] - ref["x0"]))
            if abs(p["x0"] - ref["x0"]) <= tol and abs(p["x1"] - ref["x1"]) <= tol:
                g.append(p)
                placed = True
                break
        if not placed:
            groups.append([p])

    def group_score(g):
        g_sorted = sorted(g, key=lambda d: d["y_top"])
        count_penalty = abs(len(g_sorted) - int(target_n)) * 1000 if target_n is not None else 0
        return -count_penalty + sum(panel_quality(p) for p in g_sorted)

    best = max(groups, key=group_score)
    best = sorted(best, key=lambda d: d["y_top"])

    if target_n is not None and len(best) > int(target_n):
        best = sorted(best, key=panel_quality, reverse=True)[: int(target_n)]
        best = sorted(best, key=lambda d: d["y_top"])

    # --- Post-hoc subdivision ---
    if target_n is not None and len(best) < int(target_n):
        deficit = int(target_n) - len(best)
        expanded: list[dict] = []
        for p in best:
            n_gridlines = len(p.get("gridline_y", []))
            # Can this panel be split to help fill the deficit?
            if deficit > 0 and n_gridlines >= 4:
                n_sub = min(deficit + 1, n_gridlines // 2)
                subs = _subdivide_panel_vertically(p, n_sub)
                if len(subs) > 1:
                    expanded.extend(subs)
                    deficit -= len(subs) - 1
                else:
                    expanded.append(p)
            else:
                expanded.append(p)
        best = sorted(expanded, key=lambda d: d["y_top"])

    return best


def choose_best_same_y_row(
    panels: list[dict],
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
) -> list[dict]:
    """Pick horizontally arranged panels with similar y extents."""
    if not panels:
        return []

    target_n = expected_panels or expected_cols
    candidates = panels[:]
    rows = cluster_panel_rows(candidates, tol=22)

    def row_score(row):
        row_sorted = sorted(row, key=lambda d: d["x0"])
        count_penalty = abs(len(row_sorted) - int(target_n)) * 1000 if target_n is not None else 0
        return -count_penalty + sum(panel_quality(p) for p in row_sorted)

    best = max(rows, key=row_score)
    best = sorted(best, key=lambda d: d["x0"])

    if target_n is not None and len(best) > int(target_n):
        best = sorted(best, key=panel_quality, reverse=True)[: int(target_n)]
        best = sorted(best, key=lambda d: d["x0"])

    return best


def choose_best_grid(
    panels: list[dict],
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
    all_h_lines: Optional[list[dict]] = None,
) -> list[dict]:
    """Pick a grid of panels from candidate boxes.

    Improved strategy:

    1. Group candidates into rows.
    2. If there are fewer rows than expected, try subdividing tall
       panels vertically.
    3. Within each row, ensure the right number of columns.
    """
    if not panels:
        return []

    target_n = infer_expected_panels("grid", expected_rows, expected_cols, expected_panels)
    er = int(expected_rows) if expected_rows else None
    ec = int(expected_cols) if expected_cols else None

    # Start with highest-quality panels.
    candidates = sorted(panels, key=panel_quality, reverse=True)
    if target_n is not None and len(candidates) > int(target_n) * 2:
        candidates = candidates[: int(target_n) * 2]

    # --- Step 1: try to get the right number of rows ---
    rows = cluster_panel_rows(candidates, tol=24)
    rows = sorted(rows, key=lambda r: np.median([p["y_top"] for p in r]))

    if er is not None and len(rows) < er:
        # Try subdividing tall panels to create more rows.
        expanded: list[dict] = []
        for p in candidates:
            n_gridlines = len(p.get("gridline_y", []))
            if n_gridlines >= 2 * er:
                subs = _subdivide_panel_vertically(p, er)
                expanded.extend(subs)
            else:
                expanded.append(p)

        if expanded:
            all_expanded = expanded
            # Re-cluster
            rows = cluster_panel_rows(all_expanded, tol=24)
            rows = sorted(rows, key=lambda r: np.median([p["y_top"] for p in r]))

    if er is not None:
        rows = rows[:er]

    # --- Step 2: within each row select the right number of columns ---
    selected: list[dict] = []
    for row in rows:
        if ec is not None:
            row_sorted = sorted(row, key=panel_quality, reverse=True)[:ec]
        else:
            row_sorted = row
        selected.extend(row_sorted)

    selected = sorted(selected, key=lambda d: (d["y_top"], d["x0"]))

    return selected


# ===================================================================
# Layout hint application
# ===================================================================

def apply_layout_hint(
    panels: list[dict],
    layout_mode: str = "auto",
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
    all_h_lines: Optional[list[dict]] = None,
) -> list[dict]:
    """Constrain panel candidates according to a user-supplied layout hint."""
    layout_mode = str(layout_mode).lower().strip()
    target_n = infer_expected_panels(layout_mode, expected_rows, expected_cols, expected_panels)

    if layout_mode == "auto":
        selected = panels[:]
    elif layout_mode == "single":
        selected = [max(panels, key=panel_quality)] if panels else []
    elif layout_mode == "stacked":
        selected = choose_best_same_x_stack(panels, expected_rows=expected_rows, expected_panels=target_n)
    elif layout_mode == "horizontal":
        selected = choose_best_same_y_row(panels, expected_cols=expected_cols, expected_panels=target_n)
    elif layout_mode == "grid":
        selected = choose_best_grid(
            panels,
            expected_rows=expected_rows,
            expected_cols=expected_cols,
            expected_panels=target_n,
            all_h_lines=all_h_lines,
        )
    else:
        raise ValueError("layout_mode must be 'auto', 'single', 'stacked', 'horizontal', or 'grid'")

    selected = sorted(selected, key=lambda d: (d["y_top"], d["x0"]))
    for i, p in enumerate(selected, start=1):
        p["panel_id"] = i
        p["layout_mode"] = layout_mode

    return selected


# ===================================================================
# Panels from axis-line frames and from user-supplied boxes
# ===================================================================

def _panel_from_frame(frame: Frame, source: str) -> dict:
    """Convert a :class:`~trace_digitiser.frames.Frame` into a panel dict."""
    anchors_y = [frame.bottom] + ([frame.top] if frame.has_top else []) + frame.gridlines_h + frame.ticks_y
    anchors_x = [frame.left] + ([frame.right] if frame.has_right else []) + frame.gridlines_v + frame.ticks_x
    return {
        "x0": int(round(frame.left)),
        "x1": int(round(frame.right)),
        "y_top": int(round(frame.top)),
        "y_bottom": int(round(frame.bottom)),
        "edges": {
            "left": float(frame.left), "right": float(frame.right),
            "top": float(frame.top), "bottom": float(frame.bottom),
        },
        "has_top": frame.has_top,
        "has_right": frame.has_right,
        "kind": frame.kind,
        "line_thickness": frame.line_thickness,
        # Rows of horizontal gridlines (plus the frame's own top and bottom),
        # as in the original pipeline.
        "gridline_y": sorted({int(round(v)) for v in [frame.bottom, frame.top] + frame.gridlines_h}),
        "gridline_x": sorted({int(round(v)) for v in frame.gridlines_v}),
        # Sub-pixel positions where a tick label may be anchored.
        "tick_rows": _dedupe_positions(anchors_y),
        "tick_cols": _dedupe_positions(anchors_x),
        "score": 10.0 + len(frame.gridlines_h),
        "source": source,
    }


def _dedupe_positions(values: list[float], tol: float = 1.5) -> list[float]:
    """Sort positions and merge any that are within *tol* pixels of each other."""
    out: list[float] = []
    for v in sorted(float(v) for v in values):
        if out and v - out[-1] <= tol:
            out[-1] = 0.5 * (out[-1] + v)
        else:
            out.append(v)
    return out


def panels_from_boxes(rgb: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[dict]:
    """Build panels from user-supplied plot areas.

    Each box is ``(x_left, y_top, x_right, y_bottom)`` in image pixels: the
    positions of the axis lines (or, with no frame, of the limits of the data
    region).  Gridlines and tick marks inside the box are still detected, so
    that tick labels can be anchored to them.
    """
    h, w = rgb.shape[:2]
    dark = darkness_map(rgb)
    sat = _saturation(rgb)
    panels = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("each plot area must be (x_left, y_top, x_right, y_bottom)")
        left, top, right, bottom = (float(v) for v in box)
        if not (0 <= left < right <= w - 1 + 1e-6 and 0 <= top < bottom <= h - 1 + 1e-6):
            raise ValueError(
                f"plot area {box} is not inside the {w}x{h} image, or is not (x_left, y_top, x_right, y_bottom)"
            )
        frame = Frame(left=left, right=right, top=top, bottom=bottom, source="user")
        _add_gridlines_and_ticks(frame, dark, sat)
        panels.append(_panel_from_frame(frame, "user_plot_area"))
    return panels


def _cluster(values: list[float], tol: float) -> list[float]:
    groups: list[list[float]] = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def _complete_grid(rgb: np.ndarray, panels: list[dict]) -> list[dict]:
    """Plot areas missing from a grid of plots (a heat map whose axis lines its dark colours hide, say): where the
    rows and columns of the plots found cross and no plot was found, a box with a dark frame all round is one."""
    dark = darkness_map(rgb)
    e = [p["edges"] for p in panels]
    tol = 4.0
    lefts, rights = _cluster([x["left"] for x in e], tol), _cluster([x["right"] for x in e], tol)
    tops, bottoms = _cluster([x["top"] for x in e], tol), _cluster([x["bottom"] for x in e], tol)
    if len(lefts) != len(rights) or len(tops) != len(bottoms) or len(lefts) * len(tops) <= len(panels):
        return []
    added = []
    for t, b in zip(tops, bottoms):
        for left, r in zip(lefts, rights):
            if any(abs(x["left"] - left) <= tol and abs(x["top"] - t) <= tol for x in e):
                continue
            box = [int(round(v)) for v in (left, t, r, b)]
            x0, y0, x1, y1 = box
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            # a frame: the four sides dark along nearly all their length (within a pixel of where they should be)
            sides = [
                dark[max(0, y0 - 1): y0 + 2, x0: x1 + 1].max(axis=0),
                dark[max(0, y1 - 1): y1 + 2, x0: x1 + 1].max(axis=0),
                dark[y0: y1 + 1, max(0, x0 - 1): x0 + 2].max(axis=1),
                dark[y0: y1 + 1, max(0, x1 - 1): x1 + 2].max(axis=1),
            ]
            if all(float((s_ > 140).mean()) >= 0.9 for s_ in sides):
                added.extend(panels_from_boxes(rgb, [(float(x0), float(y0), float(x1), float(y1))]))
    return added


def _select_frames(
    frames: list[Frame],
    layout_mode: str,
    target_n: Optional[int],
) -> list[Frame]:
    """Apply the layout hint to axis-line frames."""
    if not frames:
        return []
    if layout_mode == "single":
        return [max(frames, key=lambda f: f.area)]
    if target_n is not None and len(frames) > target_n:
        # More frames than plots: keep the largest, which drops stray boxes.
        frames = sorted(frames, key=lambda f: f.area, reverse=True)[:target_n]
    return _reading_order(frames)


def _lines_for_overlay(horizontal: list[AxisLine], vertical: list[AxisLine]) -> tuple[list[dict], list[dict]]:
    """Axis lines as the dicts the diagnostic overlay expects."""
    h_lines = [
        {"orientation": "h", "x0": ln.start, "x1": ln.end, "y": int(round(ln.pos)),
         "span": ln.length, "thickness": ln.thickness, "source": "axis_line"}
        for ln in horizontal
    ]
    v_lines = [
        {"orientation": "v", "x": int(round(ln.pos)), "y0": ln.start, "y1": ln.end,
         "span": ln.length, "thickness": ln.thickness, "source": "axis_line"}
        for ln in vertical
    ]
    return h_lines, v_lines


# ===================================================================
# Original gridline-clustering route
# ===================================================================

def find_plot_panels_from_gridlines(
    rgb: np.ndarray,
    layout_mode: str = "auto",
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Detect plot panels by clustering horizontal gridlines (the 1.x method).

    Returns ``(panels, h_lines, v_lines)``.  The left and right bounds come
    from the extent of the horizontal lines, so they include any tick marks
    that stick out past the axis; prefer :func:`find_plot_panels`.
    """
    h, w = rgb.shape[:2]

    # Primary route: horizontal gridline row projection.
    legacy_h_lines = detect_long_horizontal_lines(rgb, min_width_frac=0.25)
    legacy_candidates = candidates_from_horizontal_extents(
        legacy_h_lines, w, h,
        source_name="legacy_horizontal_primary",
        score_bonus=3.0,
        expected_rows=expected_rows,
    )

    # Supplemental route: axis/grid segments after suppressing coloured traces.
    axis_h_lines = detect_axis_line_segments(rgb, "horizontal", min_length_frac=0.15)
    axis_v_lines = detect_axis_line_segments(rgb, "vertical", min_length_frac=0.10)

    axis_h_candidates = candidates_from_horizontal_extents(
        axis_h_lines, w, h,
        source_name="axis_horizontal_extents",
        score_bonus=1.5,
        expected_rows=expected_rows,
    )
    vertical_candidates = candidates_from_vertical_pairs(axis_h_lines, axis_v_lines, w, h)

    all_h_lines = legacy_h_lines + axis_h_lines
    candidates = legacy_candidates + axis_h_candidates + vertical_candidates
    all_candidate_panels = dedupe_panel_candidates(candidates, w, h)
    panels = apply_layout_hint(
        all_candidate_panels,
        layout_mode=layout_mode,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        expected_panels=expected_panels,
        all_h_lines=all_h_lines,
    )
    for p in panels:
        p["edges"] = {
            "left": float(p["x0"]), "right": float(p["x1"]),
            "top": float(p["y_top"]), "bottom": float(p["y_bottom"]),
        }
        p.setdefault("has_top", True)
        p.setdefault("has_right", True)
        p.setdefault("kind", "box")
        p.setdefault("line_thickness", 1)
        p.setdefault("gridline_x", [])
        p.setdefault("tick_rows", [float(v) for v in p["gridline_y"]])
        p.setdefault("tick_cols", [])

    return panels, legacy_h_lines, axis_v_lines


# ===================================================================
# Main entry point
# ===================================================================

def find_plot_panels(
    rgb: np.ndarray,
    layout_mode: str = "auto",
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
    plot_area: Optional[list[tuple[float, float, float, float]]] = None,
    notes: Optional[list[str]] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Detect plot panels in *rgb*.

    Parameters
    ----------
    rgb : ndarray, shape (H, W, 3)
    layout_mode, expected_rows, expected_cols, expected_panels
        Layout hints.  With axis-line detection they are only needed to
        choose among candidates; they are not required.
    plot_area : list of (x_left, y_top, x_right, y_bottom), optional
        Plot areas to use as given, skipping detection.
    notes : list of str, optional
        If supplied, human-readable remarks about how the panels were found
        (for example a mismatch with the layout hint) are appended to it.

    Returns
    -------
    (panels, h_lines, v_lines)
        *panels* is the final list, numbered row by row from ``panel_id`` 1.
    """
    layout_mode = str(layout_mode).lower().strip()
    if layout_mode not in ("auto", "single", "stacked", "horizontal", "grid"):
        raise ValueError("layout_mode must be 'auto', 'single', 'stacked', 'horizontal', or 'grid'")
    if notes is None:
        notes = []
    target_n = infer_expected_panels(layout_mode, expected_rows, expected_cols, expected_panels)

    if plot_area is not None:
        panels = panels_from_boxes(rgb, list(plot_area))
        h_lines, v_lines = [], []
    else:
        frames, horizontal, vertical = find_frames_and_lines(rgb)
        frames = _select_frames(frames, layout_mode, target_n)
        panels = [_panel_from_frame(f, "axis_lines") for f in frames]
        h_lines, v_lines = _lines_for_overlay(horizontal, vertical)

        if not panels or (target_n is not None and len(panels) < target_n):
            # No usable axis lines (or fewer plots than the user expects):
            # fall back to gridline clustering if that does better.
            legacy, legacy_h, legacy_v = find_plot_panels_from_gridlines(
                rgb, layout_mode, expected_rows, expected_cols, expected_panels
            )
            better = len(legacy) > len(panels) if target_n is None else (
                abs(len(legacy) - target_n) < abs(len(panels) - target_n)
            )
            if legacy and better:
                notes.append(
                    f"Axis lines gave {len(panels)} plot area(s); used gridline clustering instead "
                    f"({len(legacy)} found). Left/right bounds from this route can include tick marks - "
                    "check the overlay, or pass plot_area."
                )
                panels, h_lines, v_lines = legacy, legacy_h, legacy_v

    if plot_area is None and len(panels) >= 3:
        added = _complete_grid(rgb, panels)
        if added:
            panels = sorted(panels + added, key=lambda p: (round(p["edges"]["top"] / 10), p["edges"]["left"]))
            notes.append(
                f"{len(added)} plot area(s) completing a grid of plots were found by their frames alone "
                "(their axis lines were hidden by what is drawn beside them); check them on the overlay."
            )

    if not panels:
        notes.append(
            "No plot area was found. Pass plot_area=(x_left, y_top, x_right, y_bottom) to set it by hand."
        )
    elif target_n is not None and len(panels) != target_n:
        notes.append(f"Expected {target_n} plot area(s) from the layout hint but found {len(panels)}.")

    for i, p in enumerate(panels, start=1):
        p["panel_id"] = i
        p["layout_mode"] = layout_mode

    return panels, h_lines, v_lines
