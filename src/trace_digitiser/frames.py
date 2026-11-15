"""Plot-area detection from axis lines.

A plot's data region is bounded by its axis lines.  This module finds the
long, thin, straight dark lines in a figure, groups the ones that meet into
axes, and reports each plot area with sub-pixel edges.  It handles both
closed frames (four sides) and open "L" axes (left and bottom only), and it
is unaffected by tick marks that stick out beyond a corner, because every
edge is taken from the *position* of the line on that side rather than from
the extent of the lines that cross it.

The detector is the default route used by
:func:`trace_digitiser.panel_detection.find_plot_panels`; the older
gridline-clustering route remains as a fallback for charts that have
gridlines but no axis lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Darkness (background minus pixel, 0-255) above which a pixel counts as ink.
INK_THRESHOLD = 20.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AxisLine:
    """A long, thin, straight dark line."""

    orientation: str  # "h" or "v"
    pos: float  # sub-pixel centre: row for "h", column for "v"
    start: int  # first pixel along the line (column for "h", row for "v")
    end: int  # last pixel along the line, inclusive
    thickness: int
    darkness: float  # mean darkness along the line, 0-255
    edge: bool = False  # the straight edge of a filled area (a bar, a dense trace) rather than a line on its own

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class Frame:
    """One plot area, bounded by axis lines.

    ``left``, ``right``, ``top`` and ``bottom`` are sub-pixel image
    coordinates (pixel centres at integers).  A side that has no line of its
    own (the top and right of "L" axes) is taken from the end of the
    perpendicular axis line and is flagged in ``has_top`` / ``has_right``.
    """

    left: float
    right: float
    top: float
    bottom: float
    has_left: bool = True
    has_right: bool = True
    has_top: bool = True
    has_bottom: bool = True
    line_thickness: int = 1
    gridlines_h: list[float] = field(default_factory=list)
    gridlines_v: list[float] = field(default_factory=list)
    ticks_y: list[float] = field(default_factory=list)
    ticks_x: list[float] = field(default_factory=list)
    source: str = "axis_lines"
    from_edges: bool = False  # its left or bottom side is the edge of a filled area, not a line

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def kind(self) -> str:
        """``"box"`` for a closed frame, ``"L"`` for left+bottom axes, else ``"partial"``."""
        if self.has_left and self.has_right and self.has_top and self.has_bottom:
            return "box"
        if self.has_left and self.has_bottom and not self.has_top and not self.has_right:
            return "L"
        return "partial"


# ---------------------------------------------------------------------------
# Darkness map
# ---------------------------------------------------------------------------

def darkness_map(rgb: np.ndarray) -> np.ndarray:
    """Return how much darker each pixel is than the local paper colour (0-255).

    The background is estimated locally (brightest value in a wide
    neighbourhood), so unevenly lit scans and off-white paper do not turn
    into ink.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    factor = max(1, min(h, w) // 160)
    small = cv2.resize(gray, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    k = max(9, int(0.25 * min(small.shape)) | 1)
    background = cv2.dilate(small, np.ones((k, k), np.uint8))
    background = cv2.GaussianBlur(background, (k, k), 0)
    background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(background.astype(np.float32) - gray.astype(np.float32), 0, 255)


# ---------------------------------------------------------------------------
# Line detection
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs (inclusive) of the True runs in a 1-D mask."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def _detect_horizontal(
    dark: np.ndarray,
    sat: np.ndarray,
    min_len: int,
    max_thick: int,
    thr: float,
) -> list[AxisLine]:
    """Find long thin grey/black horizontal lines in a darkness map."""
    h, w = dark.shape
    if w < min_len or h < 3:
        return []

    # Average along the line direction so that noise and JPEG drop-outs do not
    # break a line into pieces, then keep only runs at least ``min_len`` long.
    k = int(np.clip(min_len // 5, 3, 15)) | 1
    smooth = cv2.blur(dark, (k, 1))
    ink_s = (smooth > thr).astype(np.uint8)
    opened = cv2.morphologyEx(ink_s, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    k_trim = 3 if min(h, w) < 600 else 5
    lines: list[AxisLine] = []
    seen: set[tuple[int, int, int]] = set()

    def emit(r0: int, r1: int, x: int, cw: int, check_colour: bool = True, edge: bool = False) -> None:
        """Measure the line occupying rows r0..r1 near columns x..x+cw and record it."""
        # Exact extent.  The smoothing above spreads a line's ends and can
        # join it to a nearby label, so re-measure along the line itself:
        # the line continues for as long as there is ink in its rows, and
        # stops at the first real gap.
        band = dark[r0 : r1 + 1, :].max(axis=0)
        ref = float(np.median(band[x : x + cw]))
        support = cv2.blur(band[None, :], (k_trim, 1))[0] >= max(0.6 * thr, 0.35 * ref)
        for c0, c1 in _runs(support):
            if min(c1, x + cw - 1) - max(c0, x) + 1 < min_len // 2:
                continue  # belongs to something else in the same rows
            while c0 < c1 and band[c0] <= thr:
                c0 += 1
            while c1 > c0 and band[c1] <= thr:
                c1 -= 1
            if c1 - c0 + 1 < min_len or (r0, c0, c1) in seen:
                continue
            seen.add((r0, c0, c1))

            on_line = dark[r0 : r1 + 1, c0 : c1 + 1] > thr
            if check_colour and on_line.any() and float(np.median(sat[r0 : r1 + 1, c0 : c1 + 1][on_line])) > 0.25:
                continue  # a free-standing coloured line is a trace, not an axis

            # Sub-pixel centre from the darkness profile across the line.  If
            # other ink lies against one side (a trace resting on the axis),
            # subtract it so that it does not drag the centre towards itself.
            p_lo = max(0, r0 - 1)
            p_hi = min(h, r1 + 2)
            profile = dark[p_lo:p_hi, c0 : c1 + 1].mean(axis=1)
            sides = [profile[0] if p_lo < r0 else 0.0, profile[-1] if p_hi > r1 + 1 else 0.0]
            base = max(sides) if max(sides) > 0.5 * profile.max() else 0.5 * min(sides)
            weights = np.clip(profile - base, 0, None)
            rows = np.arange(p_lo, p_hi)
            pos = float((weights * rows).sum() / weights.sum()) if weights.sum() > 0 else 0.5 * (r0 + r1)
            lines.append(AxisLine("h", pos, c0, c1, r1 - r0 + 1, float(band[c0 : c1 + 1].mean()), edge=edge))

    # Straight, sharp edges of thick bands: (edge_row, direction, x, cw).
    edges: list[tuple[int, int, int, int]] = []

    for lab in range(1, n_labels):
        x, y, cw, ch, _ = (int(v) for v in stats[lab])
        if cw < min_len:
            continue
        # Fraction of the component's extent that it covers, row by row.
        cov = (labels[y : y + ch, x : x + cw] == lab).mean(axis=1)
        if cov.max() < 0.8:
            continue

        for b0, b1 in _runs(cov >= 0.8 * cov.max()):
            r0 = y + b0
            r1 = y + b1
            cov_above = float(ink_s[r0 - 2, x : x + cw].mean()) if r0 - 2 >= 0 else 0.0
            cov_below = float(ink_s[r1 + 2, x : x + cw].mean()) if r1 + 2 < h else 0.0
            thick = r1 - r0 + 1 > max_thick
            if thick or (min(cov_above, cov_below) <= 0.25 and max(cov_above, cov_below) > 0.55):
                # Not a free-standing line: a filled region, or a dense trace
                # lying against an axis.  In the second case the axis is the
                # straight sharp edge of the band, so keep such edges.
                if cov_below <= 0.25:
                    edges.append((r1, -1, x, cw))
                if cov_above <= 0.25:
                    edges.append((r0, +1, x, cw))
                continue
            # Rows inside a dense trace have ink on both sides: not a line.
            if min(cov_above, cov_below) > 0.55:
                continue
            emit(r0, r1, x, cw)

    typical = int(round(np.median([ln.thickness for ln in lines]))) if lines else 1
    for edge, direction, x, cw in edges:
        r0, r1 = (edge - typical + 1, edge) if direction < 0 else (edge, edge + typical - 1)
        if r0 >= 0 and r1 < h:
            # No colour test here: an axis under a coloured trace takes on its tint.
            emit(r0, r1, x, cw, check_colour=False, edge=True)

    return sorted(lines, key=lambda ln: (ln.pos, ln.start))


def _saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV-style saturation in 0-1."""
    mx = rgb.max(axis=2).astype(np.float32)
    mn = rgb.min(axis=2).astype(np.float32)
    return (mx - mn) / np.maximum(mx, 1.0)


def detect_axis_lines(
    rgb: np.ndarray,
    dark: np.ndarray | None = None,
    min_len_frac: float = 0.08,
) -> tuple[list[AxisLine], list[AxisLine]]:
    """Detect long thin horizontal and vertical lines.

    Returns ``(horizontal, vertical)``.
    """
    if dark is None:
        dark = darkness_map(rgb)
    h, w = dark.shape
    sat = _saturation(rgb)
    max_thick = max(4, int(round(0.012 * min(h, w))))
    horizontal = _detect_horizontal(dark, sat, max(20, int(min_len_frac * w)), max_thick, INK_THRESHOLD)
    vertical_t = _detect_horizontal(
        np.ascontiguousarray(dark.T), np.ascontiguousarray(sat.T),
        max(20, int(min_len_frac * h)), max_thick, INK_THRESHOLD,
    )
    vertical = [AxisLine("v", ln.pos, ln.start, ln.end, ln.thickness, ln.darkness, ln.edge) for ln in vertical_t]
    return horizontal, vertical


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def _assemble_frames(horizontal: list[AxisLine], vertical: list[AxisLine], tol: float) -> list[Frame]:
    """Build plot areas from axis lines.

    Every plot area has a bottom-left corner where the y axis ends on the
    x axis and the x axis ends on the y axis.  Gridlines never form such a
    corner (they run on past the lines they cross), so starting from these
    corners finds each plot exactly once however many gridlines it has.  A
    tick mark that sticks out past a corner lengthens a line by a few pixels,
    which the ``ext`` allowance absorbs.
    """
    frames: list[Frame] = []
    for left in vertical:
        for bottom in horizontal:
            ext = max(tol, 0.04 * min(left.length, bottom.length))
            if not (bottom.pos - tol <= left.end <= bottom.pos + ext):
                continue
            if not (left.pos - ext <= bottom.start <= left.pos + tol):
                continue

            # Right side: a vertical line on which the x axis ends.  A gridline
            # at the last tick can lie within the allowance too, so of the
            # lines that qualify the outermost one is the axis.
            right = None
            for cand in vertical:
                if cand is left or cand.pos <= left.pos + 0.5 * bottom.length:
                    continue
                if abs(cand.pos - bottom.end) > ext or not (bottom.pos - tol <= cand.end <= bottom.pos + ext):
                    continue
                if cand.length >= 0.6 * left.length and (right is None or cand.pos > right.pos):
                    right = cand
            # Top side: a horizontal line on which the y axis ends.
            top = None
            for cand in horizontal:
                if cand is bottom or cand.pos >= bottom.pos - 0.5 * left.length:
                    continue
                if abs(cand.pos - left.start) > ext or not (left.pos - ext <= cand.start <= left.pos + tol):
                    continue
                if cand.length >= 0.6 * bottom.length and (top is None or cand.pos < top.pos):
                    top = cand

            frames.append(
                Frame(
                    left=left.pos,
                    bottom=bottom.pos,
                    right=right.pos if right is not None else float(bottom.end),
                    top=top.pos if top is not None else float(left.start),
                    has_right=right is not None,
                    has_top=top is not None,
                    line_thickness=max(1, int(round(0.5 * (left.thickness + bottom.thickness)))),
                    from_edges=left.edge or bottom.edge,
                )
            )
    return frames


def _contains(outer: Frame, inner: Frame, margin: float = 3.0) -> bool:
    return (
        outer.left <= inner.left + margin
        and outer.right >= inner.right - margin
        and outer.top <= inner.top + margin
        and outer.bottom >= inner.bottom - margin
    )


def _overlap(a: Frame, b: Frame) -> float:
    """Area that two frames have in common."""
    ix = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    iy = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    return ix * iy


def _iou(a: Frame, b: Frame) -> float:
    inter = _overlap(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _filter_frames(frames: list[Frame], image_w: int, image_h: int) -> list[Frame]:
    """Drop tiny frames, duplicates, borders drawn around a plot, insets, and stray frames across a plot."""
    min_w = max(30.0, 0.06 * image_w)
    min_h = max(30.0, 0.06 * image_h)
    frames = [f for f in frames if f.width >= min_w and f.height >= min_h]

    # Duplicates: keep the more complete frame.
    frames.sort(key=lambda f: (f.kind != "box", -f.area))
    unique: list[Frame] = []
    for f in frames:
        if all(_iou(f, g) < 0.85 for g in unique):
            unique.append(f)

    # A frame made from the edge of a filled area (the side of a bar, the
    # edge of a legend's patch) inside a frame made of lines is not a plot.
    unique = [
        f for f in unique
        if not (f.from_edges and any(g is not f and not g.from_edges and _contains(g, f) for g in unique))
    ]
    # A frame that encloses another substantial frame is a border around the
    # figure, not a plot area.
    kept = [
        f for f in unique
        if not any(g is not f and _contains(f, g) and g.area >= 0.2 * f.area for g in unique)
    ]
    # A small frame inside a plot area is a legend box or an inset.
    kept = [
        f for f in kept
        if not any(g is not f and _contains(g, f) and f.area < 0.2 * g.area for g in kept)
    ]
    # Plot areas do not overlap.  A frame with lines on only two or three
    # sides that cuts across a complete one is an accident of stray lines -
    # what is left of a border drawn round the figure, say.
    boxes = [f for f in kept if f.kind == "box"]
    kept = [
        f for f in kept
        if f.kind == "box" or not any(_overlap(f, g) > 0.25 * min(f.area, g.area) for g in boxes)
    ]
    return _reading_order(kept)


def _reading_order(frames: list[Frame]) -> list[Frame]:
    """Sort frames row by row, left to right, treating nearly equal tops as one row."""
    if not frames:
        return []
    tol = 0.25 * min(f.height for f in frames)
    rows: list[list[Frame]] = []
    for f in sorted(frames, key=lambda f: f.top):
        if rows and abs(f.top - rows[-1][0].top) <= tol:
            rows[-1].append(f)
        else:
            rows.append([f])
    return [f for row in rows for f in sorted(row, key=lambda f: f.left)]


# ---------------------------------------------------------------------------
# Gridlines and tick marks inside / around a frame
# ---------------------------------------------------------------------------

def _projection_peaks(cov: np.ndarray, max_thick: int, min_cov: float = 0.7) -> list[float]:
    """Sharp peaks in a coverage profile: positions of thin full-length lines."""
    peaks: list[float] = []
    n = len(cov)
    for b0, b1 in _runs(cov >= min_cov):
        if b1 - b0 + 1 > max_thick:
            continue
        before = cov[b0 - 2] if b0 - 2 >= 0 else 0.0
        after = cov[b1 + 2] if b1 + 2 < n else 0.0
        if min(before, after) > 0.45:
            continue
        idx = np.arange(b0, b1 + 1)
        wts = cov[b0 : b1 + 1]
        peaks.append(float((idx * wts).sum() / wts.sum()))
    return peaks


def _visible_coverage(
    ink: np.ndarray,
    hidden: np.ndarray,
    axis: int,
    min_visible: float = 0.15,
    min_span: float = 0.6,
) -> np.ndarray:
    """Fraction of each row (``axis=1``) or column (``axis=0``) that is ink, among the pixels not hidden.

    Rows or columns with less than *min_visible* of their length in view
    give no evidence either way and score zero.  So do those whose ink does
    not reach across at least *min_span* of the length: what shows of a
    gridline between the strokes of a trace is spread from side to side,
    while a short bar or a legend sample is not.
    """
    if axis == 0:
        ink, hidden = ink.T, hidden.T
    n = ink.shape[1]
    visible = n - hidden.sum(axis=1)
    cov = ink.sum(axis=1) / np.maximum(visible, 1)
    first = ink.argmax(axis=1)
    last = n - 1 - ink[:, ::-1].argmax(axis=1)
    span = np.where(ink.any(axis=1), last - first + 1, 0)
    cov[(visible < min_visible * n) | (span < min_span * n)] = 0.0
    return cov


def _add_gridlines_and_ticks(frame: Frame, dark: np.ndarray, sat: np.ndarray) -> None:
    """Fill in ``gridlines_*`` and ``ticks_*`` for a frame (in place).

    Only neutral (grey/black) ink is considered, so that the vertical
    strokes of a coloured trace are not mistaken for gridlines or ticks.
    """
    h, w = dark.shape
    ink = (dark > INK_THRESHOLD) & (sat < 0.2)
    t = frame.line_thickness
    x0 = int(np.ceil(frame.left)) + t + 1
    x1 = int(np.floor(frame.right)) - t - 1
    y0 = int(np.ceil(frame.top)) + t + 1
    y1 = int(np.floor(frame.bottom)) - t - 1
    if x1 - x0 < 10 or y1 - y0 < 10:
        return
    max_thick = max(4, int(round(0.012 * min(h, w))))

    # Gridlines: thin lines that span the plot area from side to side.  A
    # trace drawn over a gridline hides part of it, so the line is judged on
    # the part that can be seen: the pixels that are not coloured ink, nor
    # right beside it (where the two blend).
    inside = np.s_[y0 : y1 + 1, x0 : x1 + 1]
    coloured = ((dark[inside] > INK_THRESHOLD) & ~ink[inside]).astype(np.uint8)
    hidden = cv2.dilate(coloured, np.ones((3, 3), np.uint8)) > 0
    seen = ink[inside] & ~hidden
    cov_rows = _visible_coverage(seen, hidden, axis=1)
    frame.gridlines_h = [y0 + p for p in _projection_peaks(cov_rows, max_thick)]
    cov_cols = _visible_coverage(seen, hidden, axis=0)
    frame.gridlines_v = [x0 + p for p in _projection_peaks(cov_cols, max_thick)]

    # Tick marks: short stubs attached to the outside (or inside) of an axis.
    # A stub must end within twice the usual tick length, which separates
    # ticks from gridlines and from traces that happen to touch the axis.
    # (in a large picture of a small plot - a figure scanned or exported at a high resolution - the ticks are long
    # for the plot: allow for the picture's size as well)
    tick_len = max(3, int(round(0.02 * min(frame.width, frame.height))), int(round(0.004 * max(h, w))))
    reach = 2 * tick_len + 1
    min_run = 2 if min(frame.width, frame.height) < 200 else 3

    def stubs(strip: np.ndarray) -> list[float]:
        """Positions along ``strip`` (axis side first) of short runs of ink that start at the axis."""
        if strip.shape[1] < reach:
            return []
        run = np.zeros(strip.shape[0], dtype=int)
        alive = np.ones(strip.shape[0], dtype=bool)
        for c in range(strip.shape[1]):
            alive &= strip[:, c]
            run += alive
        is_stub = (run >= min_run) & (run < strip.shape[1])
        return [0.5 * (b0 + b1) for b0, b1 in _runs(is_stub) if b1 - b0 + 1 <= max_thick]

    # A tick is darker than what it is drawn on - the paper, or a pale band shading part of the plot (which is
    # neutral ink too, and would make every row inside the axis a run of ink).
    lum = 255.0 - dark.astype(np.float32)
    ink = ink & ((cv2.dilate(lum, np.ones((15, 15), np.uint8)) - lum) > 50.0)
    xl = int(round(frame.left))
    yb = int(round(frame.bottom))
    half = t // 2 + 1
    # y ticks on the left axis: outside first, then inside.
    ya, yz = max(0, int(frame.top) - t), min(h, int(frame.bottom) + t + 1)
    outside = ink[ya:yz, max(0, xl - half - reach) : max(0, xl - half)][:, ::-1]
    inside = ink[ya:yz, xl + half + 1 : xl + half + 1 + reach]
    ticks_y = stubs(outside)
    if len(ticks_y) < 2:
        ticks_y = stubs(inside)
    frame.ticks_y = sorted(ya + p for p in ticks_y)
    # x ticks on the bottom axis.
    xa, xz = max(0, int(frame.left) - t), min(w, int(frame.right) + t + 1)
    outside = ink[yb + half + 1 : yb + half + 1 + reach, xa:xz].T
    inside = ink[max(0, yb - half - reach) : max(0, yb - half), xa:xz][::-1].T
    ticks_x = stubs(outside)
    if len(ticks_x) < 2:
        ticks_x = stubs(inside)
    frame.ticks_x = sorted(xa + p for p in ticks_x)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_frames_and_lines(rgb: np.ndarray) -> tuple[list[Frame], list[AxisLine], list[AxisLine]]:
    """Find the plot areas in a figure image, and the axis lines they were built from.

    Returns ``(frames, horizontal_lines, vertical_lines)`` with the frames
    ordered row by row, left to right.
    """
    dark = darkness_map(rgb)
    h, w = dark.shape
    horizontal, vertical = detect_axis_lines(rgb, dark)
    tol = float(max(3, int(round(0.012 * min(h, w)))))

    frames = _filter_frames(_assemble_frames(horizontal, vertical, tol), w, h)
    sat = _saturation(rgb)
    for frame in frames:
        _add_gridlines_and_ticks(frame, dark, sat)
    return frames, horizontal, vertical


def find_frames(rgb: np.ndarray) -> list[Frame]:
    """Find the plot areas in a figure image.

    Returns frames ordered row by row, left to right.  An empty list means
    no axis lines were found.
    """
    return find_frames_and_lines(rgb)[0]
