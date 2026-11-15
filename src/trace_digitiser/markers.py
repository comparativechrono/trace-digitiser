"""Finding markers (the dots, squares and triangles of a scatter plot) and the error bars on them.

A marker is a blob of one colour that is thicker than a line: where markers are joined by a line of their own
colour, as in ``plt.plot(x, y, "o-")``, an opening with a disc a little smaller than the markers removes the line
and leaves the markers. Markers that overlap are split by their area. Error bars are followed from each marker,
along its row (horizontal bars) or column (vertical), as far as they run unbroken, to their caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ._results import background_colour, colour_distance, exclusion_mask


@dataclass
class Marker:
    """One marker: its centre (image pixels), size and colour."""

    x: float
    y: float
    area: float
    radius: float
    rgb: tuple[int, int, int]
    series: int = 1
    shared: int = 1  # how many markers the blob it came from held (1: it stood alone)
    clipped: bool = False  # it touches the edge of the region searched, so its centre may be off
    errors: dict = field(default_factory=dict)  # "up", "down", "left", "right": the end of its error bar (pixels)


def fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill holes in *mask* smaller than *max_area* pixels (hollow markers become solid)."""
    m = mask.astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > max_area or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        out[labels == i] = True
    return out


def marker_radius(mask: np.ndarray) -> Optional[float]:
    """The radius of the markers in *mask*, or None if it holds only lines.

    The distance transform peaks at the middle of each stroke: about half a line's width on a line, about the
    marker's radius at the centre of a marker, whether the marker stands alone or sits on a line of its own colour.
    """
    m = mask.astype(np.uint8)
    if m.sum() < 6:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peaks = (dist >= cv2.dilate(dist, np.ones((5, 5), np.uint8))) & (dist >= 1.9)
    cand = dist[peaks]
    if cand.size == 0:
        return None
    # Markers are all one size, so their peaks agree; where lines cross or run thick the peaks are lower and
    # scattered. The markers' are the highest cluster of peaks with a fair share of all of them.
    need = max(3, int(0.15 * cand.size))
    for b in np.arange(np.floor(cand.max() * 2) / 2, 1.75, -0.5):
        near = cand[np.abs(cand - b) <= 0.75]
        if near.size >= need:
            return float(np.median(near))
    return float(np.median(cand))


def find_markers_in_mask(
    mask: np.ndarray,
    origin: tuple[int, int] = (0, 0),
    radius: Optional[float] = None,
    rgb_roi: Optional[np.ndarray] = None,
    series: int = 1,
    min_coverage: float = 0.0,
) -> list[Marker]:
    """The markers in a boolean *mask* (one colour), whose top-left pixel is *origin* (x, y) in the image.

    *min_coverage*: the share of the colour's pixels that must be in markers for it to count as a colour of markers
    at all (used when the colours were not given, but found).
    """
    if mask.sum() == 0:
        return []
    if radius:
        return _markers_at(mask, float(radius), origin, rgb_roi, series, min_coverage, strict=False)
    # The size of the markers. The usual estimate (the highest cluster of the distance transform's peaks with a fair
    # share of them) is tried first; where the markers sit on lines of their own colour - a fitted line, error bars -
    # the lines' ridges can outnumber them, so the other sizes the peaks suggest are tried too, the largest first,
    # and kept only if they give at least three compact markers all of a size.
    filled0 = fill_small_holes(mask, 200.0)
    usual = marker_radius(filled0)
    if usual:
        found = _markers_at(mask, usual, origin, rgb_roi, series, min_coverage, strict=False)
        if found:
            return found
    for r in _peak_scales(filled0)[:6]:
        if usual and abs(r - usual) <= 0.5:
            continue
        found = _markers_at(mask, r, origin, rgb_roi, series, min_coverage, strict=True)
        if found:
            return found
    return []


def _peak_scales(mask: np.ndarray) -> list[float]:
    """The radii the markers in *mask* may have, largest first: the heights of the distance transform's peaks, one
    per plateau (a line's ridge is one long plateau), where at least three peaks agree."""
    m = mask.astype(np.uint8)
    if m.sum() < 6:
        return []
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peaks = (dist >= cv2.dilate(dist, np.ones((5, 5), np.uint8))) & (dist >= 1.9)
    if not peaks.any():
        return []
    n, labels = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    cand = np.zeros(n - 1, dtype=np.float32)
    np.maximum.at(cand, labels[peaks] - 1, dist[peaks])
    out: list[float] = []
    for b in np.arange(np.floor(cand.max() * 2) / 2, 1.75, -0.5):
        near = cand[np.abs(cand - b) <= 0.75]
        if near.size >= 3:
            r = float(np.median(near))
            if all(abs(r - o) > 0.75 for o in out):
                out.append(r)
    return out


def _line_half_width(mask: np.ndarray) -> Optional[float]:
    """Half the width of the lines in *mask*: the distance transform along their middles, pixel by pixel (where the
    lines outnumber everything else)."""
    m = mask.astype(np.uint8)
    if m.sum() < 6:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist >= 0.9)
    return float(np.median(dist[ridge])) if ridge.any() else None


def _markers_at(
    mask: np.ndarray,
    r: float,
    origin: tuple[int, int],
    rgb_roi: Optional[np.ndarray],
    series: int,
    min_coverage: float,
    strict: bool,
) -> list[Marker]:
    """The markers of radius *r* in *mask* (see :func:`find_markers_in_mask`). *strict*: the size was a guess, so
    keep the markers only if there are at least three that stand alone, all of a size."""
    # A thin line drawn across a marker (a dashed threshold, a gridline on top) cuts it in two: close such gaps.
    closed = (
        cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ).astype(bool)
        if r >= 2.5
        else mask
    )
    filled = fill_small_holes(closed, max(30.0, 4.0 * np.pi * r * r))
    k = max(1, int(round(0.7 * r)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    dist_filled = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    disc = np.pi * r * r
    typical = areas[(areas >= 0.45 * disc) & (areas <= 2.2 * disc)]
    single = float(np.median(typical)) if typical.size else max(disc, float(np.median(areas)))
    h, w = mask.shape
    # Markers are compact: a blob that is long and thin, or sprawling, is a thick line, a bar or a block of colour.
    # A colour whose blobs are mostly not compact markers is not a colour of markers.
    singles = merged = 0
    keep: list[tuple[int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.3 * single:
            continue  # a scrap the opening left behind
        count = max(1, int(round(area / single)))
        compact = max(bw, bh) <= 2.5 * min(bw, bh) and area >= 0.4 * bw * bh
        if count == 1 and not compact:
            continue
        if count == 1:
            singles += 1
        else:
            merged += count
        keep.append((i, count))
    if singles == 0 or merged > 3 * singles:
        return []
    lone = np.array([float(stats[i, cv2.CC_STAT_AREA]) for i, c in keep if c == 1])
    alike = lone.size >= 3 and float(lone.std()) <= 0.35 * float(lone.mean())
    if strict and not alike:
        return []
    if min_coverage > 0:
        # Most of a colour of markers is in its markers; a colour of lines (or bars, or gridlines) that happens to
        # have a few thick places is mostly line. Unless the markers sit on lines of their own colour - a fitted
        # line, their error bars - and stand out from them: all of a size, and much thicker than the lines (where
        # lines cross they are a little thicker, not twice as thick). The dots of a dotted line stand alone.
        in_markers = sum(int(stats[i, cv2.CC_STAT_AREA]) for i, _ in keep)
        if in_markers < min_coverage * float(filled.sum()):
            r_line = _line_half_width(filled)
            joined = 0
            if alike and r_line and r >= 1.6 * r_line:
                _, flab, fstats, _ = cv2.connectedComponentsWithStats(filled.astype(np.uint8), connectivity=8)
                for i, c in keep:
                    if c != 1:
                        continue
                    ys, xs = np.nonzero(labels == i)
                    whole = fstats[int(np.bincount(flab[ys, xs]).argmax()), cv2.CC_STAT_AREA]
                    joined += int(whole >= 2.0 * stats[i, cv2.CC_STAT_AREA])
            if joined < max(3, 0.6 * lone.size):
                return []
    out: list[Marker] = []
    for i, count in keep:
        x, y, bw, bh, area = stats[i]
        ys, xs = np.nonzero(labels == i)
        if count == 1:
            centres = [(float(xs.mean()), float(ys.mean()))]
        else:
            # Overlapping markers: one centre at each peak of the distance from the blob's edge (a marker's middle
            # is the furthest point from its edge), keeping peaks at least most of a radius apart.
            inside = dist_filled[ys, xs]
            centres = []
            for j in np.argsort(-inside):
                if inside[j] < 0.85 * r:
                    break
                px, py = float(xs[j]), float(ys[j])
                if all((px - a) ** 2 + (py - b) ** 2 >= (0.9 * r) ** 2 for a, b in centres) and _disc_like(
                    filled, px, py, r
                ):
                    centres.append((px, py))
            if not centres:
                continue  # not markers: a thick knot of lines
            count = len(centres)
        colour = (0, 0, 0)
        if rgb_roi is not None:
            colour = tuple(int(v) for v in np.median(rgb_roi[ys, xs], axis=0))
        clipped = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        for cx, cy in centres:
            out.append(
                Marker(
                    x=cx + origin[0],
                    y=cy + origin[1],
                    area=float(area) / count,
                    radius=float(r),
                    rgb=colour,
                    series=series,
                    shared=count,
                    clipped=bool(clipped),
                )
            )
    # the same marker found twice (as a blob of its own and in a cluster): keep one
    kept: list[Marker] = []
    for m in sorted(out, key=lambda m: m.shared):
        if all((m.x - q.x) ** 2 + (m.y - q.y) ** 2 >= (0.75 * r) ** 2 for q in kept):
            kept.append(m)
    kept = _not_letters(mask, kept, origin, r)
    kept.sort(key=lambda m: (m.x, m.y))
    return kept


def _not_letters(mask: np.ndarray, markers: list[Marker], origin: tuple[int, int], r: float) -> list[Marker]:
    """Leave out "markers" that are letters: a round letter (o, e, a, filled in) of a word written in the plot, or the
    marker of a legend with its words beside it, has other glyphs of its colour all round it - pieces about its size
    that are not markers, and not joined to it. A marker on a line, or with error bars, is joined to them."""
    if not markers:
        return markers
    ox, oy = origin
    h, w = mask.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return markers
    d = 2.0 * r
    holding = set()
    for m in markers:
        y, x = int(round(m.y - oy)), int(round(m.x - ox))
        if 0 <= y < h and 0 <= x < w:
            holding.add(int(labels[y, x]))
    glyphs = []
    for i in range(1, n):
        if i in holding:
            continue
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= 6 and 0.4 * d <= max(bw, bh) <= 2.5 * d:
            glyphs.append(cents[i])
    if len(glyphs) < 2:
        return markers
    g = np.array(glyphs)
    out = []
    for m in markers:
        near = np.hypot(g[:, 0] - (m.x - ox), g[:, 1] - (m.y - oy)) <= 2.5 * d
        if int(near.sum()) < 2:
            out.append(m)
    return out


_RAYS = [(float(np.cos(a)), float(np.sin(a))) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]


def _disc_like(filled: np.ndarray, x: float, y: float, r: float) -> bool:
    """Is (x, y) the middle of a marker of radius *r*: in most directions the blob ends about a radius away? (Along a
    thick bundle of lines it runs on in two directions and ends much nearer in the others.)"""
    h, w = filled.shape
    good = 0
    for dx, dy in _RAYS:
        d = 0.0
        while d < 2.5 * r:
            px, py = int(round(x + dx * (d + 0.5))), int(round(y + dy * (d + 0.5)))
            if not (0 <= px < w and 0 <= py < h) or not filled[py, px]:
                break
            d += 0.5
        if 0.6 * r <= d <= 1.6 * r:
            good += 1
    return good >= 9


def colour_mask(rgb_roi: np.ndarray, colour, tolerance: float = 60.0) -> np.ndarray:
    return colour_distance(rgb_roi, colour) <= tolerance


def find_markers(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    colours: list,
    tolerance: float = 60.0,
    exclude=None,
    radius: Optional[float] = None,
    min_coverage: float = 0.0,
) -> list[Marker]:
    """The markers of each colour in *colours* inside *box* (x0, y0, x1, y1, inclusive) of the image."""
    x0, y0, x1, y1 = box
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    away = exclusion_mask(roi.shape[:2], exclude, origin=(x0, y0))
    out: list[Marker] = []
    for s, colour in enumerate(colours, start=1):
        m = colour_mask(roi, colour, tolerance) & ~away
        out.extend(
            find_markers_in_mask(m, origin=(x0, y0), radius=radius, rgb_roi=roi, series=s, min_coverage=min_coverage)
        )
    return out


# --------------------------------------------------------------------- error bars


def _ink(rgb: np.ndarray, box: tuple[int, int, int, int], threshold: float = 28.0) -> np.ndarray:
    """Pixels of *box* that are not the paper."""
    x0, y0, x1, y1 = box
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    return colour_distance(roi, background_colour(rgb, box)) > threshold


def _run(line: np.ndarray, clear: np.ndarray, start: int, step: int, skip: int, gap: int = 1) -> int:
    """From *start*, how far along *line* (step +1 or -1) the ink runs, after skipping *skip* pixels (the marker
    itself) and allowing gaps of up to *gap* pixels. Pixels where *clear* is True (the markers' own colour: a line
    joining them, crossing the bar) neither break the run nor extend it. Returns the index of the last ink pixel
    (*start* if there is none)."""
    n = len(line)
    i = start + step * skip
    last = None
    missing = 0
    while 0 <= i < n:
        if line[i]:
            last = i
            missing = 0
        elif not clear[i]:
            missing += 1
            if last is not None and missing > gap:
                break
            if last is None and missing > gap + 1:
                break  # a bar starts at the marker's edge
        i += step
    return last if last is not None else start


def _is_cap(ink: np.ndarray, along_row: bool, at: int, centre: int) -> bool:
    """Is there a cap - a short line across the bar, and nothing beyond it - at index *at* of the bar that runs
    through *centre*? A gridline crossing the bar runs on beyond where a cap would end."""
    h, w = ink.shape
    for d in (0, -1, 1):
        k = at + d
        if along_row:
            if not 0 <= k < w:
                continue
            line, n = ink[:, k], h
        else:
            if not 0 <= k < h:
                continue
            line, n = ink[k, :], w
        a = b = 0
        while centre - a - 1 >= 0 and line[centre - a - 1]:
            a += 1
        while centre + b + 1 < n and line[centre + b + 1]:
            b += 1
        if not (2 <= a <= 12 and 2 <= b <= 12):
            continue
        reach = max(a, b)
        beyond = np.concatenate(
            [
                line[max(0, centre - reach - 14) : max(0, centre - reach - 3)],
                line[centre + reach + 3 : centre + reach + 14],
            ]
        )
        if beyond.size == 0 or beyond.mean() < 0.3:
            return True
    return False


def _bar_end(ink, clear, along_row: bool, centre_line: int, start: int, step: int, skip: int) -> Optional[float]:
    """Where the error bar from a marker ends on one side: the nearest end over the rows (or columns) the bar is
    drawn on - a gridline under one of them carries that one on - cut at the first cap; None if there is no bar."""
    size = ink.shape[0] if along_row else ink.shape[1]
    ends = []
    for d in (0, -1, 1):
        k = min(max(centre_line + d, 0), size - 1)
        line = ink[k, :] if along_row else ink[:, k]
        free = clear[k, :] if along_row else clear[:, k]
        e = _run(line, free, start, step, skip)
        if abs(e - start) > skip + 1:
            ends.append(e)
    if not ends:
        return None
    end = min(ends, key=lambda e: abs(e - start))
    # a cap at the end: the bar ends at its middle
    caps = [
        i
        for i in range(end - 3, end + 4)
        if abs(i - start) > skip
        and 0 <= i < (ink.shape[1] if along_row else ink.shape[0])
        and _is_cap(ink, along_row, i, centre_line)
    ]
    if caps:
        near = min(caps, key=lambda i: abs(i - end))
        group = [near]
        while group[-1] + 1 in caps:
            group.append(group[-1] + 1)
        while group[0] - 1 in caps:
            group.insert(0, group[0] - 1)
        return 0.5 * (group[0] + group[-1])
    return float(end)


def _own_colour_bar(cmask: np.ndarray, cx: int, cy: int, r: float, step: int) -> Optional[float]:
    """Where an error bar of the marker's own colour ends on one side (*step* -1: above, +1: below), or None: a thin
    straight line of the colour running on from the marker's edge, centred on the marker's column, usually ending in
    a cap (a short line across it). A line joining the markers, or a fitted line through them, slants away from the
    marker's column, or is wide across it; where such a line crosses the bar, the bar runs on beyond it."""
    h, w = cmask.shape
    thin = max(3.0, 1.0 * r)

    def extent(rr: int, c: int) -> tuple[int, int]:
        a = b = c
        while a - 1 >= 0 and cmask[rr, a - 1]:
            a -= 1
        while b + 1 < w and cmask[rr, b + 1]:
            b += 1
        return a, b

    start = int(round(cy + step * (r + 1)))
    if not 0 <= start < h:
        return None
    c = next((cx + d for d in (0, -1, 1) if 0 <= cx + d < w and cmask[start, cx + d]), None)
    if c is None:
        return None
    i = start
    while 0 <= i + step < h and any(0 <= c + d < w and cmask[i + step, c + d] for d in (0, -1, 1)):
        i += step
        if not cmask[i, c]:
            c = next(c + d for d in (-1, 1) if 0 <= c + d < w and cmask[i, c + d])
    rows = list(range(start, i + step, step))
    spans = [extent(rr, c if cmask[rr, c] else next((c + d for d in (-1, 1) if 0 <= c + d < w and cmask[rr, c + d]), c))
             for rr in rows]
    widths = [b - a + 1 for a, b in spans]

    def is_thin(k: int) -> bool:
        return widths[k] <= thin and abs(0.5 * (spans[k][0] + spans[k][1]) - cx) <= 1.5 + 0.25 * widths[k]

    k = 0
    while k < len(rows) and is_thin(k):
        k += 1
    if k < 3:
        return None
    line_w = float(np.median([widths[j] for j in range(k)]))
    while True:
        # past the thin line: a cap (the end), or a line crossing the bar (the bar goes on beyond it), or something
        # else it runs into (not a bar's end that can be told)
        j = k
        while j < len(rows) and not is_thin(j):
            j += 1
        stretch = list(range(k, j))
        if not stretch:
            return float(rows[-1])
        resumes = j < len(rows) and all(is_thin(q) for q in range(j, min(len(rows), j + 3))) and len(rows) - j >= 3
        capped = all(widths[q] >= 2.0 * line_w and widths[q] <= 6.0 * r + 4
                     and abs(0.5 * (spans[q][0] + spans[q][1]) - cx) <= 0.5 * widths[q] for q in stretch)
        if resumes and len(stretch) <= max(4 * line_w, 3 * r):
            k = j  # a line crossing the bar
            while k < len(rows) and is_thin(k):
                k += 1
            continue
        if capped and j == len(rows):
            return 0.5 * (rows[stretch[0]] + rows[stretch[-1]])
        if capped and len(stretch) <= max(4 * line_w, 2 * r):
            return 0.5 * (rows[stretch[0]] + rows[stretch[-1]])
        return None


def find_error_bars(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    markers: list[Marker],
    directions: str = "auto",
    marker_colours: Optional[list] = None,
    tolerance: float = 60.0,
) -> None:
    """Follow the error bar from each marker and record its ends in ``marker.errors`` (image pixels).

    *directions*: ``"auto"`` (whichever bars are there), ``"vertical"``, ``"horizontal"``, ``"both"`` or
    ``"none"``. An error bar is ink of any colour except the markers' own that runs unbroken from the marker; a run
    that ends within a pixel or two of the marker is not a bar. Dashed gridlines through a marker stop at their
    first gap, so they are not taken for bars.
    """
    if directions == "none" or not markers:
        return
    x0, y0, x1, y1 = box
    ink = _ink(rgb, box)
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    clear = np.zeros(ink.shape, dtype=bool)
    for c in marker_colours or []:
        clear |= cv2.dilate(colour_mask(roi, c, tolerance).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ink &= ~clear
    want_v = directions in ("auto", "vertical", "both")
    want_h = directions in ("auto", "horizontal", "both")
    own = [colour_mask(roi, c, tolerance) for c in marker_colours] if marker_colours else None
    for m in markers:
        cx, cy = int(round(m.x)) - x0, int(round(m.y)) - y0
        if not (0 <= cx < ink.shape[1] and 0 <= cy < ink.shape[0]):
            continue
        skip = int(np.ceil(m.radius)) + 1
        found = {}
        if want_h:
            left = _bar_end(ink, clear, True, cy, cx, -1, skip)
            right = _bar_end(ink, clear, True, cy, cx, +1, skip)
            if left is not None:
                found["left"] = float(left + x0)
            if right is not None:
                found["right"] = float(right + x0)
            if left is not None or right is not None:
                found["capped_h"] = bool(
                    (left is None or _is_cap(ink, True, int(round(left)), cy))
                    and (right is None or _is_cap(ink, True, int(round(right)), cy))
                )
        if want_v:
            top = _bar_end(ink, clear, False, cx, cy, -1, skip)
            bottom = _bar_end(ink, clear, False, cx, cy, +1, skip)
            if top is None and bottom is None and own is not None:
                # error bars of the marker's own colour (the default in MATLAB, Prism and matplotlib)
                c_own = own[m.series - 1] if m.series - 1 < len(own) else None
                if c_own is not None:
                    top = _own_colour_bar(c_own, cx, cy, m.radius, -1)
                    bottom = _own_colour_bar(c_own, cx, cy, m.radius, +1)
            if top is not None:
                found["up"] = float(top + y0)
            if bottom is not None:
                found["down"] = float(bottom + y0)
            if top is not None or bottom is not None:
                found["capped_v"] = bool(
                    (top is None or _is_cap(ink, False, int(round(top)), cx))
                    and (bottom is None or _is_cap(ink, False, int(round(bottom)), cx))
                )
        m.errors = {k: v for k, v in found.items() if v is not None}
    if directions == "auto":
        # Bars belong to a whole series: a direction that only a few of its markers seem to have is something else
        # (the bars of the points above and below, a gridline, a neighbour's marker).
        for s in {m.series for m in markers}:
            group = [m for m in markers if m.series == s]
            for keys in (("left", "right", "capped_h"), ("up", "down", "capped_v")):
                have = sum(1 for m in group if keys[0] in m.errors or keys[1] in m.errors)
                if have < max(2, 0.5 * len(group)):
                    for m in group:
                        for k in keys:
                            m.errors.pop(k, None)
