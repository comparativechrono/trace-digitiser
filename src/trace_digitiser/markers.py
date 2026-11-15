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
    deep: Optional[np.ndarray] = None,
) -> list[Marker]:
    """The markers in a boolean *mask* (one colour), whose top-left pixel is *origin* (x, y) in the image.

    *min_coverage*: the share of the colour's pixels that must be in markers for it to count as a colour of markers
    at all (used when the colours were not given, but found). *deep*: where translucent markers lie two or more deep
    (:func:`layer_map`).
    """
    if mask.sum() == 0:
        return []
    if radius:
        return _markers_at(mask, float(radius), origin, rgb_roi, series, min_coverage, strict=False, deep=deep)
    # The size of the markers. The usual estimate (the highest cluster of the distance transform's peaks with a fair
    # share of them) is tried first; where the markers sit on lines of their own colour - a fitted line, error bars -
    # the lines' ridges can outnumber them, so the other sizes the peaks suggest are tried too, the largest first,
    # and kept only if they give at least three compact markers all of a size.
    filled0 = fill_small_holes(mask, 200.0)
    usual = marker_radius(filled0)
    if usual:
        found = _markers_at(mask, usual, origin, rgb_roi, series, min_coverage, strict=False, deep=deep)
        if found:
            return found
    for r in _peak_scales(filled0)[:6]:
        if usual and abs(r - usual) <= 0.5:
            continue
        found = _markers_at(mask, r, origin, rgb_roi, series, min_coverage, strict=True, deep=deep)
        if found:
            return found
    if usual and min_coverage <= 0:
        # The colour was given, so it has markers: bigger than the size guessed, perhaps (a few markers on a thick
        # line of their colour, whose ridge set the guess). Round lumps much bigger than that guess, all of a
        # size, are the markers.
        for r in [_lump_radius(filled0, usual), usual]:
            if r:
                found = _markers_at(
                    mask, r, origin, rgb_roi, series, min_coverage, strict=False, deep=deep, lenient=True
                )
                if found:
                    return found
    return []


def _lump_radius(filled: np.ndarray, r: float) -> Optional[float]:
    """The radius of the round lumps in *filled* well over twice the area of a disc of radius *r*, once lines as thick
    as that are opened away: None if there are none, or they are not of a size."""
    k = max(1, int(round(0.7 * r)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    sizes = []
    for i in range(1, n):
        _, _, bw, bh, area = stats[i]
        if area >= 2.5 * np.pi * r * r and max(bw, bh) <= 1.35 * min(bw, bh) and area >= 0.6 * bw * bh:
            sizes.append(float(np.sqrt(area / np.pi)))
    if not sizes:
        return None
    med = float(np.median(sizes))
    if any(abs(v - med) > 0.25 * med for v in sizes):
        return None
    return med + 0.5


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


def layer_map(rgb_roi: np.ndarray, colour, mask: np.ndarray, background) -> Optional[np.ndarray]:
    """Where markers are translucent, which pixels of *mask* lie under two or more of them (True) and which under one.

    A pixel under two translucent markers is darker than one under a single marker, so where they overlap, the edge
    of each shows across the other. The pixels of a colour fall along the line from the paper to the colour; drawn
    translucent and overlapping, they bunch at two places along it - one marker deep, and two or more - with few
    between. None if they do not (opaque markers, or none overlapping).
    """
    if rgb_roi is None or mask.sum() < 60:
        return None
    bg = np.asarray(background, dtype=np.float32)
    v = np.asarray(colour, dtype=np.float32) - bg
    vv = float(v @ v)
    if vv < 900.0:  # the colour is too near the paper to tell layers apart
        return None
    t = ((rgb_roi.astype(np.float32) - bg) @ v) / vv
    # Away from the markers' edges only: an edge is paler (and, in a JPEG, a coloured edge is blurred two pixels or
    # more), which would pass for a layer of its own.
    inner = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1] >= 3.0
    tm = t[inner]
    if tm.size < 60:
        return None
    hist, edges = np.histogram(tm, bins=60, range=(0.0, 1.8))
    hist = np.convolve(hist.astype(float), [0.25, 0.5, 0.25], mode="same")
    peaks = [i for i in range(1, 59) if hist[i] > 0 and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]]
    peaks = [i for i in peaks if hist[i] >= 0.03 * tm.size / 4.0 and edges[i] >= 0.45]
    if len(peaks) < 2:
        return None
    best = None
    for a in peaks:
        for b in peaks:
            if b - a < 4:  # at least 0.12 apart along the line
                continue
            valley = int(a + np.argmin(hist[a : b + 1]))
            if hist[valley] > 0.6 * min(hist[a], hist[b]):
                continue
            score = min(hist[a], hist[b])
            if best is None or score > best[0]:
                best = (score, valley)
    if best is None:
        return None
    deep = mask & (t >= edges[best[1]] + 0.015)
    if deep.sum() < 0.02 * mask.sum():
        return None
    # the pale edge of a marker over another is not a layer less: keep the deep part solid
    deep = cv2.morphologyEx(deep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    return deep


def _boundary_radius(blob: np.ndarray) -> Optional[float]:
    """How far the pixels on the edge of a lone marker (*blob*, a crop) lie from its centre, on average."""
    ys, xs = np.nonzero(blob)
    if ys.size < 8:
        return None
    inner = cv2.erode(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    by, bx = np.nonzero(blob & ~inner)
    if by.size < 6:
        return None
    return float(np.mean(np.hypot(bx - xs.mean(), by - ys.mean())))


def _doubled(deep: Optional[np.ndarray], centre: tuple[float, float], rb: float) -> bool:
    """Are two translucent markers drawn on the same spot here: is the middle of the marker dark all through?"""
    if deep is None:
        return False
    x, y = centre
    k = max(1.0, rb - 1.5)
    a0, a1 = max(0, int(np.floor(y - k))), min(deep.shape[0], int(np.ceil(y + k)) + 1)
    b0, b1 = max(0, int(np.floor(x - k))), min(deep.shape[1], int(np.ceil(x + k)) + 1)
    if a1 <= a0 or b1 <= b0:
        return False
    yy, xx = np.mgrid[a0:a1, b0:b1]
    core = (xx - x) ** 2 + (yy - y) ** 2 <= k * k
    return bool(core.sum() >= 4 and deep[a0:a1, b0:b1][core].mean() >= 0.85)


def _disc(radius: float) -> np.ndarray:
    k = int(np.ceil(radius))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    return (xx * xx + yy * yy <= radius * radius).astype(np.float32)




def split_overlapping(
    blob: np.ndarray, rb: float, deep: Optional[np.ndarray] = None, solid: Optional[np.ndarray] = None
) -> list[tuple[float, float, float]]:
    """The centres of the round markers whose union is *blob* (a boolean crop, holding one blob of overlapping
    markers): a list of (x, y, support), *support* being the share of each marker's outline that is seen.

    *rb* is the distance of a lone marker's edge pixels from its centre. *deep*: where translucent markers lie two or
    more deep (:func:`layer_map`), so that their edges inside the blob are seen too. *solid*: the marker's colour as
    drawn, before gaps were closed and holes filled; a marker is solid colour, where lines that only run close
    together leave gaps.

    Each pixel on an edge - of the blob, or between one layer of markers and two - lies about *rb* from the centre of
    a marker, inwards, square to the edge: each casts a vote there, and the centres are where the votes pile up (a
    Hough transform for circles of one size). Markers that are mostly hidden are then added where the blob is not yet
    covered, and any that cover nothing of their own are dropped.
    """
    h, w = blob.shape
    if rb < 1.5 or blob.sum() < 4:
        return []
    layers = blob.astype(np.float32)
    if deep is not None:
        layers += (deep & blob).astype(np.float32)
    smooth = cv2.GaussianBlur(layers, (0, 0), 0.9)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    li = layers.astype(np.int8)
    edge = np.zeros((h, w), dtype=bool)
    edge[:, 1:] |= li[:, 1:] != li[:, :-1]
    edge[:, :-1] |= li[:, :-1] != li[:, 1:]
    edge[1:, :] |= li[1:, :] != li[:-1, :]
    edge[:-1, :] |= li[:-1, :] != li[1:, :]
    # the blob's outline is taken on its inside; an edge between layers on both sides
    edge &= blob & (mag > 1e-3)
    edge[0, :] = edge[-1, :] = False  # the edge of the crop is not the edge of a marker
    edge[:, 0] = edge[:, -1] = False
    ys, xs = np.nonzero(edge)
    if ys.size == 0:
        return []
    nx, ny = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]
    cx, cy = xs + rb * nx, ys + rb * ny
    acc = np.zeros((h, w), dtype=np.float32)
    # with each vote, the direction it came from (doubled, so that opposite directions add): the votes for a marker
    # come from round its outline, those along the middle of a thick straight line from its two sides only
    c2, s2 = np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
    cos2, sin2 = nx * nx - ny * ny, 2.0 * nx * ny
    x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
    fx, fy = cx - x0, cy - y0
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = x0 + dx, y0 + dy
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        np.add.at(acc, (yy[ok], xx[ok]), wt[ok])
        np.add.at(c2, (yy[ok], xx[ok]), (wt * cos2)[ok])
        np.add.at(s2, (yy[ok], xx[ok]), (wt * sin2)[ok])
    acc = cv2.GaussianBlur(acc, (0, 0), 0.7)
    box3 = cv2.boxFilter(acc, -1, (3, 3), normalize=False)
    support = box3 / (2.0 * np.pi * rb)
    spread = np.hypot(
        cv2.boxFilter(cv2.GaussianBlur(c2, (0, 0), 0.7), -1, (3, 3), normalize=False),
        cv2.boxFilter(cv2.GaussianBlur(s2, (0, 0), 0.7), -1, (3, 3), normalize=False),
    ) / np.maximum(box3, 1e-6)
    peak = (acc >= cv2.dilate(acc, np.ones((3, 3), np.uint8))) & (support >= 0.2) & (spread <= 0.95)
    # a marker lies wholly inside the blob: most of the disc round a centre must be blob
    inside = cv2.distanceTransform(np.pad(blob, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    fits = inside >= max(1.0, rb - 0.9)
    if solid is not None:
        core = _disc(max(1.0, rb - 0.5))
        fill = cv2.filter2D(solid.astype(np.float32), -1, core, borderType=cv2.BORDER_CONSTANT) / float(core.sum())
        fits &= fill >= 0.85
    peak &= fits
    py, px = np.nonzero(peak)
    order = np.argsort(-support[py, px])
    d_min = max(1.5, 0.3 * rb)
    chosen: list[list[float]] = []
    for j in order:
        x, y = int(px[j]), int(py[j])
        # the peak's position to a fraction of a pixel: the centroid of the votes round it
        a0, a1, b0, b1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
        patch = acc[a0:a1, b0:b1]
        s = float(patch.sum())
        if s <= 0:
            continue
        gy_, gx_ = np.mgrid[a0:a1, b0:b1]
        fx_, fy_ = float((patch * gx_).sum() / s), float((patch * gy_).sum() / s)
        if all((fx_ - c[0]) ** 2 + (fy_ - c[1]) ** 2 >= d_min * d_min for c in chosen):
            chosen.append([fx_, fy_, float(support[y, x])])
    # markers hidden all but a sliver of their outline: wherever the blob is not yet covered
    cover_r = rb + 0.8
    kern = _disc(cover_r)
    area = np.pi * rb * rb
    yy, xx = np.mgrid[0:h, 0:w]

    def covered(centres) -> np.ndarray:
        out = np.zeros((h, w), dtype=bool)
        for c in centres:
            out |= (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= cover_r * cover_r
        return out

    # Only beside markers already found, and no more of them than were found: a stretch of blob that no marker's
    # outline explains is more likely thick line than a row of hidden markers.
    seen = len(chosen)
    refused = np.zeros((h, w), dtype=bool)
    for _ in range(min(seen, int(blob.sum() / max(area, 1.0)) + 1)):
        cover = covered(chosen)
        bare = blob & ~cover
        bare = cv2.morphologyEx(bare.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        if bare.sum() < 0.25 * area:
            break
        gain = cv2.filter2D(bare.astype(np.float32), -1, kern, borderType=cv2.BORDER_CONSTANT)
        beside = cv2.filter2D(cover.astype(np.float32), -1, kern, borderType=cv2.BORDER_CONSTANT)
        gain[~fits | refused | (beside < 0.25 * area)] = 0
        placed = False
        for _try in range(4):
            if float(gain.max()) < 0.3 * area:
                break
            y, x = np.unravel_index(int(np.argmax(gain)), gain.shape)
            # a hidden marker accounts for the piece of blob it uncovers; a disc laid on a long piece does not
            _, bare_labels = cv2.connectedComponents(bare.astype(np.uint8), connectivity=8)
            inside_disc = (xx - x) ** 2 + (yy - y) ** 2 <= cover_r * cover_r
            ids = np.unique(bare_labels[bare & inside_disc])
            piece = np.isin(bare_labels, ids[ids > 0])
            if (piece & inside_disc).sum() >= 0.7 * piece.sum():
                chosen.append([float(x), float(y), 0.0])
                placed = True
                break
            near = (xx - x) ** 2 + (yy - y) ** 2 <= 4.0
            refused |= near
            gain[near] = 0
        if not placed:
            break
    # drop those that cover nothing the others do not
    for c in sorted(chosen, key=lambda c: c[2]):
        others = [o for o in chosen if o is not c]
        if not others:
            break
        alone = blob & covered([c]) & ~covered(others)
        if alone.sum() < 0.06 * area and c[2] < 0.35:
            chosen = others
    return [(c[0], c[1], c[2]) for c in chosen]


def _markers_at(
    mask: np.ndarray,
    r: float,
    origin: tuple[int, int],
    rgb_roi: Optional[np.ndarray],
    series: int,
    min_coverage: float,
    strict: bool,
    deep: Optional[np.ndarray] = None,
    lenient: bool = False,
) -> list[Marker]:
    """The markers of radius *r* in *mask* (see :func:`find_markers_in_mask`). *strict*: the size was a guess, so
    keep the markers only if there are at least three that stand alone, all of a size. *deep*: where translucent
    markers lie two or more deep (:func:`layer_map`). *lenient*: the colour is known to be one of markers, so where
    most of it is in lumps that are not markers, keep the markers there are rather than none."""
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
    # Where most markers overlap others, the typical blob is two or three of them: go by the round blobs of about a
    # marker's size, which are markers on their own (a blob fills about 0.785 of its box when it is round).
    side = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]).astype(float)
    low = np.minimum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]).astype(float)
    round_ = (areas >= 0.7 * disc) & (areas <= 1.6 * disc) & (side <= 1.15 * low) & (areas >= 0.68 * side * side)
    if round_.sum() >= 2 and float(np.median(areas[round_])) < 0.85 * single:
        single = float(np.median(areas[round_]))
    h, w = mask.shape
    # Markers are compact: a blob that is long and thin, or sprawling, is a thick line, a bar or a block of colour.
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
        if count <= 50:  # a lump of more than fifty markers' area is a region of colour, not markers
            keep.append((i, count))
    if singles == 0 and not lenient:
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
    # How far a lone marker's edge pixels lie from its centre: the distance at which the edges of overlapping
    # markers are looked for.
    radii = []
    for i, count in keep:
        if count == 1 and len(radii) < 60:
            x, y, bw, bh, _ = stats[i]
            rb_i = _boundary_radius(labels[y : y + bh, x : x + bw] == i)
            if rb_i:
                radii.append(rb_i)
    rb = float(np.median(radii)) if radii else max(1.5, r - 0.9)
    rb = float(np.clip(rb, max(1.5, 0.7 * r - 0.5), 1.25 * r))  # lone "markers" with stubs of line attached
    if deep is not None and radii:
        # A darker rim round every lone marker is an outline, not a second layer of translucent markers. (A lone
        # marker dark all through is two drawn on the same spot.)
        rims = []
        for i, count in keep:
            if count == 1:
                ys, xs = np.nonzero(labels == i)
                d = deep[ys, xs]
                if d.any():
                    out_ = np.hypot(xs[d] - xs.mean(), ys[d] - ys.mean()) >= 0.6 * rb
                    rims.append(float(out_.mean()) if d.mean() > 0.12 else 0.0)
                else:
                    rims.append(0.0)
        if rims and float(np.median(rims)) > 0.7:
            deep = None
    # The centres in each blob: one for a lone marker; for overlapping ones, those that the edges seen point to (see
    # split_overlapping).
    placed: list[tuple[int, list[tuple[float, float]]]] = []
    cluster_area = explained_area = 0.0
    supports: list[float] = []
    unexplained: set[int] = set()
    plain: dict[int, list[tuple[float, float]]] = {}
    for i, count in keep:
        x, y, bw, bh, area = stats[i]
        pad = int(np.ceil(rb)) + 2
        a0, a1 = max(0, y - pad), min(h, y + bh + pad)
        b0, b1 = max(0, x - pad), min(w, x + bw + pad)
        crop = labels[a0:a1, b0:b1] == i
        sub_deep = None if deep is None else deep[a0:a1, b0:b1]
        if count == 1:
            ys, xs = np.nonzero(crop)
            centres = [(float(xs.mean()) + b0, float(ys.mean()) + a0)]
            # two markers almost on top of each other: a drawn-out blob whose edges point to two centres (drawn out
            # along any direction: its long axis over its short one)
            cov = np.cov(np.vstack([xs, ys]).astype(float)) if len(xs) > 2 else np.eye(2)
            ev = np.linalg.eigvalsh(cov)
            drawn_out = ev[0] > 0 and ev[1] > 1.5 * ev[0]
            if (max(bw, bh) > 1.25 * min(bw, bh) and area >= 0.9 * single) or (drawn_out and area >= 1.15 * single):
                found = split_overlapping(crop, rb, sub_deep, solid=mask[a0:a1, b0:b1])
                if len(found) >= 2 and min(f[2] for f in found) >= 0.35:
                    centres = [(fx + b0, fy + a0) for fx, fy, _ in found]
            if len(centres) == 1 and _doubled(deep, centres[0], rb):
                centres = centres * 2
            placed.append((i, centres))
            continue
        found = split_overlapping(crop, rb, sub_deep, solid=mask[a0:a1, b0:b1])
        share = 0.0
        if found:
            yy, xx = np.mgrid[a0:a1, b0:b1]
            union = np.zeros(crop.shape, dtype=bool)
            for fx, fy, _ in found:
                union |= (xx - b0 - fx) ** 2 + (yy - a0 - fy) ** 2 <= (rb + 1.0) ** 2
            share = float((union & crop).sum()) / float(crop.sum())
            centres = [(fx + b0, fy + a0) for fx, fy, _ in found]
            for c in list(centres):
                if _doubled(deep, c, rb) and all(
                    (c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 >= (0.5 * rb) ** 2 for o in centres if o is not c
                ):
                    centres.append(c)
        else:
            # no edges to go by: the peaks of the distance from the blob's edge, kept apart by most of a radius
            sub = np.where(crop, dist_filled[a0:a1, b0:b1], 0.0).astype(np.float32)
            tops = crop & (sub >= cv2.dilate(sub, np.ones((3, 3), np.uint8))) & (sub >= 0.85 * r)
            ys, xs = np.nonzero(tops)
            ys, xs = ys + a0, xs + b0
            inside = dist_filled[ys, xs]
            centres = []
            for j in np.argsort(-inside):
                px, py = float(xs[j]), float(ys[j])
                if all((px - a) ** 2 + (py - b) ** 2 >= (0.9 * r) ** 2 for a, b in centres) and _disc_like(
                    filled, px, py, r
                ):
                    centres.append((px, py))
        cluster_area += float(area)
        explained_area += share * float(area)
        supports.extend(f[2] for f in found if f[2] > 0)
        if share < 0.75:
            unexplained.add(i)
            # markers plainly seen among what is not markers (on the thick line of a fit, with their error bars)
            plain[i] = [(fx + b0, fy + a0) for fx, fy, sup in found if sup >= 0.4]
        if centres:
            placed.append((i, centres))
    # A colour whose blobs are mostly lumps bigger than one marker is not a colour of markers - unless the lumps are
    # markers seen overlapping: the outlines of round markers, in whose union most of each lump lies. (Nor are its
    # lumps markers, if the colour was given.)
    if merged > 3 * singles:
        seen = (
            cluster_area > 0
            and explained_area >= 0.7 * cluster_area
            and len(supports) >= 3
            and float(np.median(supports)) >= 0.33
        )
        if not seen:
            if not lenient:
                return []
            # the colour was given, and no other size of marker does better: keep the markers that stand alone, and
            # those seen making up a lump, and leave the lumps that are something else (a thick line fitted through
            # the markers, say)
            placed = [(i, plain.get(i, []) if i in unexplained else c) for i, c in placed]
            placed = [(i, c) for i, c in placed if c]
            if not placed:
                return []
    out: list[Marker] = []
    for i, centres in placed:
        x, y, bw, bh, area = stats[i]
        ys, xs = np.nonzero(labels == i)
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
    kept = _not_letters(mask, out, origin, r, rgb_roi)
    kept.sort(key=lambda m: (m.x, m.y))
    return kept


def _not_letters(
    mask: np.ndarray,
    markers: list[Marker],
    origin: tuple[int, int],
    r: float,
    rgb_roi: Optional[np.ndarray] = None,
) -> list[Marker]:
    """Leave out "markers" that are letters: a round letter (o, e, a, filled in) of a word written in the plot, or the
    marker of a legend with its words beside it, has other glyphs of its colour all round it - pieces about its size
    that are not markers, and not joined to it. A marker on a line, or with error bars, is joined to them. Glyphs are
    drawn with thin strokes, with paper round them: the pieces of markers that something drawn over them (a mean's bar,
    an error bar) has cut off are solid, or lie against what cut them, and do not count."""
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
    thick = max(1.6, 0.6 * r)
    ring_kernel = np.ones((9, 9), np.uint8)
    for i in range(1, n):
        if i in holding:
            continue
        x, y, bw, bh, area = stats[i]
        if area >= 6 and 0.4 * d <= max(bw, bh) <= 4.0 * d:
            piece = np.pad(labels[y : y + bh, x : x + bw] == i, 1).astype(np.uint8)
            if float(cv2.distanceTransform(piece, cv2.DIST_L2, 3).max()) > thick:
                continue
            if rgb_roi is not None:
                # what lies round it, a few pixels out (a JPEG blurs a coloured glyph's edge): one colour - the paper,
                # a shaded band - round a glyph; round a piece cut off a marker, partly what cut it
                a0, a1, b0, b1 = max(0, y - 4), min(h, y + bh + 4), max(0, x - 4), min(w, x + bw + 4)
                own = (labels[a0:a1, b0:b1] == i).astype(np.uint8)
                ring = (
                    cv2.dilate(own, ring_kernel).astype(bool)
                    & ~cv2.dilate(own, np.ones((3, 3), np.uint8)).astype(bool)
                    & ~mask[a0:a1, b0:b1]
                )
                if ring.sum() >= 4:
                    around = rgb_roi[a0:a1, b0:b1][ring].astype(np.float32)
                    usual = np.median(around, axis=0)
                    if float((np.sqrt(((around - usual) ** 2).sum(axis=1)) <= 40.0).mean()) < 0.6:
                        continue
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


def under_ink(rgb_roi: np.ndarray, colour, mask: np.ndarray, radius: Optional[float] = None) -> np.ndarray:
    """The pixels of dark ink drawn over markers of this colour: a mean's bar or an error bar across a column of dots,
    a black line through a marker. They are dark (black or grey), and the colour lies on both sides of them, above
    and below or left and right, within about a marker's width: the marker goes on under the ink. None for markers
    that are dark themselves, or when the markers' size cannot be told."""
    none = np.zeros(mask.shape, bool)
    c = np.asarray(colour, float).reshape(3)
    if c.mean() < 110 or c.max() - c.min() < 60 or mask.sum() < 6:
        return none  # dark or grey markers: ink drawn over them is not told from them by its darkness
    r = radius or marker_radius(fill_small_holes(mask, 200.0))
    if not r:
        return none
    rgb = rgb_roi.astype(float)
    dark = (rgb.mean(axis=2) < 110) & (rgb.max(axis=2) - rgb.min(axis=2) < 60) & ~mask
    if not dark.any():
        return none
    reach = max(3, int(round(1.2 * r)))
    m = mask.astype(np.int32)

    def beside(axis: int) -> np.ndarray:
        """Whether the colour lies within *reach* pixels both before and after each pixel along *axis*."""
        n = m.shape[axis]
        zero = np.zeros((1, m.shape[1]) if axis == 0 else (m.shape[0], 1), np.int32)
        total = np.concatenate([zero, np.cumsum(m, axis=axis)], axis=axis)  # total[i]: the sum of m[:i]
        i = np.arange(n)
        at = lambda k: np.take(total, np.clip(k, 0, n), axis=axis)  # noqa: E731
        before = at(i) - at(i - reach)
        after = at(i + 1 + reach) - at(i + 1)
        return (before > 0) & (after > 0)

    return dark & (beside(0) | beside(1))


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
    paper = background_colour(rgb, box)
    for s, colour in enumerate(colours, start=1):
        m = colour_mask(roi, colour, tolerance) & ~away
        deep = layer_map(roi, colour, m, paper)
        m = m | (under_ink(roi, colour, m, radius) & ~away)
        out.extend(
            find_markers_in_mask(
                m, origin=(x0, y0), radius=radius, rgb_roi=roi, series=s, min_coverage=min_coverage, deep=deep
            )
        )
    return out


# --------------------------------------------------------------------- error bars


def _ink(rgb: np.ndarray, box: tuple[int, int, int, int], threshold: float = 28.0) -> np.ndarray:
    """Pixels of *box* that are drawn: not the paper, nor a broad band of colour shading part of the plot (the paper,
    as far as a thin line drawn on it is concerned)."""
    x0, y0, x1, y1 = box
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    drawn = colour_distance(roi, background_colour(rgb, box)) > threshold
    # what lies under a thin line: the colour round it, once lines up to 15 pixels across are closed over
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    under = np.stack([cv2.morphologyEx(roi[..., c], cv2.MORPH_CLOSE, k) for c in range(3)], axis=2)
    lighter = np.stack([cv2.morphologyEx(roi[..., c], cv2.MORPH_OPEN, k) for c in range(3)], axis=2)
    stands_out = (np.sqrt(((roi.astype(np.float32) - under) ** 2).sum(axis=2)) > threshold) | (
        np.sqrt(((roi.astype(np.float32) - lighter) ** 2).sum(axis=2)) > threshold
    )
    return drawn & stands_out


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
    # no cap: a gridline broken (by a scan, by dashes) runs on beyond the gap, as a bar does not
    k = min(max(centre_line, 0), size - 1)
    line = ink[max(0, k - 1) : k + 2, :].any(axis=0) if along_row else ink[:, max(0, k - 1) : k + 2].any(axis=1)
    beyond = line[end + step * 3 :: step] if step > 0 else line[: max(0, end - 2)][::-1]
    if beyond.size >= 20 and float(beyond.mean()) >= 0.6:
        return None
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


def _runs_on(drawn: np.ndarray, along_row: bool, centre: int, end: float, step: int) -> bool:
    """Does a line run on, aslant, past where a bar seems to end - a line leaving the marker steeply (the line joining
    it to the next), not an error bar, which ends there (or in a cap across it)?"""
    h, w = drawn.shape
    e = int(round(end))
    hits = 0
    for d in range(1, 5):
        i = e + step * d
        if along_row:
            if not 0 <= i < w:
                return False
            line, n = drawn[:, i], h
        else:
            if not 0 <= i < h:
                return False
            line, n = drawn[i, :], w
        side = [k for k in range(centre - 6, centre + 7) if 0 <= k < n and abs(k - centre) >= 2 and line[k]]
        straight = any(line[k] for k in range(max(0, centre - 1), min(n, centre + 2)))
        if side and not straight:
            hits += 1
    return hits >= 3


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
            w_box = ink.shape[1]
            drawn_h = ink | (own[m.series - 1] if own is not None and m.series - 1 < len(own) else False)
            if left is not None and _runs_on(drawn_h, True, cy, left, -1):
                left = None
            if right is not None and _runs_on(drawn_h, True, cy, right, +1):
                right = None
            if left is not None and left <= 3 and not _is_cap(ink, True, int(round(left)), cy):
                left = None
            if right is not None and right >= w_box - 4 and not _is_cap(ink, True, int(round(right)), cy):
                right = None
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
            # a line through the marker that runs on to the edge of the plot, with no cap, is a gridline or the
            # edge of a shaded band, not an error bar; one that runs on aslant is a line leaving the marker
            h_box = ink.shape[0]
            drawn = ink | (own[m.series - 1] if own is not None and m.series - 1 < len(own) else False)
            if top is not None and _runs_on(drawn, False, cx, top, -1):
                top = None
            if bottom is not None and _runs_on(drawn, False, cx, bottom, +1):
                bottom = None
            if top is not None and top <= 3 and not _is_cap(ink, False, int(round(top)), cx):
                top = None
            if bottom is not None and bottom >= h_box - 4 and not _is_cap(ink, False, int(round(bottom)), cx):
                bottom = None
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
