"""Finding what is written inside a plot: rows of text, and legends.

A legend drawn inside the plot area - a marker or a short stretch of line, with its label to the right - is drawn in
the colours of the data, so that a reader would take its symbols for points, and its lines for a piece of trace. It
is found as two or more entries, each a symbol with a row of text just to its right, the symbols one above another
(or the entries side by side), and left out as the user's ``exclude`` boxes are.

Text is found as glyphs: small marks darker than what lies round them, of about one height, side by side along a
line, a little apart. Long lines, dots of a dotted line and dashes are not glyphs: too long, too small or too flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class TextRow:
    """A row of glyphs: its box (x0, y0, x1, y1, inclusive, in image pixels), the glyphs' height and their boxes."""

    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    glyphs: list = field(default_factory=list)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y0 + self.y1)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _ink(rgb_roi: np.ndarray, contrast: float = 70.0) -> np.ndarray:
    """Pixels much darker than the brightest pixels near them: print on paper, or on a pale shaded band."""
    lum = _luminance(rgb_roi)
    around = cv2.dilate(lum, np.ones((15, 15), np.uint8))
    return (around - lum) > contrast


def _print(rgb_roi: np.ndarray, contrast: float = 70.0) -> np.ndarray:
    """Ink that may be text: dark and grey (the words of a plot are black, the data in colour)."""
    f = rgb_roi.astype(np.float32)
    chroma = f.max(axis=2) - f.min(axis=2)
    return _ink(rgb_roi, contrast) & (chroma <= 60.0)


def find_text_rows(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    min_height: float = 6.0,
    max_height: Optional[float] = None,
    ink: Optional[np.ndarray] = None,
) -> list[TextRow]:
    """The rows of text inside *box* (x0, y0, x1, y1, inclusive) of the image: two or more glyphs of about one height,
    in a line, each within about a glyph's height of the next."""
    x0, y0, x1, y1 = box
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    h, w = roi.shape[:2]
    if h < 8 or w < 8:
        return []
    if max_height is None:
        max_height = max(min_height + 1, 0.12 * min(h, w), 12.0)
    mask = _print(roi) if ink is None else ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    glyphs = []
    for i in range(1, n):
        gx, gy, gw, gh, area = stats[i]
        if gx == 0 or gy == 0 or gx + gw >= w or gy + gh >= h:
            continue  # cut by the edge of the box: part of something bigger
        if not (min_height <= gh <= max_height) or gw > 1.8 * gh:
            continue
        if not (0.12 * gw * gh <= area <= 0.85 * gw * gh) and not (gw <= 0.35 * gh):
            continue  # a glyph is strokes: not a solid block (a bar, a marker) - unless it is a stroke itself (l, 1)
        glyphs.append((int(gx), int(gy), int(gw), int(gh)))
    if len(glyphs) < 2:
        return []
    glyphs.sort()
    # Join each glyph to the nearest one to its right that shares its line: their middles within a third of a
    # glyph's height, their extents overlapping most of the smaller, and the gap between them under a glyph's
    # height. (A lower-case letter is shorter than a capital: they share most of their extent, not all of it.)
    parent = list(range(len(glyphs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, (ax, ay, aw, ah) in enumerate(glyphs):
        best, best_gap = None, None
        for b in range(a + 1, len(glyphs)):
            bx, by, bw, bh = glyphs[b]
            gap = bx - (ax + aw)
            if gap > 1.0 * max(ah, bh):
                break
            if gap < -0.3 * min(aw, bw):
                continue
            overlap = min(ay + ah, by + bh) - max(ay, by)
            if overlap < 0.5 * min(ah, bh) or max(ah, bh) > 2.2 * min(ah, bh):
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = b, gap
        if best is not None:
            parent[find(best)] = find(a)
    groups: dict[int, list[int]] = {}
    for a in range(len(glyphs)):
        groups.setdefault(find(a), []).append(a)
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        gs = [glyphs[m] for m in members]
        rx0 = min(g[0] for g in gs)
        rx1 = max(g[0] + g[2] for g in gs) - 1
        ry0 = min(g[1] for g in gs)
        ry1 = max(g[1] + g[3] for g in gs) - 1
        height = float(np.median([g[3] for g in gs]))
        if rx1 - rx0 + 1 < 1.5 * height:
            continue
        # glyphs are of a size; the marks of a dotted or dashed line all alike, evenly spaced, and flat or tiny
        if all(g[3] <= 0.45 * g[2] for g in gs):
            continue  # dashes
        # and they stand on a line: most end at one baseline (bits of a trace, or of error bars, do not)
        bottoms = np.array([g[1] + g[3] for g in gs], dtype=float)
        tops = np.array([g[1] for g in gs], dtype=float)
        tol = max(1.5, 0.12 * height)
        on_base = np.abs(bottoms - np.median(bottoms)) <= tol
        if on_base.mean() < 0.75 or (len(gs) == 2 and not (np.abs(tops - np.median(tops)) <= tol).all()):
            continue
        # take in what is left of the row on its line: a subscript, a sign, a letter of another height
        while True:
            took = False
            for i in range(1, n):
                gx, gy, gw, gh, area = stats[i]
                if gx + gw - 1 >= rx0 or rx0 - (gx + gw) > 0.8 * height or gx <= 0:
                    continue
                if gh > 1.6 * height or gw > 2.0 * height or gh < 0.15 * height:
                    continue
                if gy > ry1 + 0.3 * height or gy + gh < ry0 - 0.3 * height:
                    continue
                if area > 0.85 * gw * gh and gw > 0.35 * gh:
                    continue  # a solid block: a marker, not print
                gs.append((int(gx), int(gy), int(gw), int(gh)))
                rx0, ry0, ry1 = int(gx), min(ry0, int(gy)), max(ry1, int(gy + gh - 1))
                took = True
            if not took:
                break
        rows.append(
            TextRow(
                x0=int(rx0 + x0),
                y0=int(ry0 + y0),
                x1=int(rx1 + x0),
                y1=int(ry1 + y0),
                height=height,
                glyphs=[(g[0] + x0, g[1] + y0, g[2], g[3]) for g in gs],
            )
        )
    return rows


def find_legend(rgb: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Boxes (x0, y0, x1, y1, image pixels) round the symbols and the words of the legends drawn inside *box*: none
    if there are none.

    An entry is a symbol - ink of any colour, in a short run of the row's band just left of a row of text, no taller
    than about twice the text and no longer than a few of its glyphs - and its text. A legend is two or more entries
    with their symbols one above another (their left ends together) or side by side in one row.
    """
    x0, y0, x1, y1 = box
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    h, w = roi.shape[:2]
    if h < 20 or w < 20:
        return []
    ink = _ink(roi)
    rows = find_text_rows(rgb, box, ink=_print(roi))
    if len(rows) < 1:
        return []
    # what is drawn: ink darker than what lies round it, or anything in colour (not the paper, nor a grey band)
    f = roi.astype(np.float32)
    drawn = ink | ((f.max(axis=2) - f.min(axis=2)) > 40.0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(drawn.astype(np.uint8), connectivity=8)
    text_pixels = np.zeros((h, w), dtype=bool)
    for r in rows:
        for gx, gy, gw, gh in r.glyphs:
            text_pixels[gy - y0 : gy - y0 + gh, gx - x0 : gx - x0 + gw] = True
    entries = []
    for r in rows:
        th = r.height
        band0 = int(max(0, r.cy - 1.0 * th - y0))
        band1 = int(min(h, r.cy + 1.0 * th - y0 + 1))
        reach0 = int(max(0, r.x0 - 6.0 * th - x0))
        reach1 = int(r.x0 - x0 - 1)
        if reach1 - reach0 < 3 or band1 - band0 < 3:
            continue
        window = np.zeros((h, w), dtype=bool)
        window[band0:band1, reach0:reach1] = True
        ids = np.unique(labels[window & drawn & ~text_pixels])
        pieces = []
        for i in ids[ids > 0]:
            sx, sy, sw, sh, sa = stats[i]
            # wholly (or nearly) inside the window: a symbol, not a trace passing by
            inside = int((labels[window] == i).sum())
            if inside < 0.85 * sa or sh > 2.5 * th or sw > 6.5 * th or sa < 4:
                continue
            pieces.append([sx, sy, sx + sw - 1, sy + sh - 1])
        if not pieces:
            continue
        # the symbol is what lies nearest the text, with whatever touches it (a marker on its stretch of line)
        pieces.sort(key=lambda b: -b[2])
        sym = list(pieces[0])
        for b in pieces[1:]:
            if b[2] >= sym[0] - 0.5 * th:
                sym = [min(sym[0], b[0]), min(sym[1], b[1]), max(sym[2], b[2]), max(sym[3], b[3])]
        # a symbol drawn in black or grey is a marker (a solid shape) or a stretch of line (flat), not more print
        patch = roi[sym[1] : sym[3] + 1, sym[0] : sym[2] + 1].astype(np.float32)
        on = drawn[sym[1] : sym[3] + 1, sym[0] : sym[2] + 1]
        if on.any() and float(np.median((patch.max(axis=2) - patch.min(axis=2))[on])) <= 40.0:
            sw_, sh_ = sym[2] - sym[0] + 1, sym[3] - sym[1] + 1
            line_sample = sh_ <= 0.4 * sw_ and sw_ >= 1.5 * th
            marker = on.mean() >= 0.75 and sh_ >= 0.5 * th and max(sw_, sh_) <= 2.0 * min(sw_, sh_)
            if not (line_sample or marker):
                continue  # (a dash between words is shorter than a legend's stretch of line, and flatter than a marker)
        # the symbol sits on the text's line, close to its left end
        scy = 0.5 * (sym[1] + sym[3]) + y0
        if abs(scy - r.cy) > 0.6 * th or (r.x0 - (sym[2] + x0)) > 3.0 * th:
            continue
        entries.append(
            (
                (sym[0] + x0, sym[1] + y0, sym[2] + x0, sym[3] + y0),
                (r.x0, r.y0, r.x1, r.y1),
                th,
            )
        )
    if len(entries) < 2:
        return []
    # entries that belong together: symbols' left ends together, one above another, each within a few lines of the
    # next; or side by side in one row
    def together(a: int, b: int) -> bool:
        sa, ta, ha = entries[a]
        sb, tb, hb = entries[b]
        hh = max(ha, hb)
        if hh > 1.35 * min(ha, hb):
            return False  # the entries of a legend are written alike
        size_a, size_b = max(sa[2] - sa[0], sa[3] - sa[1]) + 1, max(sb[2] - sb[0], sb[3] - sb[1]) + 1
        if max(size_a, size_b) > 2.0 * min(size_a, size_b):
            return False  # and their symbols drawn alike
        stacked = abs(sa[0] - sb[0]) <= max(3.0, 0.6 * hh) and abs(ta[1] - tb[1]) <= 4.5 * hh
        in_row = abs(0.5 * (ta[1] + ta[3]) - 0.5 * (tb[1] + tb[3])) <= 0.4 * hh and abs(tb[0] - ta[2]) <= 12 * hh
        return stacked or in_row

    parent = list(range(len(entries)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(entries)):
        for b in range(a + 1, len(entries)):
            if together(a, b):
                parent[find(b)] = find(a)
    groups: dict[int, list[int]] = {}
    for a in range(len(entries)):
        groups.setdefault(find(a), []).append(a)
    # each entry's symbol and words, a little beyond their edges (not the space between them, where data may lie)
    boxes = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for g in group:
            sym, text, th = entries[g]
            pad = max(2, int(round(0.15 * th)))
            boxes.append((int(sym[0]) - pad, int(sym[1]) - pad, int(sym[2]) + pad, int(sym[3]) + pad))
            boxes.append((int(text[0]) - pad, int(text[1]) - pad, int(text[2]) + pad, int(text[3]) + pad))
    return boxes


def drop_text(mask: np.ndarray, origin: tuple[int, int], rows: list[TextRow]) -> np.ndarray:
    """*mask* (a trace's pixels, whose top-left pixel is *origin* (x, y) in the image) without the glyphs of *rows*:
    each piece of the mask that lies wholly within a row's box (a label written in the trace's colour, such as
    "18:6") is taken out; a trace that runs through the box is joined to more of itself, and stays."""
    if not rows or not mask.any():
        return mask
    ox, oy = origin
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    drop = np.zeros(n, dtype=bool)
    for r in rows:
        # the row's box: its glyphs, and what lies between them (the dots of a colon, a decimal point)
        for gx, gy, gw, gh in [(r.x0, r.y0, r.x1 - r.x0 + 1, r.y1 - r.y0 + 1)]:
            a0, a1 = max(0, gy - oy - 1), min(h, gy - oy + gh + 1)
            b0, b1 = max(0, gx - ox - 1), min(w, gx - ox + gw + 1)
            if a1 <= a0 or b1 <= b0:
                continue
            for i in np.unique(labels[a0:a1, b0:b1]):
                if i == 0:
                    continue
                x, y, bw, bh, _ = stats[i]
                if x >= b0 and y >= a0 and x + bw <= b1 and y + bh <= a1:
                    drop[i] = True
    if not drop.any():
        return mask
    return mask & ~drop[labels]
