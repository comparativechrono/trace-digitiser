"""Reading the numbers printed beside an axis, and fitting the axis scale.

The work is split so that Tesseract only ever has to read tightly cropped,
enlarged tick labels:

1. Locate the labels geometrically.  Tick labels form the band of text
   nearest the axis; an axis title sits in a separate band further out.
2. Read the labels.  The first attempt reads all labels of an axis in a
   single Tesseract call (the crops are stacked into one image).  Small
   print is then read again in other ways and the readings pooled.
3. Attach each label to the tick mark or gridline it sits beside, which
   pins its position to a fraction of a pixel.
4. Fit ``value = a * pixel + b`` (or the logarithmic equivalent) by
   consensus, so that misread labels cannot spoil the result, and report
   how many labels agreed.

OCR is optional: if Tesseract is not installed the functions return no
labels and the caller falls back to user-supplied axis values.
"""

from __future__ import annotations

import functools
import itertools
import math
import shutil
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .frames import INK_THRESHOLD, _runs

SMALL_PRINT_PX = 8.0  # labels lower than this are read, but never with high confidence
_WHITELIST = "0123456789.,-+"
_MINUS_LIKE = "−–—‒~_"  # characters OCR produces for a minus sign


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TickLabel:
    """One tick label found beside an axis."""

    pos: float  # label centre along the axis: row for a y axis, column for an x axis
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive) in image pixels
    exponent_box: Optional[tuple[int, int, int, int]] = None  # a raised exponent, as in 10^3
    text: str = ""  # the most frequent OCR reading
    readings: list[str] = field(default_factory=list)  # every OCR reading, one per attempt
    values: list[float] = field(default_factory=list)  # possible numeric readings
    doubtful: list[bool] = field(default_factory=list)  # per value: a less likely interpretation
    anchor: Optional[float] = None  # the tick mark / gridline this label belongs to

    @property
    def where(self) -> float:
        """Best estimate of the tick position this label marks."""
        return self.anchor if self.anchor is not None else self.pos


@dataclass
class AxisFit:
    """Result of fitting an axis scale to tick labels."""

    a: float
    b: float
    scale: str  # "linear" or "log"
    inliers: list[tuple[float, float]]  # (pixel, value) pairs that agree with the fit
    n_labels: int  # labels that were read as numbers
    rms_px: float  # scatter of the agreeing labels about the fit, in pixels
    anchored: int  # how many of the agreeing labels were pinned to a tick or gridline
    exact: bool = False  # the labels are equally spaced and their values step by exactly the same amount
    n_plain: int = -1  # agreeing labels whose value is the straightforward reading of the text (-1: all)
    n_located: int = 0  # labels found beside the axis, read or not
    usual_step: bool = True  # the step between labels is one that axes are drawn with (10, 25, 0.5, 24 ...)
    glyph_px: float = 0.0  # height of the labels' print in pixels (0 if not known)
    rival: Optional[tuple[float, float]] = None  # another scale the labels support as well: its values at two labels

    @property
    def ambiguous(self) -> bool:
        """True if another, different scale is supported by the labels just as well as this one."""
        return self.rival is not None

    @property
    def n_total(self) -> int:
        """Labels the fit has to answer for: all that were found, legible or not."""
        return max(self.n_located, self.n_labels)

    @property
    def confidence(self) -> str:
        """How far to trust the fit.

        ``"high"``
            The labels are equally spaced, at least three of them step by
            exactly the same amount, at most one in four disagrees or could
            not be read, they are pinned to tick marks or gridlines, and
            the print is large enough to read reliably.
        ``"medium"``
            At least three labels agree, they are at least half of those
            found, and they step by an amount that axes are usually drawn
            with (or nearly all of them agree).  Usually right, but worth a
            look.
        ``"low"``
            Only two labels support the scale, or too few of the labels
            found agree with it to rule out an accident of misreading.  See
            :attr:`usable`.
        """
        n = len(self.inliers)
        total = self.n_total
        missing = total - n
        plain = n if self.n_plain < 0 else self.n_plain
        nearly_all = missing <= (0 if total <= 4 else 1)
        if n < 3:
            return "low"
        if not self.exact:
            # Unevenly spaced labels can only be held to a line, a much
            # weaker test, so nearly all of them have to agree.
            return "medium" if nearly_all and plain >= 2 else "low"
        if self.scale == "log":
            # Exponents are small whole numbers, which misreadings can
            # easily put in step by accident.
            believable = missing <= 1 and (n >= 4 or missing == 0)
        elif self.usual_step:
            believable = 2 * n >= total
        else:
            believable = nearly_all
        if not believable or plain < 2:
            return "low"
        allowed = 0 if total <= 3 else 1 if total <= 7 else 2
        if not (missing <= allowed and 2 * plain >= n and plain >= 2 and self.rms_px <= 1.5 and self.anchored >= 2):
            return "medium"
        # Print this small is misread in ways that agree with each other -
        # the same digit or the decimal point dropped from every label.  And
        # when the readings support another scale just as well (a lost minus
        # sign in each of two exponents, say), neither can be trusted fully.
        return "medium" if self.small_print or self.ambiguous else "high"

    @property
    def small_print(self) -> bool:
        """True if the labels are too small (under 8 pixels high) for OCR to be relied on."""
        return 0 < self.glyph_px < SMALL_PRINT_PX

    @property
    def usable(self) -> bool:
        """False if the fit is more likely to be an accident of misread labels than the scale of the axis.

        A low-confidence fit is kept only when the axis has so few labels
        that two agreeing is all that could be hoped for.
        """
        if self.confidence != "low":
            return True
        return len(self.inliers) >= 2 and self.n_total <= 3

    def value_at(self, pixel):
        v = self.a * np.asarray(pixel, dtype=float) + self.b
        return np.power(10.0, v) if self.scale == "log" else v


@dataclass(frozen=True)
class Preparation:
    """One way of presenting tick labels to Tesseract."""

    target_height: int  # enlarge the text to about this many pixels
    binarise: bool  # threshold to pure black and white
    sharpen: float  # unsharp-mask radius (0 = none)
    lanczos: bool  # Lanczos rather than bicubic enlargement
    stacked: bool  # read all labels in one call (fast) or one call per label
    psm: int  # Tesseract page-segmentation mode
    gamma: float = 1.0  # below 1 thickens faint strokes before enlarging


# Small print is read differently at different enlargements, and no single
# setting reads every label of a low-resolution figure.  Labels that are not
# settled by the first (fast) attempt are therefore read again in other ways
# and the readings pooled; the axis fit then keeps the readings that agree.
# The settings and their order were chosen on labels 5 to 7 pixels high in
# three typefaces: each one added reads labels that the ones before it miss.
PREPARATIONS = (
    Preparation(44, True, 2.0, True, True, 6),
    Preparation(64, False, 2.0, True, False, 7),
    Preparation(64, True, 0.0, False, False, 7, gamma=0.6),
    Preparation(32, False, 2.0, True, False, 7),
    Preparation(64, False, 0.0, False, False, 7, gamma=0.6),
)


# ---------------------------------------------------------------------------
# Tesseract plumbing
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """True if the Tesseract program can be run."""
    try:
        import pytesseract

        if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
            return False
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _prepare_crop(gray: np.ndarray, prep: Preparation) -> np.ndarray:
    """Enlarge and sharpen one label crop (dark text on white)."""
    if prep.gamma != 1.0:
        g = gray.astype(np.float32)
        lo, hi = float(g.min()), float(np.percentile(g, 98))
        ink = 1.0 - np.clip((g - lo) / max(1.0, hi - lo), 0.0, 1.0)
        gray = (255.0 * (1.0 - np.power(ink, prep.gamma))).astype(np.uint8)
    h = max(1, gray.shape[0])
    scale = float(np.clip(prep.target_height / h, 1.0, 14.0))
    interpolation = cv2.INTER_LANCZOS4 if prep.lanczos else cv2.INTER_CUBIC
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
    if prep.sharpen > 0:
        blurred = cv2.GaussianBlur(big, (0, 0), prep.sharpen)
        big = cv2.addWeighted(big, 1.6, blurred, -0.6, 0)
    if prep.binarise:
        _, big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return big


def _read_crops(
    crops: list[np.ndarray],
    prep: Preparation = PREPARATIONS[0],
    numbers_only: bool = True,
) -> list[str]:
    """OCR a list of greyscale label crops.

    With ``prep.stacked`` the crops are placed in a single column, one label
    per line, and read with one Tesseract call instead of one per label.
    With *numbers_only* Tesseract is restricted to digits, signs and
    separators, which makes it far more reliable on tick labels.
    """
    if not crops or not tesseract_available():
        return ["" for _ in crops]
    import pytesseract
    from pytesseract import Output

    prepared = [_prepare_crop(c, prep) for c in crops]
    pad = 18
    config = f"--psm {prep.psm}" + (f" -c tessedit_char_whitelist={_WHITELIST}" if numbers_only else "")

    if not prep.stacked:
        out = []
        for img in prepared:
            single = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            try:
                out.append(pytesseract.image_to_string(single, config=config).strip())
            except Exception:
                out.append("")
        return out

    width = max(p.shape[1] for p in prepared) + 2 * pad
    rows: list[np.ndarray] = [np.full((pad, width), 255, np.uint8)]
    spans: list[tuple[int, int]] = []
    y = pad
    for p in prepared:
        line = np.full((p.shape[0], width), 255, np.uint8)
        line[:, pad : pad + p.shape[1]] = p
        rows.append(line)
        spans.append((y, y + p.shape[0]))
        y += p.shape[0]
        rows.append(np.full((pad, width), 255, np.uint8))
        y += pad
    sheet = np.vstack(rows)

    try:
        data = pytesseract.image_to_data(sheet, config=config, output_type=Output.DICT)
    except Exception:
        return ["" for _ in crops]

    words: list[list[tuple[int, str]]] = [[] for _ in crops]
    for text, left, top, height in zip(data["text"], data["left"], data["top"], data["height"]):
        text = str(text).strip()
        if not text:
            continue
        centre = top + 0.5 * height
        for i, (a, b) in enumerate(spans):
            if a - pad / 2 <= centre <= b + pad / 2:
                words[i].append((left, text))
                break
    return ["".join(t for _, t in sorted(w)) for w in words]


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    t = "".join("-" if ch in _MINUS_LIKE else ch for ch in str(text).strip())
    return "".join(ch for ch in t if ch in "0123456789.,-+")


def parse_number(text: str) -> list[tuple[float, bool]]:
    """Return the numbers a tick-label string may stand for.

    Each entry is ``(value, doubtful)``.  OCR of small print drops decimal
    points and confuses separators, so where the text is ambiguous the less
    likely reading is included and marked doubtful:

    * ``"1,000"`` is one thousand; ``"0,5"`` is a decimal comma.
    * ``"05"`` cannot be an integer label, so it is read as 0.5 (the point
      was lost); the plain reading 5 is kept as the doubtful one.
    """
    t = _clean(text)
    if not t:
        return []
    sign = -1.0 if t.startswith("-") else 1.0
    body = t.lstrip("+-").replace("-", "").replace("+", "")
    if not any(ch.isdigit() for ch in body):
        return []

    if "," in body and "." not in body:
        head, _, tail = body.rpartition(",")
        if len(tail) == 3 and head.replace(",", "").isdigit() and head[0] != "0":
            body = body.replace(",", "")  # thousands separator
        else:
            body = body.replace(",", ".")  # decimal comma
    else:
        body = body.replace(",", "")
    if body.count(".") > 1:
        head, _, tail = body.rpartition(".")
        body = head.replace(".", "") + "." + tail
    body = body.rstrip(".")
    if body.startswith("."):
        body = "0" + body
    if not body:
        return []
    try:
        plain = sign * float(body)
    except ValueError:
        return []
    if not math.isfinite(plain):
        return []

    if "." not in body and len(body) > 1 and body[0] == "0":
        lost_point = sign * float("0." + body[1:])  # "05" -> 0.5, "025" -> 0.25
        return [(lost_point, False)] + ([(plain, True)] if plain != lost_point else [])
    return [(plain, False)]


def _combine_power(base_text: str, exponent_text: str) -> list[tuple[float, bool]]:
    """Interpret a number that is followed by a small raised mark.

    ``10`` with a raised ``3`` is a thousand.  A raised minus sign is often
    lost in small print, so the reciprocal is offered as a less likely
    reading.  Every other reading is doubtful: the base alone (the raised
    mark may be a degree sign, but is more often an exponent that could not
    be read) and the two run together (``103``).
    """
    base = _clean(base_text)
    digits = "".join(ch for ch in base if ch.isdigit())
    exponent = _clean(exponent_text)
    exponent_digits = "".join(ch for ch in exponent if ch.isdigit())
    out: list[tuple[float, bool]] = []

    def add(value: float, doubtful: bool) -> None:
        if math.isfinite(value) and all(value != v for v, _ in out):
            out.append((value, doubtful))

    if digits == "10" and "." not in base and "," not in base and 0 < len(exponent_digits) <= 3:
        sign = -1.0 if base.startswith("-") else 1.0
        power = -int(exponent_digits) if exponent.startswith("-") else int(exponent_digits)
        add(sign * 10.0 ** power, False)
        if power != 0:
            add(sign * 10.0 ** -power, True)
    else:
        # Not legibly a power of ten: most likely one that was misread, so
        # the base alone is kept only as a doubtful reading.
        for value, _ in parse_number(base_text):
            add(value, True)
    for value, _ in parse_number(base_text + exponent_text.lstrip("-")):
        add(value, True)
    return out


# ---------------------------------------------------------------------------
# Locating labels
# ---------------------------------------------------------------------------

def _text_ink(dark_strip: np.ndarray) -> np.ndarray:
    """Ink mask for printed text in a strip of the darkness map."""
    if dark_strip.size == 0:
        return np.zeros(dark_strip.shape, dtype=bool)
    top = float(np.percentile(dark_strip, 99.5))
    return dark_strip > max(INK_THRESHOLD, 0.35 * top)


def _glyph_height(ink: np.ndarray) -> float:
    """Typical height of a character in an ink mask."""
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    heights = [int(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 3]
    heights = [v for v in heights if v >= 4]
    return float(np.median(heights)) if heights else 0.0


def _without_specks(ink: np.ndarray, glyph_h: float) -> np.ndarray:
    """Remove marks too small to be print and too far from any print to belong to it.

    Scans and noisy images carry specks.  One that falls in line with a
    label stretches the label's box and is read as a point or a minus sign.
    A real decimal point or comma is just as small, but sits right beside a
    digit; a minus sign is long enough not to be taken for a speck.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return ink
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    small = extent < 0.35 * glyph_h
    small[0] = False
    if not small.any():
        return ink
    print_mask = (~small)[labels] & ink
    reach = max(2, int(round(0.45 * glyph_h)))
    beside_print = cv2.dilate(print_mask.astype(np.uint8), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
    keep = ~small
    for i in np.flatnonzero(small):
        keep[i] = bool(beside_print[labels == i].any())
    keep[0] = False
    return keep[labels]


def _merge_runs(runs: list[tuple[int, int]], max_gap: float) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _exponent_start(ink: np.ndarray) -> Optional[int]:
    """Column at which a raised exponent begins in an upright label, or None.

    An exponent is a run of trailing glyphs that are smaller than the base
    and sit clear above its baseline.
    """
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    glyphs = [stats[i] for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 3]
    if len(glyphs) >= 3:
        glyphs.sort(key=lambda g: g[cv2.CC_STAT_LEFT])
        main_h = max(g[cv2.CC_STAT_HEIGHT] for g in glyphs[:2])
        baseline = max(g[cv2.CC_STAT_TOP] + g[cv2.CC_STAT_HEIGHT] for g in glyphs[:2])

        def raised(g) -> bool:
            bottom = g[cv2.CC_STAT_TOP] + g[cv2.CC_STAT_HEIGHT]
            return baseline - bottom >= 0.25 * main_h and g[cv2.CC_STAT_HEIGHT] <= 0.85 * main_h

        start = None
        for g in reversed(glyphs[2:]):
            if not raised(g):
                break
            start = int(g[cv2.CC_STAT_LEFT])
        if start is not None:
            return start
    return _raised_tail(ink)


def _raised_tail(ink: np.ndarray) -> Optional[int]:
    """Column at which the label's ink stops reaching down to the baseline, or None.

    Finds an exponent that touches its base, so that the two are one blob
    (as they are in tightly set type).  The tail must be wide enough to be a
    character: the bar of a 7 or a 4 also ends clear of the baseline, but in
    a column or two.
    """
    columns = np.flatnonzero(ink.any(axis=0))
    rows = np.flatnonzero(ink.any(axis=1))
    if len(columns) < 3 or len(rows) < 8:
        return None
    baseline, height = int(rows[-1]), int(rows[-1] - rows[0] + 1)
    lowest = np.array([np.flatnonzero(ink[:, c])[-1] if ink[:, c].any() else -1 for c in range(ink.shape[1])])
    start = None
    for c in range(int(columns[-1]), int(columns[0]) - 1, -1):
        if lowest[c] < 0:
            continue  # a gap between characters
        if baseline - lowest[c] < 0.25 * height:
            break
        start = c
    if start is None:
        return None
    tail, whole = int(columns[-1]) - start + 1, int(columns[-1] - columns[0]) + 1
    if tail < 0.3 * height or start - int(columns[0]) < 0.5 * height or tail > 0.6 * whole:
        return None
    return start


def find_tick_labels(
    rgb: np.ndarray,
    dark: np.ndarray,
    panel: dict,
    axis: str,
    limit: Optional[float] = None,
) -> list[TickLabel]:
    """Locate the tick labels of one axis of a panel (positions only, no OCR).

    Parameters
    ----------
    axis : ``"y"`` (labels to the left of the plot) or ``"x"`` (labels below).
    limit : float, optional
        Do not look further out than this image coordinate (a column for
        ``"y"``, a row for ``"x"``) - normally the edge of the neighbouring
        panel.
    """
    h, w = dark.shape
    edges = panel["edges"]
    half = panel.get("line_thickness", 1) // 2 + 1
    span = max(edges["right"] - edges["left"], edges["bottom"] - edges["top"])
    pad = int(max(6, 0.04 * span))

    # Work in a frame where the first index is the distance outward from the
    # axis and the second runs along it, whichever axis this is.
    if axis == "y":
        x_hi = int(math.floor(edges["left"])) - half  # the column next to the axis
        x_lo = max(0, x_hi - int(0.45 * span) - 20)
        if limit is not None:
            x_lo = max(x_lo, int(math.ceil(limit)))
        y_lo = max(0, int(edges["top"]) - pad)
        y_hi = min(h - 1, int(edges["bottom"]) + pad)
        if x_hi - x_lo < 4 or y_hi - y_lo < 4:
            return []
        outward = dark[y_lo : y_hi + 1, x_lo : x_hi + 1][:, ::-1].T
        lo, hi = edges["top"] - y_lo, edges["bottom"] - y_lo
    elif axis == "x":
        y_lo = int(math.ceil(edges["bottom"])) + half
        y_hi = min(h - 1, y_lo + int(0.30 * span) + 20)
        if limit is not None:
            y_hi = min(y_hi, int(math.floor(limit)))
        # A label is centred on its tick, so the first and last reach well
        # past the ends of the axis ("1990" under the origin).
        beyond = max(pad, int(0.15 * span))
        x_lo = max(0, int(edges["left"]) - beyond)
        x_hi = min(w - 1, int(edges["right"]) + beyond)
        if y_hi - y_lo < 4 or x_hi - x_lo < 4:
            return []
        outward = dark[y_lo : y_hi + 1, x_lo : x_hi + 1]
        lo, hi = edges["left"] - x_lo, edges["right"] - x_lo
    else:
        raise ValueError("axis must be 'x' or 'y'")

    ink = _text_ink(outward)

    # Remove what touches the axis side of the strip: tick marks (thin
    # stubs) and, beyond the ends of the axis, the other axis's end label,
    # which pokes round the corner and is cut off by the edge of the strip.
    max_thick = max(3, int(round(0.012 * min(h, w))) + 1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_TOP] > 1:
            continue
        centre = stats[i, cv2.CC_STAT_LEFT] + 0.5 * (stats[i, cv2.CC_STAT_WIDTH] - 1)
        if stats[i, cv2.CC_STAT_WIDTH] <= max_thick or not lo <= centre <= hi:
            ink[labels == i] = False

    glyph_h = _glyph_height(ink if axis == "x" else ink.T)
    if glyph_h <= 0:
        return []
    ink = _without_specks(ink, glyph_h)

    # A tick label is centred on its tick, so it cannot reach further past
    # the end of the axis than half its own length: half a line of print on
    # a y axis, a few characters on an x axis.
    reach = (0.75 if axis == "y" else 4.0) * glyph_h
    ink[:, : max(0, int(math.floor(lo - reach)))] = False
    ink[:, int(math.ceil(hi + reach)) + 1 :] = False

    # Bands of text by distance from the axis.  The nearest band holds the
    # tick labels; an axis title is in a band of its own further out.
    depth_runs = _merge_runs(_runs(ink.any(axis=1)), max_gap=max(1.0, 0.2 * glyph_h))
    depth_runs = [r for r in depth_runs if ink[r[0] : r[1] + 1].sum() >= 4]

    def to_image_box(da: int, db: int, a: int, b: int) -> tuple[int, int, int, int]:
        """Box given as (distance range, along range) -> (x0, y0, x1, y1) in the image."""
        if axis == "y":
            return (x_hi - db, y_lo + a, x_hi - da, y_lo + b)
        return (x_lo + a, y_lo + da, x_lo + b, y_lo + db)

    def labels_in(d0: int, d1: int) -> list[TickLabel]:
        band = ink[d0 : d1 + 1]
        along = band.any(axis=0)
        # Along a y axis each label is a line of its own; along an x axis
        # the digits of a label sit closer together than neighbouring labels.
        runs = _runs(along) if axis == "y" else _merge_runs(_runs(along), max_gap=max(1.0, 0.45 * glyph_h))
        out: list[TickLabel] = []
        for a, b in runs:
            piece = band[:, a : b + 1]
            if piece.sum() < 4:
                continue
            depths = np.flatnonzero(piece.any(axis=1))
            da, db = int(depths[0]) + d0, int(depths[-1]) + d0
            box = to_image_box(da, db, a, b)
            height = box[3] - box[1] + 1
            if not 0.5 * glyph_h <= height <= 3.0 * glyph_h:
                continue
            if not lo - 0.5 * glyph_h <= 0.5 * (a + b) <= hi + 0.5 * glyph_h:
                continue
            label = TickLabel(pos=0.5 * (box[1] + box[3]) if axis == "y" else 0.5 * (box[0] + box[2]), box=box)
            upright = ink[da : db + 1, a : b + 1].T[:, ::-1] if axis == "y" else ink[da : db + 1, a : b + 1]
            start = _exponent_start(upright)
            if start is not None and start > 0:
                # Split into base and exponent; the exponent occupies the top
                # of the label from column `start` rightwards.
                rows = np.flatnonzero(upright[:, start:].any(axis=1))
                label.exponent_box = (box[0] + start, box[1] + int(rows[0]), box[2], box[1] + int(rows[-1]))
                label.box = (box[0], box[1], box[0] + start - 1, box[3])
            out.append(label)
        return out

    best: list[TickLabel] = []
    for k, (d0, d1) in enumerate(depth_runs):
        if d0 > 4.0 * glyph_h + 0.12 * span:
            break  # too far from the axis to be its tick labels
        # One label can fall into two bands: right-aligned numbers leave a
        # clear column between their digits (10, 12, 14 ...), and an exponent
        # or a minus sign stands apart from the digits.  Absorb the next band
        # if it is close and every piece of it lines up with a label already
        # found.  A rotated axis title has letters between the labels too, so
        # it is left out.
        for e0, e1 in depth_runs[k + 1 :]:
            if e0 - d1 - 1 > 0.6 * glyph_h:
                break
            lines = _runs(ink[d0 : d1 + 1].any(axis=0))
            pieces = _runs(ink[e0 : e1 + 1].any(axis=0))
            slack = 0.6 * glyph_h  # an exponent rises above its base
            if not all(
                any(la - slack <= a and b <= lb + slack and a <= lb and b >= la for la, lb in lines)
                for a, b in pieces
            ):
                break
            d1 = e1
        found = labels_in(d0, d1)
        if len(found) > len(best):
            best = found
        if len(found) >= 2:
            break
    return best


# ---------------------------------------------------------------------------
# Reading labels
# ---------------------------------------------------------------------------

# Letters that OCR produces in place of digits in small print.
_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "s": "5", "B": "8", "Z": "2"}
)


def labels_are_numeric(rgb: np.ndarray, labels: list[TickLabel]) -> bool:
    """Decide whether a row of tick labels is numbers or words.

    The labels are read once without restricting Tesseract to digits.  An
    axis labelled "1N 1D 2N ..." or "Mon Tue ..." then shows its letters,
    and is not sent for numeric calibration (where, forced to produce
    digits, Tesseract would invent some).
    """
    if not labels:
        return False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    texts = _read_crops([_crop(gray, lab.box) for lab in labels], PREPARATIONS[0], numbers_only=False)
    read = [t.translate(_DIGIT_LOOKALIKES) for t in texts if t.strip()]
    if not read:
        return True  # nothing legible this way: let the numeric reader try
    numeric = sum(1 for t in read if not any(ch.isalpha() for ch in t) and any(ch.isdigit() for ch in t))
    return numeric >= 0.5 * len(read)


def _crop(gray: np.ndarray, box: tuple[int, int, int, int], margin: int = 2) -> np.ndarray:
    h, w = gray.shape
    x0, y0, x1, y1 = box
    return gray[max(0, y0 - margin) : min(h, y1 + margin + 1), max(0, x0 - margin) : min(w, x1 + margin + 1)]


def _add_value(lab: TickLabel, value: float, doubtful: bool) -> None:
    """Record a possible value for a label; a value is doubtful only if every reading of it is."""
    if value in lab.values:
        j = lab.values.index(value)
        lab.doubtful[j] = lab.doubtful[j] and doubtful
    else:
        lab.values.append(value)
        lab.doubtful.append(doubtful)


def _whole_number(text: str) -> Optional[tuple[float, str]]:
    """``(sign, digits)`` if *text* was read as a whole number - no point, comma or exponent - else None."""
    if "^" in text:
        return None
    t = _clean(text)
    body = t.lstrip("+-").replace("-", "").replace("+", "")
    if not body.isdigit():
        return None
    return (-1.0 if t.startswith("-") else 1.0), body


def _text_units(text: str) -> float:
    """Printed width of a number, in digit widths.

    Digits are all the same width in the typefaces used for axes; a point or
    comma takes about half a digit, and a minus sign about one.
    """
    return sum(1.0 if ch.isdigit() else 0.5 if ch in ".," else 0.9 for ch in _clean(text))


def _readings_that_fit(labels: list[TickLabel]) -> list[list[str]]:
    """Each label's readings, most frequent first, less those too long or too short for the label.

    Digits are equally wide, so the width of a label gives the number of
    characters in it once the width of one character is known.  The height
    of the print bounds that width, and each reading narrows it further: a
    label *w* pixels wide, read as *n* characters, needs a character width
    between about ``w / n`` and ``w / (n - 1)``.  A reading that no
    believable character width allows is dropped (``6`` for a label as wide
    as ``0.6``); and if most labels agree on a character width, so are the
    readings that do not fit it (``000`` for ``5000``).
    """
    ranked = [
        sorted(set(lab.readings), key=lambda t: (-lab.readings.count(t), lab.readings.index(t))) for lab in labels
    ]
    if not any(ranked):
        return ranked
    height = float(np.median([lab.box[3] - lab.box[1] + 1 for lab in labels]))
    # Digits are 0.6 to 1.0 times as wide as they are tall in every common typeface.
    narrowest, widest = 0.5 * height, 1.15 * height + 0.5

    def span(text: str, width: int) -> Optional[tuple[float, float]]:
        """Character widths with which *text* would be about *width* pixels wide."""
        units = _text_units(text.partition("^")[0])
        if units <= 0:
            return None
        # From the left edge of the first character to the right edge of the
        # last is between (units - 0.85) and (units + 0.1) character widths:
        # the last character has no gap after it, and a 1 is narrow.
        lo = max(narrowest, width / (units + 0.1))
        hi = min(widest, width / (units - 0.85)) if units > 0.85 else widest
        return (lo, hi) if lo <= hi else None

    spans = [[span(t, lab.box[2] - lab.box[0] + 1) for t in r] for lab, r in zip(labels, ranked)]

    def supporters(advance: float) -> int:
        return sum(1 for ss in spans if any(sp is not None and sp[0] <= advance <= sp[1] for sp in ss))

    edges = sorted({edge for ss in spans for sp in ss if sp is not None for edge in sp})
    most = max((supporters(e) for e in edges), default=0)
    read = sum(1 for r in ranked if r)
    if most >= 3 and most >= 0.6 * read:
        agreed = [e for e in edges if supporters(e) == most]
        return [
            [t for t, sp in zip(r, ss) if sp is not None and any(sp[0] <= e <= sp[1] for e in agreed)]
            for r, ss in zip(ranked, spans)
        ]
    return [[t for t, sp in zip(r, ss) if sp is not None] for r, ss in zip(ranked, spans)]


def _interpret(labels: list[TickLabel]) -> None:
    """Work out each label's possible values from all its readings so far (in place)."""
    for lab, ranked in zip(labels, _readings_that_fit(labels)):
        lab.text = ranked[0] if ranked else ""
        lab.values, lab.doubtful = [], []
        # A reading with a decimal point shows up the same digits read
        # without one ("39.0" and "390") as having lost the point.
        with_point = {
            "".join(ch for ch in t if ch.isdigit()) for t in ranked if "^" not in t and "." in _clean(t)
        }
        for t in ranked:
            base, sep, exponent = t.partition("^")
            whole = _whole_number(t)
            lost_point = whole is not None and whole[1] in with_point
            for value, doubtful in (_combine_power(base, exponent) if sep else parse_number(base)):
                _add_value(lab, value, doubtful or lost_point)
    _restore_lost_points(labels)
    _share_decimals(labels)


def read_tick_labels(
    rgb: np.ndarray,
    labels: list[TickLabel],
    preparation: Preparation = PREPARATIONS[0],
) -> list[TickLabel]:
    """OCR the given labels (in place) and return those that read as numbers.

    Calling this again with another *preparation* adds to each label's
    readings; ``values`` is rebuilt from all the readings so far.
    """
    if not labels:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    crops = [_crop(gray, lab.box) for lab in labels]
    with_exponent = [i for i, lab in enumerate(labels) if lab.exponent_box is not None]
    crops += [_crop(gray, labels[i].exponent_box) for i in with_exponent]  # type: ignore[arg-type]
    texts = _read_crops(crops, preparation)
    exponent_text = dict(zip(with_exponent, texts[len(labels) :]))

    for i, lab in enumerate(labels):
        if _clean(texts[i]):
            lab.readings.append(texts[i] if i not in exponent_text else f"{texts[i]}^{exponent_text[i]}")
    _interpret(labels)
    return [lab for lab in labels if lab.values]


def _restore_lost_points(labels: list[TickLabel]) -> None:
    """Apply evidence of a dropped decimal point to the whole axis, in place.

    A label read as ``"05"`` shows that the axis is written with one decimal
    and that OCR is losing the point.  Its neighbours ``"10"`` and ``"15"``
    are then far more likely to be 1.0 and 1.5 than ten and fifteen, so for
    every label read without a point the scaled value becomes the preferred
    reading and the plain one is marked doubtful.  Only a label's leading
    reading counts as evidence - and by this point it has been checked
    against the width of the label, so ``"00"`` is a label two characters
    wide (``0.0``), not a misreading of ``0`` or ``300``.
    """
    decimals = 0
    for lab in labels:
        whole = _whole_number(lab.text) if lab.text else None
        if whole and len(whole[1]) > 1 and whole[1][0] == "0":
            decimals = max(decimals, len(whole[1]) - 1)
    if decimals == 0:
        return
    for lab in labels:
        for t in dict.fromkeys(lab.readings):
            whole = _whole_number(t)
            if whole is None:
                continue
            plain = whole[0] * float(whole[1])
            scaled = plain / 10.0 ** decimals
            if scaled == plain or plain not in lab.values:
                continue
            lab.doubtful[lab.values.index(plain)] = True
            if scaled in lab.values:
                lab.doubtful[lab.values.index(scaled)] = False
            else:
                lab.values.append(scaled)
                lab.doubtful.append(False)


def _share_decimals(labels: list[TickLabel]) -> None:
    """Offer the decimal reading of a label read without its point, in place.

    Where at least two labels of the axis were read with the same number of
    decimals, a label read as a bare whole number may have lost its point
    (``365`` beside ``36.0`` and ``37.0``), so the scaled value is added for
    the axis fit to weigh.  If one of the labels ends in a zero after the
    point (``5.0``), the axis is written with that many decimals throughout,
    and a bare number can only be a lost point: then the scaled value is the
    likely one and the plain one doubtful.  Otherwise it is the other way
    round, because ``0, 2.5, 5, 7.5, 10`` is an ordinary way to label an axis.
    """
    seen: dict[int, int] = {}
    throughout = False
    for lab in labels:
        t = _clean(lab.text) if "^" not in lab.text else ""
        if t.count(".") == 1 and t.replace(".", "").lstrip("+-").isdigit():
            fraction = t.rpartition(".")[2]
            seen[len(fraction)] = seen.get(len(fraction), 0) + 1
            throughout = throughout or (fraction != "" and set(fraction) == {"0"})
    if not seen:
        return
    decimals, count = max(seen.items(), key=lambda item: item[1])
    if count < 2 or decimals == 0:
        return
    for lab in labels:
        for t in dict.fromkeys(lab.readings):
            whole = _whole_number(t)
            if whole is None:
                continue
            plain = whole[0] * float(whole[1])
            scaled = plain / 10.0 ** decimals
            if plain not in lab.values or scaled == plain:
                continue  # the reading was set aside, or is nought either way
            if throughout:
                lab.doubtful[lab.values.index(plain)] = True
            if scaled in lab.values:
                lab.doubtful[lab.values.index(scaled)] = lab.doubtful[lab.values.index(scaled)] and not throughout
            else:
                lab.values.append(scaled)
                lab.doubtful.append(not throughout)


def anchor_labels(labels: list[TickLabel], anchors: list[float], tolerance: Optional[float] = None) -> None:
    """Attach each label to the nearest tick mark or gridline, in place."""
    if not labels or not anchors:
        return
    anchors_arr = np.asarray(sorted(anchors), dtype=float)
    for lab in labels:
        x0, y0, x1, y1 = lab.box
        size = min(x1 - x0 + 1, y1 - y0 + 1)
        tol = tolerance if tolerance is not None else max(2.5, 0.6 * size)
        i = int(np.argmin(np.abs(anchors_arr - lab.pos)))
        if abs(anchors_arr[i] - lab.pos) <= tol:
            lab.anchor = float(anchors_arr[i])


# ---------------------------------------------------------------------------
# Fitting the axis scale
# ---------------------------------------------------------------------------

def _round_steps(values: list[float], scale: str) -> bool:
    """True if neighbouring labels differ by a round amount (1, 2, 2.5 or 5 times a power of ten).

    Axes are drawn with round tick steps, so of two candidate scales that
    explain the same number of labels, the one with round steps is right.
    *values* are in the fitted space (log10 for a log scale).
    """
    values = sorted({round(v, 9) for v in values})
    if len(values) < 2:
        return False
    step = min(b - a for a, b in zip(values, values[1:]))
    if step <= 0:
        return False
    if scale == "log":
        return abs(step - round(step)) < 0.02 and round(step) >= 1
    mantissa = step / 10.0 ** math.floor(math.log10(step))
    return any(abs(mantissa - m) < 0.02 * m for m in (1.0, 2.0, 2.5, 5.0, 10.0))


# Leading figures of the steps that axes are labelled in: the round ones,
# and those of clocks, calendars and angles (12, 24, 48, 7, 15, 45, 90, 180 ...).
_USUAL_STEPS = (1.0, 1.2, 1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.6, 4.0, 4.5, 4.8, 5.0, 6.0, 7.0, 7.5, 8.0, 9.0)


def _is_usual_step(step: float, scale: str) -> bool:
    """True if labels a *step* apart are what an axis would be drawn with.

    A scale that misread labels happen to agree on almost never has such a
    step (9, -100 and -209 are 109 apart).  On a logarithmic axis the step
    is in powers of ten and must be a whole number of them.
    """
    step = abs(step)
    if not step > 0 or not math.isfinite(step):
        return False
    if scale == "log":
        return round(step) >= 1 and abs(step - round(step)) < 0.01
    mantissa = step / 10.0 ** math.floor(math.log10(step))
    return any(abs(mantissa - m) < 1e-6 * m for m in (*_USUAL_STEPS, 10.0))


def _decimals(value: float) -> int:
    """Decimal places needed to print *value*."""
    return len(f"{value:.10f}".rstrip("0").partition(".")[2])


def _nearest_round(step: float, scale: str) -> Optional[float]:
    """The round tick step nearest to *step*, or None if *step* is not close to one."""
    if not step > 0 or not math.isfinite(step):
        return None
    if scale == "log":
        candidates = [float(max(1, round(step)))]
    else:
        decade = 10.0 ** math.floor(math.log10(step))
        candidates = [m * decade for m in (1.0, 2.0, 2.5, 5.0, 10.0)]
    best = min(candidates, key=lambda c: abs(c - step))
    return best if abs(best - step) <= 0.08 * step else None


def _on_lattice(agree: list[tuple], slope: float, spacing_px: float, scale: str) -> list[tuple]:
    """Drop readings that do not fall on the axis's regular tick values.

    Tick labels step by a round amount (0, 50, 100 ...).  The fitted slope
    and the spacing of the labels give that step; a reading such as 8 where
    0 belongs may lie close enough to the line to pass as agreeing, but it
    is not a multiple of the step and so is recognised as a misreading.
    Axes whose labels do not step by a round amount are left alone.
    """
    if len(agree) < 3:
        return agree
    step = _nearest_round(abs(slope) * spacing_px, scale)
    if step is None:
        return agree
    residues = [(t / step) % 1.0 for _, t, _, _ in agree]

    def near(r: float, r0: float) -> bool:
        # Tick values are exact multiples of the step, so this only has to
        # absorb floating-point error.
        d = abs(r - r0)
        return min(d, 1.0 - d) <= 0.02

    members = max(([i for i, r in enumerate(residues) if near(r, r0)] for r0 in residues), key=len)
    return [agree[i] for i in members]


# How far, in tick steps, a label's value may be from its place among
# equally stepped values.  Printed labels are exact, so this only has to
# cover rounding in the print ("0.33", "0.67", "1.00").
_STEP_TOLERANCE = 0.03


def _rungs(where: list[float], tolerances: list[float]) -> Optional[list[int]]:
    """Place the labels on a ladder of equally spaced ticks.

    Returns each label's rung (0 for the first along the axis), or None if
    the labels are not equally spaced - as with hand-picked tick positions,
    or the 1, 2, 5, 10 of some logarithmic axes.  A gap where a label was
    not found simply leaves a rung empty.
    """
    n = len(where)
    if n < 2:
        return None
    order = sorted(range(n), key=lambda i: where[i])
    w = np.array([where[i] for i in order], dtype=float)
    unit = float(np.diff(w).min())
    if unit < 4:
        return None
    origin = float(w[0])
    k = np.zeros(n)
    for _ in range(3):
        k = np.round((w - origin) / unit)
        if len(np.unique(k)) < n:
            return None
        unit, origin = (float(v) for v in np.polyfit(k, w, 1))
        if unit < 4:
            return None
    if np.any(np.abs(w - (origin + unit * k)) > np.array([tolerances[i] for i in order])):
        return None
    rungs = [0] * n
    for rank, i in enumerate(order):
        rungs[i] = int(k[rank] - k[0])
    return rungs


def _ladder_fits(rungs: list[int], where: list[float], options: list[list[tuple[float, bool]]], scale: str):
    """Fits proposed by pairs of equally spaced labels, held to exact steps.

    Tick values step by exactly the same amount from one tick to the next.
    Two labels propose the step; every other label either sits on its rung
    of that ladder of values or does not, and a misreading almost never
    does.  Yields ``(a, b, agree, step)`` with ``value = a * pixel + b``,
    for each agreeing label ``(index, value, doubtful, miss in pixels)``,
    and the step in value from one rung to the next.
    """

    def slack(value: float, step: float) -> float:
        """How far a printed value may be from its rung: rounding in the print, no more."""
        if scale == "log":
            return min(_STEP_TOLERANCE * abs(step), 0.01)
        return min(_STEP_TOLERANCE * abs(step), 0.5 * 10.0 ** -_decimals(value)) + 1e-9 * max(1.0, abs(value))

    seen: set[tuple] = set()
    n = len(rungs)
    for i, j in itertools.combinations(range(n), 2):
        for (t1, _), (t2, _) in itertools.product(options[i], options[j]):
            if t1 == t2:
                continue
            step = (t2 - t1) / (rungs[j] - rungs[i])
            first = t1 - step * rungs[i]
            agree = []
            for m in range(n):
                target = first + step * rungs[m]
                choice = None
                for t, doubtful in options[m]:
                    miss = abs(t - target)
                    if miss <= slack(t, step) and (choice is None or (doubtful, miss) < (choice[2], choice[3])):
                        choice = (m, t, doubtful, miss)
                if choice is not None:
                    agree.append(choice)
            signature = tuple((m, round(t, 9)) for m, t, _, _ in agree)
            if len(agree) < 2 or signature in seen:
                continue
            seen.add(signature)
            # Values on the ladder that best matches the agreeing labels
            # (the same as the labels themselves unless they were rounded).
            ks = np.array([rungs[m] for m, _, _, _ in agree], dtype=float)
            if np.ptp(ks) == 0:
                continue
            step, first = (float(v) for v in np.polyfit(ks, np.array([t for _, t, _, _ in agree]), 1))
            ps = np.array([where[m] for m, _, _, _ in agree])
            a, b = (float(v) for v in np.polyfit(ps, first + step * ks, 1))
            if a == 0:
                continue
            misses = [(m, t, d, abs((first + step * rungs[m] - b) / a - where[m])) for m, t, d, _ in agree]
            yield a, b, misses, step


def _line_fits(
    where: list[float],
    tolerances: list[float],
    options: list[list[tuple[float, bool]]],
    scale: str,
):
    """Fits proposed by pairs of unevenly spaced labels, held to a line within about a pixel.

    Yields ``(a, b, agree, step)`` as :func:`_ladder_fits` does.
    """
    gaps = np.diff(sorted(set(where)))
    gaps = gaps[gaps >= 4]
    spacing_px = float(gaps.min()) if len(gaps) else 0.0

    def consensus(a: float, b: float):
        """Labels agreeing with value = a * pixel + b: (index, value, doubtful, miss in pixels)."""
        agree = []
        for i, opts in enumerate(options):
            choice = None
            for t, doubtful in opts:
                miss = abs((t - b) / a - where[i])
                if miss <= tolerances[i] and (choice is None or (doubtful, miss) < (choice[2], choice[3])):
                    choice = (i, t, doubtful, miss)
            if choice is not None:
                agree.append(choice)
        return agree

    seen: set[tuple] = set()
    for i, j in itertools.combinations(range(len(where)), 2):
        if abs(where[j] - where[i]) < 4:
            continue
        for (t1, _), (t2, _) in itertools.product(options[i], options[j]):
            if t1 == t2:
                continue
            a = (t2 - t1) / (where[j] - where[i])
            b = t1 - a * where[i]
            agree = consensus(a, b)
            # Refine: refit to the agreeing labels and look again.
            for _ in range(3):
                if len(agree) < 2:
                    break
                ps = np.array([where[k] for k, _, _, _ in agree])
                ts = np.array([t for _, t, _, _ in agree])
                if np.ptp(ps) < 4 or np.ptp(ts) == 0:
                    break
                a2, b2 = (float(v) for v in np.polyfit(ps, ts, 1))
                again = consensus(a2, b2)
                if len(again) < len(agree):
                    break
                settled = [(k, t) for k, t, _, _ in again] == [(k, t) for k, t, _, _ in agree]
                a, b, agree = a2, b2, again
                if settled:
                    break
            kept = _on_lattice(agree, a, spacing_px, scale)
            if 2 <= len(kept) < len(agree):
                ps = np.array([where[k] for k, _, _, _ in kept])
                ts = np.array([t for _, t, _, _ in kept])
                if np.ptp(ps) >= 4 and np.ptp(ts) != 0:
                    a, b = (float(v) for v in np.polyfit(ps, ts, 1))
                    agree = [(k, t, d, abs((t - b) / a - where[k])) for k, t, d, _ in kept]
            signature = tuple((k, round(t, 9)) for k, t, _, _ in agree)
            if len(agree) < 2 or signature in seen:
                continue
            seen.add(signature)
            yield a, b, agree, a * spacing_px


def fit_axis(
    labels: list[TickLabel],
    increasing: Optional[bool] = None,
    scales: tuple[str, ...] = ("linear", "log"),
    n_located: Optional[int] = None,
) -> Optional[AxisFit]:
    """Fit the axis scale to tick labels by consensus.

    Every pair of labels proposes a scale, and the proposal that the most
    labels agree with is kept - counting a doubtful reading (a restored
    decimal point, an exponent run together with its base) as half a label,
    then preferring a linear scale and round tick steps.  Where the labels
    are equally spaced, as they nearly always are, agreeing means stepping
    by exactly the same amount from label to label, which a misread label
    almost never does; otherwise it means lying on the fitted line to
    within about a pixel.  Requiring agreement makes the fit immune to
    misread labels, and the number of agreeing labels is reported so that a
    weak calibration can be flagged (:attr:`AxisFit.confidence`).

    Parameters
    ----------
    increasing : bool, optional
        Require values to increase (True) or decrease (False) with the pixel
        coordinate.  A y axis normally decreases down the image; an x axis
        increases to the right.  None accepts either.
    scales : tuple of str
        Scales to try: ``"linear"`` and/or ``"log"``.
    n_located : int, optional
        How many labels were found beside the axis, including any that
        could not be read; used in judging confidence.

    Returns
    -------
    AxisFit or None
        None if fewer than two labels are usable.
    """
    usable = [lab for lab in labels if lab.values]
    if len(usable) < 2:
        return None

    def tolerance(lab: TickLabel) -> float:
        if lab.anchor is not None:
            return 1.5
        x0, y0, x1, y1 = lab.box
        return max(2.5, 0.35 * min(x1 - x0 + 1, y1 - y0 + 1))

    # Rungs are counted over every label found, read or not, so that an
    # unread label between two others still takes its place on the ladder.
    rungs = _rungs([lab.where for lab in labels], [tolerance(lab) for lab in labels])
    if rungs is not None:
        rungs = [r for r, lab in zip(rungs, labels) if lab.values]
    else:
        rungs = _rungs([lab.where for lab in usable], [tolerance(lab) for lab in usable])
    tols = [tolerance(lab) for lab in usable]
    where = [lab.where for lab in usable]

    candidates: list[tuple] = []
    for scale in scales:
        options: list[list[tuple[float, bool]]] = []
        for lab in usable:
            opts = []
            for v, doubtful in zip(lab.values, lab.doubtful):
                if scale == "linear":
                    opts.append((v, doubtful))
                elif v > 0:
                    opts.append((math.log10(v), doubtful))
            options.append(opts)

        if rungs is not None:
            fits = _ladder_fits(rungs, where, options, scale)
        else:
            fits = _line_fits(where, tols, options, scale)
        for a, b, agree, step in fits:
            if increasing is not None and (a > 0) != increasing:
                continue
            ts_list = [t for _, t, _, _ in agree]
            # A log scale needs a real spread of magnitudes to be believed.
            if scale == "log" and (max(ts_list) - min(ts_list) < 0.9 or len(agree) < 3):
                continue
            plain = sum(1 for _, _, doubtful, _ in agree if not doubtful)
            rms = math.sqrt(sum(m * m for _, _, _, m in agree) / len(agree))
            usual = _is_usual_step(step, scale)
            key = (
                plain + 0.5 * (len(agree) - plain),
                plain,
                usual,
                scale == "linear",
                # (between log scales the labels support equally, one that
                # steps by a decade: an exponent misread drops a digit or a
                # sign far more often than it invents one)
                -abs(step) if scale == "log" and step is not None else 0.0,
                _round_steps(ts_list, scale),
                -rms,
            )
            candidates.append((key, scale, a, b, agree, rms, plain, usual))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, scale, a, b, agree, rms, plain, usual = candidates[0]
    rival = _rival(candidates, where)
    inliers = [(float(where[k]), float(10.0 ** t if scale == "log" else t)) for k, t, _, _ in agree]
    return AxisFit(
        a=float(a), b=float(b), scale=scale, inliers=inliers, n_labels=len(usable), rms_px=float(rms),
        anchored=sum(1 for k, _, _, _ in agree if usable[k].anchor is not None),
        exact=rungs is not None, n_plain=plain, n_located=max(n_located or 0, len(labels)), usual_step=usual,
        rival=rival,
    )


def _rival(candidates: list[tuple], where: list[float]) -> Optional[tuple[float, float]]:
    """The values, at the first and last labels, of a different scale that the labels support exactly as well as the
    best one (as many agreeing labels, as many of them plain readings), if there is one."""
    best = candidates[0]
    lo, hi = min(where), max(where)

    def ends(c):
        _, scale, a, b = c[:4]
        v = [a * lo + b, a * hi + b]
        return [10.0 ** x for x in v] if scale == "log" else v

    e0 = ends(best)
    span = abs(e0[1] - e0[0]) or 1.0
    for c in candidates[1:]:
        if c[0][:2] != best[0][:2]:
            break
        e = ends(c)
        if best[1] == "log" and c[1] == "log":
            differ = any(abs(math.log10(max(x, 1e-300)) - math.log10(max(y, 1e-300))) > 0.05 for x, y in zip(e0, e))
        else:
            differ = any(abs(x - y) > 0.01 * span for x, y in zip(e0, e))
        if differ:
            return (float(e[0]), float(e[1]))
    return None


def calibrate_axis(
    rgb: np.ndarray,
    dark: np.ndarray,
    panel: dict,
    axis: str,
    limit: Optional[float] = None,
    scales: tuple[str, ...] = ("linear", "log"),
) -> tuple[Optional[AxisFit], list[TickLabel]]:
    """Find, read and fit the tick labels of one axis.

    Returns ``(fit, labels)``.  *labels* are all the tick labels found
    beside the axis, with whatever was read from each (``readings`` and
    ``values`` are empty for one that could not be read).  *fit* is None if
    fewer than two labels could be read or no two of them agree on a scale.
    Check :attr:`AxisFit.usable` before relying on a fit: one that most
    labels disagree with is returned, so that it can be reported, but
    should not be applied.
    """
    if not tesseract_available():
        return None, []
    located = find_tick_labels(rgb, dark, panel, axis, limit=limit)
    if len(located) < 2:
        return None, located
    if axis == "x" and not labels_are_numeric(rgb, located):
        return None, []  # interval names or words, not a numeric axis
    anchor_labels(located, panel.get("tick_rows" if axis == "y" else "tick_cols", []))

    fit: Optional[AxisFit] = None
    for preparation in PREPARATIONS:
        read_tick_labels(rgb, located, preparation)
        fit = fit_axis(located, increasing=(axis == "x"), scales=scales)
        if fit is not None and fit.exact and len(fit.inliers) == fit.n_total >= 3:
            break  # every label agrees; no need to read them again another way
    if fit is None or not fit.usable:
        # An inverted axis (depth, rank) is unusual but legitimate.  It is
        # also what negative labels look like when their minus signs are not
        # read, so nearly every label has to agree.
        inverted = fit_axis(located, increasing=(axis != "x"), scales=scales)
        if inverted is not None and inverted.confidence != "low" and len(inverted.inliers) >= inverted.n_total - 1:
            fit = inverted
    if fit is not None:
        fit.glyph_px = print_height(located)
    return fit, located


def print_height(labels: list[TickLabel]) -> float:
    """Height in pixels of the print of *labels* (0 if there are none)."""
    if not labels:
        return 0.0
    return float(np.median([lab.box[3] - lab.box[1] + 1 for lab in labels]))
