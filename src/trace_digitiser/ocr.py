"""OCR utilities for tick-label reading and x-label detection.

Uses Tesseract (via pytesseract) with multiple preprocessing variants
and PSM modes for robustness on small, low-resolution tick labels.
"""

from __future__ import annotations

import cv2
import numpy as np


class _LazyTesseract:
    """pytesseract, imported when it is first used: the package can be imported (and used with axis values given
    by hand) where pytesseract is not installed."""

    def __getattr__(self, name):
        import pytesseract

        return getattr(pytesseract, name)


pytesseract = _LazyTesseract()


class Output:
    """pytesseract's output types, without importing it."""

    BYTES = "bytes"
    DATAFRAME = "data.frame"
    DICT = "dict"
    STRING = "string"

# ===================================================================
# Preprocessing
# ===================================================================

def preprocess_for_ocr(
    crop: np.ndarray,
    scale: int = 6,
) -> list[tuple[str, np.ndarray]]:
    """Create OCR-ready variants of an image crop.

    Uses Lanczos interpolation for sharper upscaling and applies
    unsharp masking before binarisation.  Returns grayscale,
    sharpened+Otsu, and sharpened+adaptive variants.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    # Unsharp mask: sharpen edges blurred by upscaling.
    blurred = cv2.GaussianBlur(big, (0, 0), 1.0)
    sharp = cv2.addWeighted(big, 1.5, blurred, -0.5, 0)

    variants: list[tuple[str, np.ndarray]] = [("sharp", sharp)]

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("sharp_otsu", otsu))

    adaptive = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 7)
    variants.append(("sharp_adaptive", adaptive))

    # Also try Otsu without sharpening (sometimes better for clean images).
    _, otsu_plain = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu_plain))

    return variants


# ===================================================================
# Number parsing
# ===================================================================

def extract_numbers_from_text(text: str) -> list[float]:
    """Extract numeric values from OCR text without regex."""
    out: list[float] = []
    current = ""
    for ch in str(text):
        if ch.isdigit() or ch in ".-+":
            current += ch
        else:
            if current not in ("", ".", "-", "+", "-.", "+."):
                try:
                    out.append(float(current))
                except ValueError:
                    pass
            current = ""
    if current not in ("", ".", "-", "+", "-.", "+."):
        try:
            out.append(float(current))
        except ValueError:
            pass
    return out


# ===================================================================
# Y-axis tick OCR
# ===================================================================

def ocr_number_near_y(
    rgb: np.ndarray,
    panel: dict,
    y: int,
) -> list[float]:
    """Attempt to OCR numeric tick labels near a horizontal gridline.

    Crops a band immediately to the left of the panel at height *y* and
    tries multiple crop heights, scale factors, preprocessing variants,
    and PSM modes for robustness on small, low-resolution tick labels.

    Improvements over the original notebook:

    * Tighter crops (±8 to ±14 px) added alongside wider ones — these
      isolate a single tick label without neighbouring text.
    * Scale factor adapts to crop size — smaller crops get higher
      magnification.
    * Lanczos interpolation and unsharp masking improve edge clarity.
    """
    h, w = rgb.shape[:2]
    x0 = panel["x0"]

    left = max(0, int(x0 - 0.12 * w))
    right = max(1, int(x0 - 4))

    candidates: list[float] = []

    # Tight crops first (best for isolating small labels), then wider.
    for half_height in [10, 14, 20, 28]:
        top = max(0, int(y - half_height))
        bottom = min(h, int(y + half_height))
        crop = rgb[top:bottom, left:right]

        if crop.size == 0 or crop.shape[0] < 4:
            continue

        # Adapt scale to crop size: smaller crops need more magnification.
        crop_h = crop.shape[0]
        scale = 10 if crop_h < 20 else 8 if crop_h < 35 else 6

        for _, img_variant in preprocess_for_ocr(crop, scale=scale):
            for psm in [7, 13]:
                config = f"--psm {psm} -c tessedit_char_whitelist=0123456789.-"
                try:
                    text = pytesseract.image_to_string(img_variant, config=config)
                except Exception:
                    continue
                for val in extract_numbers_from_text(text):
                    if abs(val) < 10_000:
                        candidates.append(round(val, 1))

    # Unique while preserving order.
    unique: list[float] = []
    for val in candidates:
        if val not in unique:
            unique.append(val)
    return unique


# ===================================================================
# X-label OCR
# ===================================================================

def ocr_text_band(
    rgb: np.ndarray,
    left: float,
    top: float,
    right: float,
    bottom: float,
    psm: int = 11,
) -> list[dict]:
    """Run Tesseract OCR on a rectangular band, returning text boxes."""
    h, w = rgb.shape[:2]
    left_i, right_i = max(0, int(left)), min(w, int(right))
    top_i, bottom_i = max(0, int(top)), min(h, int(bottom))
    crop = rgb[top_i:bottom_i, left_i:right_i]

    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    scale = 2.5
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    try:
        data = pytesseract.image_to_data(big, output_type=Output.DATAFRAME, config=f"--psm {psm}")
    except Exception:
        return []
    data = data.dropna(subset=["text"])

    items: list[dict] = []
    for _, row in data.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        try:
            conf = float(row["conf"])
        except Exception:
            conf = -1

        if conf < 25:
            continue

        x = left_i + (float(row["left"]) + float(row["width"]) / 2) / scale
        y = top_i + (float(row["top"]) + float(row["height"]) / 2) / scale

        items.append(
            {
                "text": text,
                "conf": conf,
                "x": x,
                "y": y,
                "left": left_i + float(row["left"]) / scale,
                "top": top_i + float(row["top"]) / scale,
                "width": float(row["width"]) / scale,
                "height": float(row["height"]) / scale,
            }
        )

    return items


def clean_axis_label_text(raw_text: str) -> str:
    """Keep only simple label characters (letters, digits, ``_+/-``)."""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+/-")
    return "".join(ch for ch in str(raw_text) if ch in allowed)


def label_has_letter_or_symbol(text: str) -> bool:
    """Reject labels that are plain numbers (likely y-axis tick values)."""
    for ch in text:
        if ch.isalpha() or ch in "+/-_":
            return True
    return False


# Common OCR character confusions (letter ↔ digit).
_OCR_LETTER_TO_DIGIT = {"S": "5", "s": "5", "O": "0", "o": "0", "I": "1",
                         "l": "1", "Z": "2", "z": "2", "B": "8", "G": "6"}


def _fix_ocr_confusions(labels: list[dict]) -> list[dict]:
    """Correct common OCR digit/letter confusions using pattern consensus.

    If the majority of labels follow a ``<digits><letters>`` pattern
    (e.g. ``1N``, ``2D``, ``7N``), any label whose leading characters
    are *letters* that look like digits (S→5, O→0, etc.) is corrected
    to match the dominant pattern.  This fixes the common Tesseract
    confusion of reading ``5`` as ``S`` at low resolution.
    """
    if len(labels) < 3:
        return labels

    def _split_digit_alpha(text: str) -> tuple[str, str] | None:
        """Split text into a leading digit part and trailing alpha part."""
        i = 0
        while i < len(text) and text[i].isdigit():
            i += 1
        if i == 0 or i == len(text):
            return None
        head, tail = text[:i], text[i:]
        if tail.isalpha():
            return head, tail
        return None

    # Check how many labels match <digits><letters> pattern.
    n_match = sum(1 for label in labels if _split_digit_alpha(label["text"]) is not None)

    if n_match < len(labels) * 0.6:
        return labels  # Pattern not dominant enough to apply corrections.

    corrected: list[dict] = []
    for label in labels:
        text = label["text"]
        if _split_digit_alpha(text) is not None:
            corrected.append(label)
            continue

        # Try substituting leading letters that look like digits.
        new_text = list(text)
        changed = False
        for i, ch in enumerate(new_text):
            if ch.isdigit():
                break  # Reached the digit part, stop.
            if ch in _OCR_LETTER_TO_DIGIT:
                new_text[i] = _OCR_LETTER_TO_DIGIT[ch]
                changed = True
            else:
                break  # Non-substitutable letter at the start, give up.

        if changed:
            fixed = "".join(new_text)
            if _split_digit_alpha(fixed) is not None:
                label = dict(label)
                label["text"] = fixed
        corrected.append(label)

    return corrected


def detect_x_labels(
    rgb: np.ndarray,
    panel: dict,
    max_y: int | None = None,
) -> list[dict]:
    """Detect x-axis labels for a panel by OCR-ing top and bottom bands.

    Parameters
    ----------
    rgb : ndarray
        Full image in RGB.
    panel : dict
        Panel dict.
    max_y : int, optional
        Maximum y-pixel for the lower scan band.  When panels are
        stacked this should be set to the top of the next panel so
        that title text between panels is not picked up as x-labels.
    """
    panel_height = panel["y_bottom"] - panel["y_top"]
    img_h = rgb.shape[0]

    # Cap the lower band: use the smaller of a proportional margin
    # (25 % of panel height) or the space to the next panel / image edge.
    lower_margin = int(0.25 * panel_height)
    lower_limit = min(img_h, panel["y_bottom"] + lower_margin)
    if max_y is not None:
        lower_limit = min(lower_limit, max_y)

    bands = [
        (
            panel["x0"],
            panel["y_top"],
            panel["x1"],
            min(panel["y_bottom"], panel["y_top"] + 0.18 * panel_height),
        ),
        (
            panel["x0"],
            max(0, panel["y_bottom"] - 0.12 * panel_height),
            panel["x1"],
            lower_limit,
        ),
    ]

    items: list[dict] = []
    for band in bands:
        items.extend(ocr_text_band(rgb, *band, psm=11))

    cleaned: list[dict] = []
    for it in items:
        text = clean_axis_label_text(it["text"])
        if len(text) == 0 or len(text) > 6:
            continue
        if not label_has_letter_or_symbol(text):
            continue
        # Reject genotype-like annotations: +/+, -/-, +/-, etc.
        stripped = text.replace("/", "").replace("+", "").replace("-", "")
        if len(stripped) == 0:
            continue
        if panel["x0"] <= it["x"] <= panel["x1"]:
            it = dict(it)
            it["text"] = text
            cleaned.append(it)

    cleaned = sorted(cleaned, key=lambda d: (d["x"], -d["conf"]))

    # --- Y-position consistency filter ---
    # Real axis labels cluster at a consistent y. Reject outliers.
    if len(cleaned) >= 3:
        ys = np.array([it["y"] for it in cleaned])
        median_y = float(np.median(ys))
        tol = max(15, 0.15 * panel_height)
        cleaned = [it for it in cleaned if abs(it["y"] - median_y) < tol]

    deduped: list[dict] = []
    for it in cleaned:
        too_close = False
        for prev in deduped:
            if abs(it["x"] - prev["x"]) < 25 and it["text"] == prev["text"]:
                too_close = True
                break
        if not too_close:
            deduped.append(it)

    deduped = _fix_ocr_confusions(deduped)
    return sorted(deduped, key=lambda d: d["x"])
