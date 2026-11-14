"""X-axis calibration: turning pixel columns into data values.

Mirrors the y axis (:mod:`trace_digitiser.calibration`).  An x calibration
is a dict with ``a``, ``b`` and ``scale_type``; the mapping is
``x_value = a * x_pixel + b``, or ``10 ** (a * x_pixel + b)`` when
``scale_type`` begins with ``"log"``.

Numeric tick labels are read by OCR (:func:`calibrate_x`).  Axes labelled
with words or interval names ("1N", "Mon") cannot be calibrated from the
labels; give the axis limits or reference points instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import cv2
import numpy as np

from .axis_labels import calibrate_axis, tesseract_available
from .ocr import extract_numbers_from_text, preprocess_for_ocr, pytesseract

# ===================================================================
# Calibrations that need no OCR
# ===================================================================

def x_calibration_from_points(
    points: Sequence[tuple[float, float]],
    scale: str = "linear",
    method: str = "reference_points",
) -> dict:
    """Calibrate from ``(pixel, value)`` reference points along the x axis."""
    pts = [(float(px), float(v)) for px, v in points]
    if len(pts) < 2:
        raise ValueError("at least two (pixel, value) reference points are needed")
    px = np.array([q[0] for q in pts])
    values = np.array([q[1] for q in pts])
    if np.ptp(px) < 1e-9:
        raise ValueError("reference points must be at different pixel positions")
    if scale == "log":
        if np.any(values <= 0):
            raise ValueError("reference values on a log axis must be positive")
        targets = np.log10(values)
    elif scale in ("linear", "auto"):
        targets = values
    else:
        raise ValueError("scale must be 'linear' or 'log'")
    if np.ptp(targets) == 0:
        raise ValueError("reference points must have different values")
    a, b = np.polyfit(px, targets, 1)
    kind = "user_supplied" if method == "axis_limits" else "user_reference"
    return {
        "a": float(a), "b": float(b),
        "scale_type": ("log_" if scale == "log" else "") + kind,
        "inliers": pts, "ocr_candidates": {}, "confidence": "", "method": method,
    }


def x_calibration_from_range(panel: dict, v_min: float, v_max: float, scale: str = "linear") -> dict:
    """Calibrate from the axis limits: *v_min* at the left edge of the plot area, *v_max* at the right."""
    e = panel.get("edges") or {"left": float(panel["x0"]), "right": float(panel["x1"])}
    return x_calibration_from_points([(e["left"], v_min), (e["right"], v_max)], scale=scale, method="axis_limits")


def apply_x(x_calib: dict, x_pixel):
    """Convert pixel columns to x values with an x calibration dict."""
    v = x_calib["a"] * np.asarray(x_pixel, dtype=float) + x_calib["b"]
    return np.power(10.0, v) if str(x_calib.get("scale_type", "")).startswith("log") else v


# ===================================================================
# OCR calibration
# ===================================================================

def calibrate_x(
    rgb: np.ndarray,
    dark: np.ndarray,
    panel: dict,
    limit: Optional[float] = None,
    scale: str = "auto",
) -> Optional[dict]:
    """Calibrate the x axis of one panel from numeric tick labels, or return None.

    A calibration is returned only when at least three labels, and at least
    half of those found, agree on a scale (see :attr:`AxisFit.confidence`).
    The stricter rule than for the y axis is deliberate: x axes are often
    labelled with interval names, and a few stray digits must not be
    mistaken for a numeric axis.
    """
    scales = {"auto": ("linear", "log"), "linear": ("linear",), "log": ("log",)}.get(scale)
    if scales is None:
        raise ValueError("scale must be 'auto', 'linear' or 'log'")
    if not tesseract_available():
        return None
    fit, labels = calibrate_axis(rgb, dark, panel, "x", limit=limit, scales=scales)
    if fit is None or fit.confidence == "low":
        return None
    return {
        "a": fit.a, "b": fit.b,
        "scale_type": "log_x_ocr_value" if fit.scale == "log" else "x_ocr_value",
        "inliers": fit.inliers,
        "ocr_candidates": {int(round(lab.where)): list(lab.values) for lab in labels},
        "confidence": fit.confidence, "method": "tick_labels",
        "n_labels": fit.n_total, "rms_px": fit.rms_px,
        "label_px": fit.glyph_px if fit.glyph_px > 0 else None,
    }


# ===================================================================
# Tick detection along x-axis
# ===================================================================

def _detect_x_tick_positions(rgb: np.ndarray, panel: dict) -> list[int]:
    """Find likely x-tick positions from short vertical marks below the
    plot panel's bottom border.

    Returns a sorted list of x-pixel positions.
    """
    h, w = rgb.shape[:2]
    y_bottom = panel["y_bottom"]
    x0, x1 = panel["x0"], panel["x1"]

    # Look in a thin band just below the panel.
    band_top = max(0, y_bottom - 3)
    band_bottom = min(h, y_bottom + 12)
    band = rgb[band_top:band_bottom, x0:x1]

    if band.size == 0:
        return []

    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    # Vertical morphological opening to isolate tick marks.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, band.shape[0] // 2)))
    ticks = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find x positions of tick marks.
    col_sum = ticks.sum(axis=0)
    threshold = col_sum.max() * 0.3 if col_sum.max() > 0 else 1

    from .geometry import merge_contiguous

    active_cols = np.where(col_sum > threshold)[0]
    if len(active_cols) == 0:
        return []

    groups = merge_contiguous(list(active_cols), max_gap=3)
    positions = [x0 + (g[0] + g[1]) // 2 for g in groups]

    return sorted(positions)


def _ocr_number_near_x(
    rgb: np.ndarray,
    panel: dict,
    x: int,
) -> list[float]:
    """OCR numeric labels near x-pixel *x* below the panel."""
    h, w = rgb.shape[:2]
    y_bottom = panel["y_bottom"]

    # Crop a region below the panel centred on x.
    half_w = 35
    left = max(0, x - half_w)
    right = min(w, x + half_w)
    top = min(h, y_bottom + 2)
    bottom = min(h, y_bottom + 55)

    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        return []

    candidates: list[float] = []
    for _, img_variant in preprocess_for_ocr(crop, scale=5):
        for psm in [7, 8, 13]:
            config = f"--psm {psm} -c tessedit_char_whitelist=0123456789.-"
            try:
                text = pytesseract.image_to_string(img_variant, config=config)
            except Exception:
                continue
            for val in extract_numbers_from_text(text):
                if abs(val) < 100_000:
                    candidates.append(round(val, 2))

    # Unique, order-preserving.
    seen: set[float] = set()
    unique: list[float] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# ===================================================================
# Calibration fitting
# ===================================================================

def calibrate_x_axis(
    rgb: np.ndarray,
    panel: dict,
    verbose: bool = False,
) -> Optional[dict]:
    """Attempt to fit a linear x-axis calibration for *panel* (the 1.x method).

    Superseded by :func:`calibrate_x`, which is faster and reports a
    confidence; kept for callers that used it directly.

    Returns a dict with ``a``, ``b``, ``scale_type``, ``inliers``,
    or None if calibration fails.

    The mapping is::

        x_value = a * x_pixel + b
    """
    tick_xs = _detect_x_tick_positions(rgb, panel)

    if len(tick_xs) < 2:
        if verbose:
            print(f"  Panel {panel['panel_id']}: x-axis: only {len(tick_xs)} tick positions found, skipping")
        return None

    # OCR numbers near each tick.
    ocr: dict[int, list[float]] = {}
    for tx in tick_xs:
        vals = _ocr_number_near_x(rgb, panel, tx)
        if vals:
            ocr[tx] = vals

    if len(ocr) < 2:
        if verbose:
            print(f"  Panel {panel['panel_id']}: x-axis: only {len(ocr)} ticks with OCR values, skipping")
        return None

    # Build (x_pixel, x_value) pairs.
    pairs: list[tuple[float, float]] = []
    for xp, vals in ocr.items():
        for v in vals:
            pairs.append((float(xp), float(v)))

    # Try all candidate fits and pick the one with most inliers.
    best: dict | None = None
    for i in range(len(pairs)):
        x1, v1 = pairs[i]
        for j in range(i + 1, len(pairs)):
            x2, v2 = pairs[j]
            if abs(x2 - x1) < 10:
                continue

            a = (v2 - v1) / (x2 - x1)
            b = v1 - a * x1

            # x-axis usually increases left-to-right → a > 0.
            if a <= 0:
                continue

            inliers: list[tuple[float, float]] = []
            score = 0
            total_error = 0.0

            for xp, vals in ocr.items():
                pred = a * xp + b
                errors = [abs(pred - vv) for vv in vals]
                k = int(np.argmin(errors))
                err = errors[k]
                tol = max(5.0, 0.08 * abs(v2 - v1))

                if err <= tol:
                    score += 1
                    total_error += err
                    inliers.append((float(xp), float(vals[k])))

            if best is None or (score, -total_error) > (best["score"], -best["total_error"]):
                best = {
                    "a": a,
                    "b": b,
                    "score": score,
                    "total_error": total_error,
                    "inliers": inliers,
                }

    if best is None or best["score"] < 2:
        if verbose:
            print(f"  Panel {panel['panel_id']}: x-axis: no valid calibration found")
        return None

    # Refit from inliers.
    inlier_arr = np.array(best["inliers"], dtype=float)
    if len(inlier_arr) >= 2:
        a, b = np.polyfit(inlier_arr[:, 0], inlier_arr[:, 1], deg=1)
    else:
        a, b = best["a"], best["b"]

    result = {
        "a": float(a),
        "b": float(b),
        "scale_type": "x_ocr_value",
        "inliers": best["inliers"],
        "ocr_candidates": {int(k): v for k, v in ocr.items()},
    }

    if verbose:
        print(f"  Panel {panel['panel_id']}: x-axis calibration: x_value = {a:.6g} * x_pixel + {b:.6g}")
        print(f"    left value: {a * panel['x0'] + b:.4g}")
        print(f"    right value: {a * panel['x1'] + b:.4g}")
        print(f"    inliers: {best['inliers']}")

    return result


def apply_x_calibration(x_pixel: float, x_calib: Optional[dict]) -> Optional[float]:
    """Convert an x-pixel to an x-value using calibration, or return None."""
    if x_calib is None:
        return None
    return x_calib["a"] * x_pixel + x_calib["b"]
