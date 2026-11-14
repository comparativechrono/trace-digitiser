"""Y-axis calibration: turning pixel rows into data values.

A calibration can come from three places, tried in this order by
:func:`trace_digitiser.digitise`:

1. **The user** - reference points (:func:`calibration_from_points`) or the
   axis limits (:func:`calibration_from_range`).  Always the most reliable.
2. **Tick labels read by OCR** (:func:`calibrate_y`) - the labels beside
   the axis are located, read and fitted by consensus
   (:mod:`trace_digitiser.axis_labels`).  The result carries a confidence.
3. **Nothing** (:func:`normalised_calibration`) - values run from 0 at the
   bottom of the plot area to 1 at the top.

The linear model is ``y_value = a * y_pixel + b``; because image rows
increase downward, a normal axis has a negative ``a``.  A logarithmic axis
uses ``log10(y_value) = a * y_pixel + b``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .axis_labels import _nearest_round, calibrate_axis, print_height, tesseract_available
from .models import Calibration
from .ocr import ocr_number_near_y

# ===================================================================
# Calibrations that need no OCR
# ===================================================================

def _edges(panel: dict) -> dict:
    return panel.get("edges") or {
        "left": float(panel["x0"]), "right": float(panel["x1"]),
        "top": float(panel["y_top"]), "bottom": float(panel["y_bottom"]),
    }


def normalised_calibration(panel: dict) -> Calibration:
    """No real calibration: 0 at the bottom of the plot area, 1 at the top."""
    e = _edges(panel)
    a = -1.0 / max(1e-9, e["bottom"] - e["top"])
    return Calibration(a=a, b=-a * e["bottom"], scale_type="normalised_0_to_1", method="none")


def calibration_from_range(panel: dict, v_min: float, v_max: float, scale: str = "linear") -> Calibration:
    """Calibrate from the axis limits: *v_min* at the bottom edge of the plot area, *v_max* at the top.

    This assumes the axis lines mark the axis limits, which is the usual
    case for a framed plot.  For axes that run past their last tick, use
    :func:`calibration_from_points` with two tick positions instead.
    """
    return calibration_from_points(
        [(_edges(panel)["bottom"], v_min), (_edges(panel)["top"], v_max)], scale=scale, method="axis_limits",
    )


def calibration_from_points(
    points: Sequence[tuple[float, float]],
    scale: str = "linear",
    method: str = "reference_points",
) -> Calibration:
    """Calibrate from ``(pixel, value)`` reference points.

    Two points define the scale; with more, a least-squares line is fitted.
    *pixel* is the image row of a tick (or any known level) and *value* is
    what the axis reads there.
    """
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
    return Calibration(
        a=float(a), b=float(b),
        scale_type=("log_" if scale == "log" else "") + kind,
        inliers=pts, method=method, orientation_flipped=bool(a > 0),
    )


# ===================================================================
# OCR calibration
# ===================================================================

def _plausible_exhaustive(cal: Calibration, panel: dict) -> bool:
    """Sanity check for the exhaustive OCR route, which has no notion of confidence.

    Accept it only if three or more labels agree and they step by a round
    amount, which is what real tick labels do and garbled readings do not.
    """
    if cal.scale_type not in ("ocr_value", "log_ocr_value") or len(cal.inliers) < 3:
        return False
    values = sorted({round(v, 9) for _, v in cal.inliers})
    if len(values) < 3:
        return False
    if cal.is_log:
        if min(values) <= 0:
            return False
        values = [math.log10(v) for v in values]
    step = min(b - a for a, b in zip(values, values[1:]))
    return _nearest_round(step, "log" if cal.is_log else "linear") is not None


def calibrate_y(
    rgb: np.ndarray,
    dark: np.ndarray,
    panel: dict,
    limit: Optional[float] = None,
    scale: str = "auto",
    exhaustive_fallback: bool = False,
    verbose: bool = False,
    axis: str = "y",
) -> Calibration:
    """Calibrate the y axis of one panel from its tick labels.

    Parameters
    ----------
    limit : float, optional
        Column beyond which not to look for labels (the neighbouring panel).
    axis : {"y", "y2"}
        The labels to read: those of the left-hand axis (``"y"``), or those of a second y axis on the right
        (``"y2"``).
    scale : {"auto", "linear", "log"}
        Restrict the fit to one kind of axis; ``"auto"`` tries both.
    exhaustive_fallback : bool
        If the labels cannot be located or do not agree, try the 1.x method
        (many OCR attempts beside each gridline).  Off by default: it takes
        some fifteen seconds a panel, and on axes that the present method
        declines to read it was right in none of twenty trials.  Its result
        is always marked low confidence.

    Returns
    -------
    Calibration
        ``scale_type`` is ``"normalised_0_to_1"`` if no calibration could be
        made.  Check ``confidence`` otherwise.
    """
    scales = {"auto": ("linear", "log"), "linear": ("linear",), "log": ("log",)}.get(scale)
    if scales is None:
        raise ValueError("scale must be 'auto', 'linear' or 'log'")
    if not tesseract_available():
        return normalised_calibration(panel)

    if axis not in ("y", "y2"):
        raise ValueError("axis must be 'y' or 'y2'")
    fit, labels = calibrate_axis(rgb, dark, panel, axis, limit=limit, scales=scales)
    candidates = {int(round(lab.where)): list(lab.values) for lab in labels}
    found = len(labels)
    label_px = print_height(labels) or None

    def from_fit() -> Calibration:
        return Calibration(
            a=fit.a, b=fit.b,
            scale_type="log_ocr_value" if fit.scale == "log" else "ocr_value",
            orientation_flipped=bool(fit.a > 0),
            ocr_candidates=candidates, inliers=fit.inliers,
            confidence=fit.confidence, method="tick_labels",
            n_labels=found, rms_px=fit.rms_px, label_px=label_px,
        )

    if verbose:
        print(f"Panel {panel.get('panel_id', '?')} {'right-hand ' if axis == 'y2' else ''}y-axis labels:")
        for lab in labels:
            print(f"  row {lab.where:7.1f}: read {lab.readings} -> {lab.values}")

    if fit is not None and fit.confidence != "low":
        return from_fit()

    if exhaustive_fallback and axis == "y" and scale != "log" and len(panel.get("gridline_y", [])) >= 2:
        legacy = robust_y_calibration(rgb, panel, verbose=verbose)
        if _plausible_exhaustive(legacy, panel):
            legacy.method = "tick_labels_exhaustive"
            legacy.confidence = "low"
            legacy.n_labels = len([v for v in legacy.ocr_candidates.values() if v])
            return legacy

    if fit is not None and fit.usable:
        return from_fit()  # an axis with two or three labels, two of them agreeing: flagged as low confidence
    # Not calibrated; say what was found so that the caller can report it.
    none = normalised_calibration(panel)
    none.ocr_candidates, none.n_labels, none.label_px = candidates, found, label_px
    return none


# ===================================================================
# Orientation guard
# ===================================================================

def orient_calibration_upward(a: float, b: float, panel: dict) -> tuple[float, float, bool]:
    """Ensure the calibration maps the top of the panel to a higher value.

    Returns ``(a, b, was_flipped)``.
    """
    top_y = panel["y_top"]
    bottom_y = panel["y_bottom"]
    top_val = a * top_y + b
    bottom_val = a * bottom_y + b

    if top_val >= bottom_val:
        return float(a), float(b), False

    new_top_val = max(top_val, bottom_val)
    new_bottom_val = min(top_val, bottom_val)
    new_a = (new_bottom_val - new_top_val) / max(1, bottom_y - top_y)
    new_b = new_top_val - new_a * top_y
    return float(new_a), float(new_b), True


# ===================================================================
# Arithmetic progression for evenly-spaced gridlines
# ===================================================================

def _try_arithmetic_progression(
    ocr: dict[int, list[float]],
    y_lines: list[int],
) -> dict | None:
    """Find the best arithmetic progression through OCR candidates.

    For evenly-spaced gridlines the tick values must satisfy
    ``v_i = v_0 + i * step``.  This constraint is much stronger than
    the pairwise RANSAC and rejects spurious single-digit OCR readings
    (e.g. Tesseract reading "7" from "300") that happen to be
    internally consistent but wrong.

    To avoid small-number OCR noise dominating, progressions are
    compared by ``(score, abs_step, −error)`` — a progression spanning
    a wider value range is preferred when inlier counts are equal,
    since multi-digit readings are harder for Tesseract to fabricate
    from noise than single-digit artifacts.

    Returns a dict with ``a``, ``b``, ``score``, ``total_err``,
    ``inliers``, or None if no progression is found.
    """
    sorted_ys = sorted(y_lines)
    n = len(sorted_ys)
    if n < 3:
        return None

    # Check gridlines are approximately evenly spaced.
    gaps = np.diff(sorted_ys).astype(float)
    if len(gaps) < 2:
        return None
    cv = float(np.std(gaps) / max(1.0, np.mean(gaps)))
    if cv > 0.20:
        return None

    # Try every pair of (gridline_index, candidate_value) as anchors.
    best: dict | None = None

    for i in range(n):
        for v_i in ocr.get(sorted_ys[i], []):
            for j in range(i + 1, n):
                for v_j in ocr.get(sorted_ys[j], []):
                    n_gaps = j - i
                    step = (v_j - v_i) / n_gaps

                    # Reject near-zero steps and positive steps (values
                    # should decrease downward for a normal y-axis).
                    if step >= -0.5:
                        continue

                    # Implied value at the first gridline.
                    v_0 = v_i - i * step

                    # Score: how many gridlines have an OCR candidate
                    # matching the predicted value?
                    score = 0
                    total_err = 0.0
                    inliers: list[tuple[float, float]] = []
                    tol = max(0.5, abs(step) * 0.20)

                    for k in range(n):
                        expected = v_0 + k * step
                        cands = ocr.get(sorted_ys[k], [])
                        if not cands:
                            continue
                        errors = [abs(expected - c) for c in cands]
                        best_k = int(np.argmin(errors))
                        err = errors[best_k]
                        if err <= tol:
                            score += 1
                            total_err += err
                            inliers.append((float(sorted_ys[k]), float(cands[best_k])))

                    if score < 3:
                        continue

                    abs_step = abs(step)
                    if best is None or (score, abs_step, -total_err) > (
                        best["score"],
                        best.get("_abs_step", 0),
                        -best["total_err"],
                    ):
                        # Convert progression to linear calibration.
                        pixel_step = float(np.median(gaps))
                        a = step / pixel_step
                        b = v_0 - a * sorted_ys[0]
                        best = {
                            "a": a, "b": b,
                            "score": score, "total_err": total_err,
                            "inliers": inliers,
                            "_abs_step": abs_step,
                        }

    return best


# ===================================================================
# Robust y-axis calibration
# ===================================================================

def robust_y_calibration(
    rgb: np.ndarray,
    panel: dict,
    verbose: bool = False,
) -> Calibration:
    """Fit a y-pixel → y-value mapping by exhaustive OCR beside each gridline.

    This is the calibration method of versions 1.x.  It makes dozens of
    Tesseract calls per gridline and has no measure of confidence, so
    :func:`calibrate_y` is now tried first and this is kept as a fallback
    for figures whose labels cannot be located geometrically.

    Steps:

    1. OCR numbers near all detected gridlines.
    2. Try candidate line fits between OCR pairs (linear and log).
    3. Score fits by number of inlier gridlines.
    4. Refit from inliers if possible.
    5. Enforce upward orientation.
    6. Fall back to normalised 0–1 scale if OCR is insufficient.

    Log-scale detection: if the OCR'd values are all positive and
    better explained by ``log10(y_value) = a * y_pixel + b``, the
    calibration uses ``scale_type = "log_ocr_value"``.
    """
    y_lines = panel["gridline_y"]
    ocr: dict[int, list[float]] = {int(y): ocr_number_near_y(rgb, panel, int(y)) for y in y_lines}

    pairs: list[tuple[float, float]] = []
    for y, vals in ocr.items():
        for v in vals:
            if abs(v) < 10_000:
                pairs.append((float(y), float(v)))

    best: dict | None = None
    best_log: dict | None = None

    if len(pairs) >= 2:
        for i in range(len(pairs)):
            y1, v1 = pairs[i]
            for j in range(i + 1, len(pairs)):
                y2, v2 = pairs[j]
                if abs(y2 - y1) < 20 or abs(v2 - v1) < 1:
                    continue

                # --- Linear fit ---
                a = (v2 - v1) / (y2 - y1)
                b = v1 - a * y1

                # Normal y-axis: values increase upward → a < 0.
                if a < 0:
                    inliers: list[tuple[float, float]] = []
                    score = 0
                    total_error = 0.0

                    for yy, vals in ocr.items():
                        if not vals:
                            continue
                        pred = a * yy + b
                        errors = [abs(pred - vv) for vv in vals]
                        k = int(np.argmin(errors))
                        err = errors[k]
                        tol = max(18.0, 0.08 * abs(v2 - v1))

                        if err <= tol:
                            score += 1
                            total_error += err
                            inliers.append((float(yy), float(vals[k])))

                    if best is None or (score, -total_error) > (best["score"], -best["total_error"]):
                        best = {
                            "a": a, "b": b,
                            "score": score, "total_error": total_error,
                            "inliers": inliers,
                        }

                # --- Log fit: log10(value) = a_log * y_pixel + b_log ---
                if v1 > 0 and v2 > 0:
                    lv1, lv2 = np.log10(v1), np.log10(v2)
                    if abs(lv2 - lv1) < 0.3:
                        continue
                    a_log = (lv2 - lv1) / (y2 - y1)
                    b_log = lv1 - a_log * y1

                    if a_log >= 0:
                        continue

                    log_inliers: list[tuple[float, float]] = []
                    log_score = 0
                    log_error = 0.0

                    for yy, vals in ocr.items():
                        if not vals:
                            continue
                        pred_log = a_log * yy + b_log
                        pred_val = 10 ** pred_log
                        errors = [abs(pred_val - vv) / max(1e-9, abs(vv)) for vv in vals if vv > 0]
                        if not errors:
                            continue
                        k = int(np.argmin(errors))
                        rel_err = errors[k]

                        if rel_err <= 0.20:  # 20 % relative tolerance
                            log_score += 1
                            log_error += rel_err
                            pos_vals = [vv for vv in vals if vv > 0]
                            log_inliers.append((float(yy), float(pos_vals[k])))

                    if log_score >= 2:
                        if best_log is None or (log_score, -log_error) > (best_log["score"], -best_log["total_error"]):
                            best_log = {
                                "a": a_log, "b": b_log,
                                "score": log_score, "total_error": log_error,
                                "inliers": log_inliers,
                            }

    orientation_flipped = False
    use_log = False

    # --- Arithmetic progression approach (new) ---
    # For evenly-spaced gridlines, this is more robust than pairwise
    # RANSAC because it rejects spurious partial-digit OCR readings.
    prog = _try_arithmetic_progression(ocr, y_lines)
    if prog is not None and prog["score"] >= 3:
        ransac_score = best["score"] if best is not None else 0

        # Prefer progression if it has equal or better score.
        # Also prefer it when scores are close but the progression
        # spans a much larger value range — single-digit OCR artifacts
        # (e.g. "7" from "300") produce small ranges that look
        # internally consistent but are wrong.
        prog_range = max(abs(v) for _, v in prog["inliers"]) if prog["inliers"] else 0
        ransac_range = max(abs(v) for _, v in best["inliers"]) if best and best["inliers"] else 0

        prefer_prog = (
            prog["score"] > ransac_score
            or (prog["score"] == ransac_score and prog_range >= ransac_range)
            or (prog["score"] >= ransac_score - 1 and prog_range > 10 * max(1, ransac_range))
        )

        if prefer_prog:
            best = {
                "a": prog["a"],
                "b": prog["b"],
                "score": prog["score"],
                "total_error": prog["total_err"],
                "inliers": prog["inliers"],
            }

    # Decide between linear and log.
    if best_log is not None and best_log["score"] >= 2:
        linear_score = best["score"] if best is not None else 0
        if best_log["score"] > linear_score or (best_log["score"] == linear_score and best is None):
            use_log = True

    if use_log and best_log is not None:
        inlier_arr = np.array(best_log["inliers"], dtype=float)
        log_vals = np.log10(inlier_arr[:, 1])
        if len(inlier_arr) >= 2:
            a, b = np.polyfit(inlier_arr[:, 0], log_vals, deg=1)
        else:
            a, b = best_log["a"], best_log["b"]
        scale_type = "log_ocr_value"
        # For log scale, "upward" means higher values at top → a < 0.
        if a > 0:
            a, b = -a, -(b)  # flip
            orientation_flipped = True
        best_used = best_log
    elif best is not None and best["score"] >= 2:
        inlier_arr = np.array(best["inliers"], dtype=float)
        if len(inlier_arr) >= 2:
            a, b = np.polyfit(inlier_arr[:, 0], inlier_arr[:, 1], deg=1)
        else:
            a, b = best["a"], best["b"]

        scale_type = "ocr_value"
        a, b, orientation_flipped = orient_calibration_upward(a, b, panel)
        best_used = best
    else:
        a = -1.0 / max(1, panel["y_bottom"] - panel["y_top"])
        b = -a * panel["y_bottom"]
        scale_type = "normalised_0_to_1"
        best_used = {"score": 0, "inliers": []}
        a, b, orientation_flipped = orient_calibration_upward(a, b, panel)

    if verbose:
        print(f"Panel {panel['panel_id']} y-axis OCR candidates:")
        for y in y_lines:
            print(f"  y={y}: {ocr[y]}")
        if scale_type == "log_ocr_value":
            print(f"  calibration: log10(value) = {a:.6g} * y_pixel + {b:.6g}")
            print(f"  top value: {10 ** (a * panel['y_top'] + b):.3g}")
            print(f"  bottom value: {10 ** (a * panel['y_bottom'] + b):.3g}")
        else:
            print(f"  calibration: value = {a:.6g} * y_pixel + {b:.6g}")
            print(f"  top value: {a * panel['y_top'] + b:.3g}")
            print(f"  bottom value: {a * panel['y_bottom'] + b:.3g}")
        print(f"  scale_type: {scale_type}")
        print(f"  inliers: {best_used['inliers']}")

    return Calibration(
        a=float(a),
        b=float(b),
        scale_type=scale_type,
        orientation_flipped=orientation_flipped,
        ocr_candidates=ocr,
        inliers=best_used["inliers"],
        method="none" if scale_type == "normalised_0_to_1" else "tick_labels_exhaustive",
    )


# ===================================================================
# Cross-row propagation
# ===================================================================

def propagate_y_calibration_across_rows(
    panels: list[dict],
    verbose: bool = False,
) -> list[dict]:
    """Copy y-axis calibration from labelled panels to row-aligned siblings.

    In horizontal multi-panel figures only the leftmost subplot may have
    y-axis labels.  This copies its calibration to sibling panels at
    the same y-pixel extents.
    """
    if not panels:
        return panels

    used: set[int] = set()
    row_groups: list[list[dict]] = []

    for i, p in enumerate(panels):
        if i in used:
            continue
        group = [p]
        used.add(i)
        for j, q in enumerate(panels):
            if j in used:
                continue
            same_row = abs(p["y_top"] - q["y_top"]) < 18 and abs(p["y_bottom"] - q["y_bottom"]) < 18
            if same_row:
                group.append(q)
                used.add(j)
        row_groups.append(group)

    for group in row_groups:
        refs = [p for p in group if p["y_calibration"]["scale_type"] == "ocr_value"]
        if not refs:
            continue

        ref = refs[0]
        ref_cal = ref["y_calibration"]
        ref_top_val = ref_cal["a"] * ref["y_top"] + ref_cal["b"]
        ref_bottom_val = ref_cal["a"] * ref["y_bottom"] + ref_cal["b"]

        for p in group:
            # Only panels that have no calibration of their own inherit one.
            if p["y_calibration"]["scale_type"] != "normalised_0_to_1":
                continue

            a = (ref_bottom_val - ref_top_val) / max(1, p["y_bottom"] - p["y_top"])
            b = ref_top_val - a * p["y_top"]

            p["y_calibration"] = {
                **p["y_calibration"],
                "a": float(a),
                "b": float(b),
                "scale_type": f"propagated_from_panel_{ref['panel_id']}",
                "method": "propagated",
                "confidence": ref_cal.get("confidence", ""),
            }

            if verbose:
                print(f"Panel {p['panel_id']}: propagated y calibration from panel {ref['panel_id']}")

    return panels
