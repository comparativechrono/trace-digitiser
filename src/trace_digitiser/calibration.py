"""Y-axis calibration: OCR-based linear fitting and cross-panel propagation.

The calibration model is::

    y_value = a * y_pixel + b

Because image y-coordinates increase downward, normal scientific axes
will have a negative ``a``.
"""

from __future__ import annotations

import numpy as np

from .models import Calibration
from .ocr import ocr_number_near_y


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
                    if best is None or (score, abs_step, -total_err) > (best["score"], best.get("_abs_step", 0), -best["total_err"]):
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
    """Fit a linear y-pixel → y-value mapping for one panel.

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
            if p["y_calibration"]["scale_type"] == "ocr_value":
                continue

            a = (ref_bottom_val - ref_top_val) / max(1, p["y_bottom"] - p["y_top"])
            b = ref_top_val - a * p["y_top"]

            p["y_calibration"] = {
                **p["y_calibration"],
                "a": float(a),
                "b": float(b),
                "scale_type": f"propagated_from_panel_{ref['panel_id']}",
            }

            if verbose:
                print(f"Panel {p['panel_id']}: propagated y calibration from panel {ref['panel_id']}")

    return panels
