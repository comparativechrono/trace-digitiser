"""Colour-based trace segmentation.

Detects coloured traces inside a panel by thresholding in HSV colour
space, grouping pixels by hue, and cleaning via connected components.

Improvements over the original notebook:

* **CIELAB clustering** — An optional mode clusters non-background
  pixels in CIELAB colour space (perceptually uniform) using
  mini-batch k-means, which separates traces with similar hues better
  than the HSV histogram approach.
* **Achromatic trace detection** — A separate pass detects black/grey
  traces by looking for low-saturation, non-background, horizontally
  continuous structures (distinct from grid/axis lines).
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import merge_contiguous


# ===================================================================
# Hue clustering (original HSV approach)
# ===================================================================

def circular_hue_groups(
    hues: np.ndarray,
    n_bins: int = 36,
    min_frac_of_peak: float = 0.18,
    min_pixels: int = 80,
) -> list[dict]:
    """Group hue values into dominant colour groups with wrap-around handling."""
    if len(hues) == 0:
        return []

    hist, _ = np.histogram(hues, bins=n_bins, range=(0, 1))
    if hist.max() < min_pixels:
        return []

    active = hist >= max(min_pixels, min_frac_of_peak * hist.max())
    visited = np.zeros(n_bins, dtype=bool)
    groups: list[list[int]] = []

    for i in range(n_bins):
        if visited[i] or not active[i]:
            continue

        group = [i]
        visited[i] = True

        j = (i + 1) % n_bins
        while j != i and active[j] and not visited[j]:
            group.append(j)
            visited[j] = True
            j = (j + 1) % n_bins

        groups.append(group)

    # Merge wrap-around red groups.
    if len(groups) >= 2 and 0 in groups[0] and (n_bins - 1) in groups[-1]:
        groups = [groups[-1] + groups[0]] + groups[1:-1]

    ranges: list[dict] = []
    for g in groups:
        ranges.append(
            {
                "bins": g,
                "lo": min(g) / n_bins,
                "hi": (max(g) + 1) / n_bins,
                "n_bins": n_bins,
            }
        )

    return ranges


def hue_mask_from_group(H: np.ndarray, hue_group: dict) -> np.ndarray:
    """Create a boolean mask for pixels belonging to a hue group."""
    n_bins = hue_group["n_bins"]
    idx = np.floor(H * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    return np.isin(idx, hue_group["bins"])


# ===================================================================
# CIELAB clustering (improvement)
# ===================================================================

def _cluster_lab(
    roi_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    max_clusters: int = 6,
    min_cluster_pixels: int = 80,
) -> list[np.ndarray]:
    """Cluster foreground pixels in CIELAB space via mini-batch k-means.

    Returns a list of boolean masks (same shape as *foreground_mask*),
    one per detected cluster.
    """
    if foreground_mask.sum() < min_cluster_pixels:
        return []

    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_pixels = lab[foreground_mask]  # shape (N, 3)

    # Adaptive cluster count: try several k and pick via silhouette-like heuristic.
    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    best_labels = None
    best_k = 1

    for k in range(2, min(max_clusters + 1, len(fg_pixels) // min_cluster_pixels + 1)):
        km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(1024, len(fg_pixels)))
        labels = km.fit_predict(fg_pixels)

        # Keep only clusters with enough pixels.
        counts = np.bincount(labels, minlength=k)
        n_valid = (counts >= min_cluster_pixels).sum()

        if n_valid >= 2:
            best_labels = labels
            best_k = k

    if best_labels is None:
        # Single cluster: everything is one trace.
        return [foreground_mask.copy()]

    masks: list[np.ndarray] = []
    full_labels = np.full(foreground_mask.shape, -1, dtype=int)
    full_labels[foreground_mask] = best_labels

    for c in range(best_k):
        m = full_labels == c
        if m.sum() >= min_cluster_pixels:
            masks.append(m)

    return masks


# ===================================================================
# Achromatic (black/grey) trace detection (improvement)
# ===================================================================

def _detect_achromatic_traces(
    roi_rgb: np.ndarray,
    existing_masks: list[np.ndarray],
    panel_height: int,
    min_col_fraction: float = 0.20,
    min_pixels: int = 100,
) -> list[np.ndarray]:
    """Detect dark, low-saturation traces not caught by colour segmentation.

    Strategy: find non-background, low-saturation pixels that are NOT
    covered by any existing colour mask, then keep those with enough
    horizontal continuity to be a trace (as opposed to axis lines or
    text, which tend to be sparser or thinner).
    """
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Dark-to-medium, unsaturated pixels.
    candidate = (S < 0.15) & (V > 0.05) & (V < 0.65)

    # Exclude anything already claimed by a coloured trace.
    for m in existing_masks:
        candidate = candidate & ~m

    # Exclude very thin horizontal structures (likely gridlines/axes):
    # morphological open with a tall narrow kernel removes thin horizontals.
    candidate_u8 = candidate.astype(np.uint8) * 255
    # Remove thin horizontal lines (≤3px tall).
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, k_open)

    # Also remove tall thin vertical structures (tick marks).
    k_open2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, k_open2)

    candidate = candidate_u8 > 0

    # Check horizontal continuity: a trace should span a significant
    # fraction of the panel width.
    col_counts = candidate.sum(axis=0)  # per-column pixel count
    active_cols = (col_counts > 0).sum()

    if active_cols < min_col_fraction * w:
        return []

    if candidate.sum() < min_pixels:
        return []

    # Clean via connected components.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(candidate, dtype=bool)
    for lab in range(1, n_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        comp_w = stats[lab, cv2.CC_STAT_WIDTH]
        if area >= min_pixels and comp_w >= min_col_fraction * w:
            cleaned |= labels == lab

    if cleaned.sum() < min_pixels:
        return []

    return [cleaned]


# ===================================================================
# RGB refinement
# ===================================================================

def _refine_mask_by_colour(
    roi_rgb: np.ndarray,
    initial_mask: np.ndarray,
    foreground: np.ndarray,
    max_rgb_distance: float = 85.0,
    min_component_pixels: int = 20,
) -> np.ndarray:
    """Expand a trace mask to include colour-similar pixels.

    The initial HSV hue grouping can be too narrow — anti-aliased or
    JPEG-compressed pixels at trace edges may fall outside the detected
    hue range or below the saturation threshold.  This function recovers
    them by finding *any* pixel in the ROI whose RGB colour is close to
    the trace's median, keeping only connected components that overlap
    the initial mask (to avoid adding isolated noise).

    Parameters
    ----------
    roi_rgb : ndarray, shape (H, W, 3)
        Panel ROI in RGB.
    initial_mask : ndarray, shape (H, W), bool
        The mask from hue grouping.
    foreground : ndarray, shape (H, W), bool
        The saturation-thresholded foreground (used as a fallback; the
        main search runs on all non-background pixels).
    max_rgb_distance : float
        Maximum Euclidean RGB distance from the trace median.
    min_component_pixels : int
        Minimum connected-component size to keep.

    Returns
    -------
    ndarray, shape (H, W), bool
        Refined mask.
    """
    if initial_mask.sum() < 50:
        return initial_mask

    # Median colour of the core trace.
    median_rgb = np.median(roi_rgb[initial_mask], axis=0).astype(np.float32)

    # Adaptive distance: scale with the trace's saturation.
    # Highly saturated traces (deep blue) tolerate a wider search;
    # pale traces (light pink) need a tighter threshold to avoid
    # catching background pixels with similar subtle tints.
    hsv_check = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    trace_median_sat = float(np.median(hsv_check[:, :, 1][initial_mask]))
    sat_scale = np.clip(trace_median_sat / 80.0, 0.4, 1.5)  # scale up for vivid traces
    effective_distance = max_rgb_distance * sat_scale

    # RGB distance of every pixel to the trace median.
    diff = roi_rgb.astype(np.float32) - median_rgb
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    # Exclude obvious background: very bright (white/near-white) and very dark.
    gray = np.mean(roi_rgb.astype(np.float32), axis=2)
    not_background = (gray > 25) & (gray < 235)

    # Require a minimum saturation to exclude truly unsaturated background.
    S = hsv_check[:, :, 1] / 255.0
    min_sat = max(0.04, trace_median_sat / 255.0 * 0.25)
    has_some_colour = S > min_sat

    # Candidate expansion: non-background, somewhat-saturated pixels within colour distance.
    candidates = not_background & has_some_colour & (distance < effective_distance)

    # Merge with initial mask.
    merged = initial_mask | candidates

    # Keep only connected components that overlap the initial mask.
    merged_u8 = merged.astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged_u8, connectivity=8)

    refined = np.zeros_like(merged, dtype=bool)
    for lab in range(1, n_labels):
        component = labels == lab
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < min_component_pixels:
            continue
        # Must overlap with the original core mask.
        if (component & initial_mask).any():
            refined |= component

    return refined


# ===================================================================
# Main trace detection
# ===================================================================

def detect_trace_masks(
    rgb: np.ndarray,
    panel: dict,
    use_lab: bool = False,
) -> list[dict]:
    """Detect coloured trace pixels within one panel.

    Parameters
    ----------
    rgb : ndarray, shape (H, W, 3)
        Full image in RGB.
    panel : dict
        Panel dict with ``x0``, ``x1``, ``y_top``, ``y_bottom``.
    use_lab : bool
        If True, use CIELAB k-means clustering instead of HSV
        histogram binning.  Requires scikit-learn.

    Returns
    -------
    list of dict
        Each dict has ``trace_id``, ``mask``, ``pixel_count``,
        ``median_rgb``, ``hue_group``.
    """
    x0, x1 = panel["x0"], panel["x1"]
    y0, y1 = panel["y_top"], panel["y_bottom"]

    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    H = hsv[:, :, 0] / 180.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    # Saturated, non-background pixels.
    coloured = (S > 0.13) & (V > 0.16)

    # Remove tiny specks.
    coloured_u8 = coloured.astype(np.uint8) * 255
    coloured_u8 = cv2.morphologyEx(coloured_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    coloured = coloured_u8 > 0

    masks: list[dict] = []

    # ------------------------------------------------------------------
    # Colour segmentation
    # ------------------------------------------------------------------
    if use_lab and coloured.sum() > 200:
        try:
            cluster_masks = _cluster_lab(roi, coloured)
        except ImportError:
            cluster_masks = None

        if cluster_masks:
            for idx, cm in enumerate(cluster_masks, start=1):
                if cm.sum() < 80:
                    continue
                median_rgb = np.median(roi[cm], axis=0).astype(int).tolist()
                masks.append(
                    {
                        "trace_id": idx,
                        "mask": cm,
                        "pixel_count": int(cm.sum()),
                        "median_rgb": median_rgb,
                        "hue_group": (0.0, 1.0),
                    }
                )

    if not masks:
        # Fall back to HSV hue grouping.
        hue_groups = circular_hue_groups(H[coloured])

        if not hue_groups and coloured.sum() > 100:
            hue_groups = [{"bins": list(range(36)), "lo": 0.0, "hi": 1.0, "n_bins": 36}]

        for idx, hg in enumerate(hue_groups, start=1):
            mask = coloured & hue_mask_from_group(H, hg)

            # Remove small connected components.
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            cleaned = np.zeros_like(mask, dtype=bool)
            for lab in range(1, n_labels):
                area = stats[lab, cv2.CC_STAT_AREA]
                if area >= 45:
                    cleaned |= labels == lab

            if cleaned.sum() < 80:
                continue

            # Refine: recover anti-aliased / compressed edge pixels
            # that fell outside the narrow hue bin but are colour-similar.
            cleaned = _refine_mask_by_colour(roi, cleaned, coloured)

            median_rgb = np.median(roi[cleaned], axis=0).astype(int).tolist()
            masks.append(
                {
                    "trace_id": idx,
                    "mask": cleaned,
                    "pixel_count": int(cleaned.sum()),
                    "median_rgb": median_rgb,
                    "hue_group": (hg["lo"], hg["hi"]),
                }
            )

    # ------------------------------------------------------------------
    # Achromatic trace detection (black/grey traces)
    # ------------------------------------------------------------------
    existing_colour_masks = [m["mask"] for m in masks]
    achromatic = _detect_achromatic_traces(
        roi, existing_colour_masks, panel_height=y1 - y0
    )
    for am in achromatic:
        next_id = max((m["trace_id"] for m in masks), default=0) + 1
        median_rgb = np.median(roi[am], axis=0).astype(int).tolist()
        masks.append(
            {
                "trace_id": next_id,
                "mask": am,
                "pixel_count": int(am.sum()),
                "median_rgb": median_rgb,
                "hue_group": (0.0, 0.0),  # achromatic sentinel
            }
        )

    return masks
