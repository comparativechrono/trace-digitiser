"""Trace segmentation: which pixels of a plot area belong to which trace.

Coloured traces are found by thresholding saturation, grouping pixels by
hue, and recovering the paler anti-aliased pixels at each trace's edges.
Black and grey traces are found separately, as dark neutral ink that is
neither an axis line, a gridline, a tick mark nor text.

All size thresholds scale with the plot area, so the same code works on a
240-pixel thumbnail and on a 3000-pixel scan, and nothing here erodes the
image, so one-pixel-wide lines survive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import cv2
import numpy as np

from .frames import INK_THRESHOLD, _saturation, darkness_map

# ===================================================================
# Hue clustering
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
# CIELAB clustering (optional; needs scikit-learn)
# ===================================================================

def _cluster_lab(
    roi_rgb: np.ndarray,
    foreground_mask: np.ndarray,
    max_clusters: int = 6,
    min_cluster_pixels: int = 80,
) -> list[np.ndarray]:
    """Cluster foreground pixels in CIELAB space via mini-batch k-means.

    Returns a list of boolean masks (same shape as *foreground_mask*),
    one per detected cluster.  Requires scikit-learn
    (``pip install trace-digitiser[lab]``).
    """
    if foreground_mask.sum() < min_cluster_pixels:
        return []

    from sklearn.cluster import MiniBatchKMeans  # type: ignore

    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_pixels = lab[foreground_mask]  # shape (N, 3)

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
# Helpers
# ===================================================================

def parse_colour(colour) -> tuple[int, int, int]:
    """Turn ``"#rrggbb"``, ``"rrggbb"`` or an ``(r, g, b)`` sequence into an RGB tuple."""
    if isinstance(colour, str):
        text = colour.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6:
            raise ValueError(f"cannot read colour {colour!r}; use '#rrggbb' or (r, g, b)")
        try:
            return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"cannot read colour {colour!r}; use '#rrggbb' or (r, g, b)") from exc
    values = tuple(int(v) for v in colour)
    if len(values) != 3 or any(not 0 <= v <= 255 for v in values):
        raise ValueError(f"cannot read colour {colour!r}; use '#rrggbb' or (r, g, b) with values 0-255")
    return values  # type: ignore[return-value]


def _interior_mask(
    panel: dict,
    shape: tuple[int, int],
    on_axes: bool = False,
    saturation: np.ndarray | None = None,
) -> np.ndarray:
    """True where a trace may be.

    By default this is strictly inside the axis lines, with a pixel to
    spare, which is what a black or grey trace needs: nothing else separates
    it from the axis.  With *on_axes* the axis lines themselves are allowed
    too, for a coloured trace that can be told from them by its colour - so
    a trace resting on the x axis is followed all the way down to it.  An
    axis line that is itself coloured (judged from *saturation*) is always
    kept out.
    """
    h, w = shape
    edges = panel.get("edges")
    interior = np.ones(shape, dtype=bool)
    if edges is None:
        return interior
    half = 0.5 * panel.get("line_thickness", 1)
    strict = half + 1.0
    x0, y0 = panel["x0"], panel["y_top"]
    cols = np.arange(w) + x0
    rows = np.arange(h) + y0

    def margin(on_line: np.ndarray, along_rows: bool) -> float:
        """How far inside an edge to stop: beyond the line for a grey line, short of it otherwise."""
        if not on_axes:
            return strict
        if saturation is not None and on_line.any():
            band = saturation[on_line, :] if along_rows else saturation[:, on_line]
            if band.size and float(np.median(band)) > 0.2:
                return strict  # a coloured axis line would be taken for trace
        return -(half + 0.5)

    near = half + 0.5
    m = margin(np.abs(cols - edges["left"]) <= near, along_rows=False)
    interior[:, cols <= edges["left"] + m] = False
    m = margin(np.abs(rows - edges["bottom"]) <= near, along_rows=True)
    interior[rows >= edges["bottom"] - m, :] = False
    if panel.get("has_right", True):
        m = margin(np.abs(cols - edges["right"]) <= near, along_rows=False)
        interior[:, cols >= edges["right"] - m] = False
    if panel.get("has_top", True):
        m = margin(np.abs(rows - edges["top"]) <= near, along_rows=True)
        interior[rows <= edges["top"] + m, :] = False
    return interior


def _apply_exclusions(mask: np.ndarray, panel: dict, exclude) -> None:
    """Clear user-supplied rectangles ``(x_left, y_top, x_right, y_bottom)`` from *mask* in place."""
    if not exclude:
        return
    h, w = mask.shape
    for box in exclude:
        if len(box) != 4:
            raise ValueError("each exclude region must be (x_left, y_top, x_right, y_bottom)")
        xa, ya, xb, yb = (int(round(float(v))) for v in box)
        xa, xb = sorted((xa - panel["x0"], xb - panel["x0"]))
        ya, yb = sorted((ya - panel["y_top"], yb - panel["y_top"]))
        mask[max(0, ya) : max(0, min(h, yb + 1)), max(0, xa) : max(0, min(w, xb + 1))] = False


def _min_pixels(shape: tuple[int, int]) -> int:
    """Smallest blob worth keeping, scaled to the plot area."""
    return int(np.clip(8e-5 * shape[0] * shape[1], 6, 40))


def _drop_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask
    keep = np.zeros(n_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
    return keep[labels]


def _drop_stray_groups(mask: np.ndarray, reach: int) -> np.ndarray:
    """Remove ink that is far from the trace (legend samples, coloured text).

    Pixels within *reach* of each other are treated as one group.  If one
    group clearly is the trace (it holds most of the pixels), small groups
    elsewhere are dropped.  A dotted or heavily broken trace has no dominant
    group, and is left untouched.
    """
    total = int(mask.sum())
    if total == 0:
        return mask
    k = 2 * reach + 1
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8))
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    if n_labels <= 2:
        return mask
    counts = np.bincount(labels[mask], minlength=n_labels)
    main = int(counts.max())
    if main < 0.7 * total:
        return mask
    keep = counts >= 0.15 * main
    keep[0] = False
    return mask & keep[labels]


def _drop_halos(masks: list[dict], saturation: np.ndarray, reach: int) -> list[dict]:
    """Remove "traces" that are only the pale fringe of a real trace.

    JPEG compression and pixel noise leave a halo of off-colour pixels
    around a strongly coloured line.  Their hue can differ enough from the
    line's to form a group of their own, but they give themselves away: they
    cling to the line, and they are much paler than it.
    """
    if len(masks) < 2:
        return masks
    k = 2 * reach + 1
    kernel = np.ones((k, k), np.uint8)
    order = sorted(masks, key=lambda m: -m["pixel_count"])
    kept: list[dict] = []
    for m in order:
        paleness = float(np.median(saturation[m["mask"]]))
        is_halo = False
        for parent in kept:
            near = cv2.dilate(parent["mask"].astype(np.uint8), kernel) > 0
            clings = (m["mask"] & near).sum() >= 0.7 * m["pixel_count"]
            if clings and paleness < 0.6 * float(np.median(saturation[parent["mask"]])):
                is_halo = True
                break
        if not is_halo:
            kept.append(m)
    kept.sort(key=lambda m: m["trace_id"])
    for i, m in enumerate(kept, start=1):
        m["trace_id"] = i
    return kept


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

    The initial HSV hue grouping can be too narrow - anti-aliased or
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
        Pixels that may be added (the plot interior).
    max_rgb_distance : float
        Maximum Euclidean RGB distance from the trace median.
    min_component_pixels : int
        Minimum connected-component size to keep.

    Returns
    -------
    ndarray, shape (H, W), bool
        Refined mask.
    """
    if initial_mask.sum() < 2 * min_component_pixels:
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
    candidates = foreground & not_background & has_some_colour & (distance < effective_distance)

    # Merge with initial mask.
    merged = initial_mask | candidates

    # Keep only connected components that overlap the initial mask.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return initial_mask
    touches = np.zeros(n_labels, dtype=bool)
    touches[np.unique(labels[initial_mask])] = True
    touches &= stats[:, cv2.CC_STAT_AREA] >= min_component_pixels
    touches[0] = False
    return touches[labels]


# ===================================================================
# Black / grey traces
# ===================================================================

def _remove_gridlines(mask: np.ndarray, panel: dict) -> np.ndarray:
    """Erase gridlines from a mask but keep the places where a trace crosses them."""
    out = mask.copy()
    h, w = mask.shape
    half = panel.get("line_thickness", 1) // 2 + 1
    spread = np.ones(3, dtype=np.uint8)

    for r_abs in panel.get("gridline_y", []):
        r = int(r_abs) - panel["y_top"]
        a, b = r - half, r + half
        if a <= 0 or b >= h - 1:
            continue  # the frame itself; already outside the interior
        above = cv2.dilate(mask[a - 1].astype(np.uint8)[None, :], spread[None, :])[0] > 0
        below = cv2.dilate(mask[b + 1].astype(np.uint8)[None, :], spread[None, :])[0] > 0
        out[a : b + 1, ~(above | below)] = False

    for c_abs in panel.get("gridline_x", []):
        c = int(c_abs) - panel["x0"]
        a, b = c - half, c + half
        if a <= 0 or b >= w - 1:
            continue
        left = cv2.dilate(mask[:, a - 1].astype(np.uint8)[:, None], spread[:, None])[:, 0] > 0
        right = cv2.dilate(mask[:, b + 1].astype(np.uint8)[:, None], spread[:, None])[:, 0] > 0
        out[~(left | right), a : b + 1] = False

    return out


def _keep_curve_like(mask: np.ndarray, panel: dict, min_width_frac: float = 0.04) -> np.ndarray:
    """Keep ink that runs across the plot; drop text, tick stubs and specks.

    Pieces within a couple of pixels of each other are judged together, so a
    curve broken where a gridline was removed stays whole.
    """
    h, w = mask.shape
    reach = max(2, panel.get("line_thickness", 1) + 1)
    k = 2 * reach + 1
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    if n_labels <= 1:
        return mask
    min_width = max(10.0, min_width_frac * w) + 2 * reach
    keep = stats[:, cv2.CC_STAT_WIDTH] >= min_width
    keep[0] = False
    return mask & keep[labels]


def _detect_achromatic_traces(
    roi_rgb: np.ndarray,
    existing_masks: list[np.ndarray],
    panel: dict | None = None,
    interior: np.ndarray | None = None,
    dark: np.ndarray | None = None,
    min_col_fraction: float = 0.15,
    grey_level: float | None = None,
) -> list[np.ndarray]:
    """Detect a black or grey trace not caught by colour segmentation.

    The trace is the dark, neutral ink that is left once axis lines,
    gridlines, tick marks and text have been taken away.  Gridlines are
    removed by position (they were located with the axes) and, when they are
    paler than the trace, by darkness; text and ticks are removed because
    they do not run across the plot.

    Parameters
    ----------
    min_col_fraction : float
        The ink must cover at least this fraction of the plot's width to be
        accepted as a trace.
    grey_level : float, optional
        If given (0 = black, 255 = white), look for ink of this grey level
        rather than for the darkest ink present.
    """
    panel = panel or {"x0": 0, "y_top": 0}
    shape = roi_rgb.shape[:2]
    if interior is None:
        interior = _interior_mask(panel, shape)
    if dark is None:
        dark = darkness_map(roi_rgb)
    sat = _saturation(roi_rgb)

    candidate = interior & (sat < 0.22) & (dark > INK_THRESHOLD)
    for m in existing_masks:
        candidate &= ~m
    min_px = _min_pixels(shape)
    if candidate.sum() < min_px:
        return []

    # Keep the darkest neutral ink: a black curve over grey gridlines.
    if grey_level is None:
        strongest = float(np.percentile(dark[candidate], 99))
        candidate &= dark > max(35.0, 0.4 * strongest)
    else:
        target = 255.0 - float(grey_level)
        candidate &= np.abs(dark - target) <= max(40.0, 0.35 * target)

    candidate = _remove_gridlines(candidate, panel)
    candidate = _keep_curve_like(candidate, panel)

    if candidate.sum() < min_px:
        return []
    if candidate.any(axis=0).sum() < min_col_fraction * shape[1]:
        return []
    return [candidate]


# ===================================================================
# Main trace detection
# ===================================================================

def _trace_record(roi: np.ndarray, mask: np.ndarray, trace_id: int, hue_group: tuple[float, float], kind: str) -> dict:
    return {
        "trace_id": trace_id,
        "mask": mask,
        "pixel_count": int(mask.sum()),
        "median_rgb": np.median(roi[mask], axis=0).astype(int).tolist(),
        "hue_group": hue_group,
        "kind": kind,
    }


def detect_trace_masks(
    rgb: np.ndarray,
    panel: dict,
    use_lab: bool = False,
    colours: Optional[Sequence] = None,
    colour_tolerance: float = 60.0,
    exclude: Optional[Sequence[Sequence[float]]] = None,
    achromatic: str = "auto",
    darkness: Optional[np.ndarray] = None,
) -> list[dict]:
    """Detect trace pixels within one panel.

    Parameters
    ----------
    rgb : ndarray, shape (H, W, 3)
        Full image in RGB.
    panel : dict
        Panel dict with ``x0``, ``x1``, ``y_top``, ``y_bottom``.
    use_lab : bool
        If True, use CIELAB k-means clustering instead of HSV
        histogram binning.  Requires scikit-learn.
    colours : sequence of colours, optional
        Trace colours to extract, as ``"#rrggbb"`` strings or ``(r, g, b)``
        tuples.  When given, automatic colour grouping is skipped and one
        trace is returned per colour (in the order given).
    colour_tolerance : float
        Maximum RGB distance from a colour in *colours*.
    exclude : sequence of (x_left, y_top, x_right, y_bottom), optional
        Image regions to ignore, e.g. a legend drawn inside the plot.
    achromatic : {"auto", "always", "never"}
        Whether to look for a black/grey trace.  ``"auto"`` accepts one
        readily when the plot has no coloured trace, and only if it spans at
        least half the plot's width when it does.
    darkness : ndarray, optional
        The image's darkness map (:func:`trace_digitiser.frames.darkness_map`),
        if already computed.

    Returns
    -------
    list of dict
        Each dict has ``trace_id``, ``mask``, ``pixel_count``,
        ``median_rgb``, ``hue_group`` and ``kind`` (``"colour"`` or
        ``"achromatic"``).
    """
    if achromatic not in ("auto", "always", "never"):
        raise ValueError("achromatic must be 'auto', 'always' or 'never'")

    x0, x1 = panel["x0"], panel["x1"]
    y0, y1 = panel["y_top"], panel["y_bottom"]

    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    shape = roi.shape[:2]
    if shape[0] < 3 or shape[1] < 3:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    H = hsv[:, :, 0] / 180.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    interior = _interior_mask(panel, shape)  # for neutral ink
    on_plot = _interior_mask(panel, shape, on_axes=True, saturation=S)  # for coloured ink
    _apply_exclusions(interior, panel, exclude)
    _apply_exclusions(on_plot, panel, exclude)
    dark = (darkness if darkness is not None else darkness_map(rgb))[y0 : y1 + 1, x0 : x1 + 1]
    min_px = _min_pixels(shape)
    reach = max(2, int(round(0.012 * min(shape))))

    masks: list[dict] = []

    # ------------------------------------------------------------------
    # Traces of user-specified colours
    # ------------------------------------------------------------------
    if colours:
        roi_f = roi.astype(np.float32)
        for idx, colour in enumerate(colours, start=1):
            target = np.array(parse_colour(colour), dtype=np.float32)
            distance = np.sqrt(((roi_f - target) ** 2).sum(axis=2))
            neutral = float(target.max() - target.min()) / max(1.0, float(target.max())) < 0.2
            if neutral:
                mask = interior & (distance <= colour_tolerance)
                mask = _keep_curve_like(_remove_gridlines(mask, panel), panel)
            else:
                mask = _drop_small_components(on_plot & (distance <= colour_tolerance), min_px)
            if mask.sum() >= min_px:
                masks.append(_trace_record(roi, mask, idx, (0.0, 0.0) if neutral else (0.0, 1.0),
                                           "achromatic" if neutral else "colour"))
        return masks

    # ------------------------------------------------------------------
    # Coloured traces
    # ------------------------------------------------------------------
    # Saturated, non-background pixels.
    coloured = on_plot & (S > 0.13) & (V > 0.16)

    if use_lab and coloured.sum() > 200:
        try:
            cluster_masks = _cluster_lab(roi, coloured)
        except ImportError as exc:
            raise ImportError(
                "use_lab=True needs scikit-learn: pip install 'trace-digitiser[lab]'"
            ) from exc
        for cm in cluster_masks:
            if cm.sum() >= 2 * min_px:
                masks.append(_trace_record(roi, cm, len(masks) + 1, (0.0, 1.0), "colour"))

    if not masks:
        hue_groups = circular_hue_groups(H[coloured], min_pixels=max(12, 2 * min_px))

        for hg in hue_groups:
            mask = _drop_small_components(coloured & hue_mask_from_group(H, hg), min_px)
            if mask.sum() < 2 * min_px:
                continue

            # Recover anti-aliased / compressed edge pixels that fell
            # outside the narrow hue bin but are colour-similar.
            mask = _refine_mask_by_colour(roi, mask, on_plot, min_component_pixels=min_px)
            mask = _drop_stray_groups(mask, reach)
            if mask.sum() < 2 * min_px:
                continue

            masks.append(_trace_record(roi, mask, len(masks) + 1, (hg["lo"], hg["hi"]), "colour"))

        masks = _drop_halos(masks, S, reach)

    # ------------------------------------------------------------------
    # Black / grey trace
    # ------------------------------------------------------------------
    if achromatic != "never":
        strict = bool(masks) and achromatic == "auto"
        found = _detect_achromatic_traces(
            roi,
            [m["mask"] for m in masks],
            panel=panel,
            interior=interior,
            dark=dark,
            min_col_fraction=0.5 if strict else 0.15,
        )
        for am in found:
            masks.append(_trace_record(roi, am, len(masks) + 1, (0.0, 0.0), "achromatic"))

    return masks
