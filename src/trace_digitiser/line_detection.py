"""Horizontal and vertical line detection for panel boundary evidence.

Two complementary detectors:

* :func:`detect_long_horizontal_lines` — primary row-projection detector
  that finds gridlines/borders reliably in dense traces.
* :func:`detect_axis_line_segments` — supplementary detector that keeps
  only low-saturation (grey/black) structures, suppressing coloured
  trace pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import merge_contiguous


# ---------------------------------------------------------------------------
# Primary horizontal line detector
# ---------------------------------------------------------------------------

def detect_long_horizontal_lines(
    rgb: np.ndarray,
    min_width_frac: float = 0.12,
) -> list[dict]:
    """Robust row-projection horizontal-line detector with x-segment splitting.

    After finding a horizontal row with enough structure, the row is split
    into contiguous x-runs so that side-by-side panels on the same y-level
    are not accidentally merged.

    Parameters
    ----------
    rgb : ndarray, shape (H, W, 3)
        Input image in RGB order.
    min_width_frac : float
        Minimum fraction of image width for a segment to be kept.

    Returns
    -------
    list of dict
        Each dict has keys ``orientation``, ``x0``, ``x1``, ``y``,
        ``span``, ``thickness``, ``source``.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)[1]

    kernel_len = max(30, w // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    row_score = (horiz > 0).sum(axis=1)
    rows = np.where(row_score > min_width_frac * w)[0]
    row_groups = merge_contiguous(rows, max_gap=2)

    lines: list[dict] = []
    for y0, y1 in row_groups:
        region = horiz[y0 : y1 + 1, :]
        if region.sum() == 0:
            continue

        row_counts = (region > 0).sum(axis=1)
        y = y0 + int(np.argmax(row_counts))

        col_score = (region > 0).sum(axis=0)
        cols = np.where(col_score > 0)[0]
        x_groups = merge_contiguous(cols, max_gap=max(5, w // 150))

        for x0, x1 in x_groups:
            span = x1 - x0 + 1
            if span >= min_width_frac * w:
                lines.append(
                    {
                        "orientation": "h",
                        "x0": int(x0),
                        "x1": int(x1),
                        "y": int(y),
                        "span": int(span),
                        "thickness": int(y1 - y0 + 1),
                        "source": "legacy_row_projection_segmented",
                    }
                )

    return sorted(lines, key=lambda d: (d["y"], d["x0"]))


# ---------------------------------------------------------------------------
# Supplementary axis/grid segment detector
# ---------------------------------------------------------------------------

def detect_axis_line_segments(
    rgb: np.ndarray,
    direction: str = "horizontal",
    min_length_frac: float = 0.18,
) -> list[dict]:
    """Detect long axis/grid segments while suppressing coloured trace pixels.

    Keeps only low-saturation grey/black pixels so that coloured traces
    are much less likely to become false panel borders.

    Parameters
    ----------
    rgb : ndarray, shape (H, W, 3)
    direction : ``"horizontal"`` or ``"vertical"``
    min_length_frac : float
        Minimum fraction of image width (horizontal) or height (vertical).

    Returns
    -------
    list of dict
        Horizontal dicts have ``x0``, ``x1``, ``y``; vertical dicts
        have ``x``, ``y0``, ``y1``.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    low_saturation = S < 55
    not_white = gray < 238
    very_dark = gray < 110
    axis_like = ((low_saturation & not_white) | very_dark).astype(np.uint8) * 255

    if direction == "horizontal":
        kernel_len = max(35, w // 24)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        opened = cv2.morphologyEx(axis_like, cv2.MORPH_OPEN, kernel)
        min_len = min_length_frac * w
    elif direction == "vertical":
        kernel_len = max(35, h // 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        opened = cv2.morphologyEx(axis_like, cv2.MORPH_OPEN, kernel)
        min_len = min_length_frac * h
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    segments: list[dict] = []
    for lab in range(1, n_labels):
        x, y, ww, hh, area = stats[lab]
        if area <= 0:
            continue

        if direction == "horizontal":
            if ww < min_len or hh > max(18, 0.045 * h):
                continue
            segments.append(
                {
                    "orientation": "h",
                    "x0": int(x),
                    "x1": int(x + ww - 1),
                    "y": int(y + hh / 2),
                    "span": int(ww),
                    "thickness": int(hh),
                    "area": int(area),
                    "source": "axis_low_saturation",
                }
            )
        else:
            if hh < min_len or ww > max(18, 0.045 * w):
                continue
            segments.append(
                {
                    "orientation": "v",
                    "x": int(x + ww / 2),
                    "y0": int(y),
                    "y1": int(y + hh - 1),
                    "span": int(hh),
                    "thickness": int(ww),
                    "area": int(area),
                    "source": "axis_low_saturation",
                }
            )

    if direction == "horizontal":
        return sorted(segments, key=lambda d: (d["y"], d["x0"]))
    return sorted(segments, key=lambda d: (d["x"], d["y0"]))
