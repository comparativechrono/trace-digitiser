"""Straightening a figure that was scanned or photographed slightly askew.

Everything downstream assumes that the axes run along the rows and columns
of the image: an axis line tilted by half a degree crosses several pixel
rows, is no longer found as a line, and a value read 600 pixels along it is
off by five pixels.  A page laid on a scanner by hand is rarely straighter
than that, so the tilt is measured and undone before anything else.

The tilt is found from the image as a whole rather than from any one line.
Turn a figure until its axes, gridlines and lines of print run exactly
along the pixel rows and columns, and its ink piles up in few rows and
columns; at any other angle it is smeared across many.  The angle at which
the row and column totals are most sharply peaked is the angle of the
figure.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .frames import INK_THRESHOLD, darkness_map

__all__ = ["estimate_tilt", "straighten"]

# A tilt smaller than this is left alone: it costs less accuracy than
# resampling the image would (0.1 degree is one pixel in 570).
MIN_TILT_DEGREES = 0.15


def _peakedness(ink: np.ndarray, angle: float) -> float:
    """Sharpness of the row and column totals of *ink* after turning it by *angle* degrees."""
    h, w = ink.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    turned = cv2.warpAffine(ink, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    rows, cols = turned.sum(axis=1), turned.sum(axis=0)
    return float((rows * rows).sum() + (cols * cols).sum())


def estimate_tilt(rgb: np.ndarray, max_angle: float = 5.0, darkness: Optional[np.ndarray] = None) -> float:
    """Angle in degrees by which a figure is turned anticlockwise from upright.

    Returns 0.0 for a figure that is straight, that is tilted by more than
    *max_angle* (a tilt that large is better fixed by eye), or that has no
    lines to judge by.  The estimate is good to a few hundredths of a
    degree for a figure with axis lines a few hundred pixels long.
    """
    dark = darkness if darkness is not None else darkness_map(rgb)
    ink = np.clip(dark.astype(np.float32) / 255.0, 0.0, 1.0)
    ink[dark <= INK_THRESHOLD] = 0.0
    if not ink.any():
        return 0.0
    h, w = ink.shape
    shrink = min(1.0, 1200.0 / max(h, w))
    if shrink < 1.0:
        ink = cv2.resize(ink, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)

    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.25)
    coarse_scores = np.array([_peakedness(ink, a) for a in coarse])
    k = int(coarse_scores.argmax())
    if k in (0, len(coarse) - 1):
        return 0.0  # still rising at the limit of the search: not a tilt this can measure
    step = 0.02
    fine = np.arange(coarse[k] - 0.3, coarse[k] + 0.3 + 1e-9, step)
    fine_scores = np.array([_peakedness(ink, a) for a in fine])
    j = int(fine_scores.argmax())
    # The lines must stand out: with nothing straight in the image the
    # totals are much the same at every angle, and the "best" is noise.
    if fine_scores[j] < 1.1 * float(np.median(coarse_scores)):
        return 0.0
    best = float(fine[j])
    if 0 < j < len(fine) - 1:
        before, at, after = fine_scores[j - 1], fine_scores[j], fine_scores[j + 1]
        curvature = before - 2.0 * at + after
        if curvature < 0:
            best += step * 0.5 * float(before - after) / float(curvature)
    # Turning the image by `best` straightens it, so it was tilted by the opposite.
    return -best if abs(best) >= 1e-3 else 0.0


def straighten(rgb: np.ndarray, tilt: float) -> np.ndarray:
    """Turn *rgb* clockwise by *tilt* degrees, undoing an anticlockwise tilt of that size.

    The image keeps its size; the corners turned out of view are lost and
    those turned into view are filled with the colour of the page.
    """
    if tilt == 0:
        return rgb
    h, w = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    page = tuple(int(v) for v in np.median(border, axis=0))
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -tilt, 1.0)
    return cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=page)
