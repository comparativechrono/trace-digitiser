"""Low-level geometry helpers for panel detection.

All functions operate on plain dicts with the keys used throughout the
pipeline (``x0``, ``x1``, ``y_top``, ``y_bottom``) so that they work
both on raw candidate dicts and on :class:`~trace_digitiser.models.Panel`
instances (via ``__dict__``).
"""

from __future__ import annotations


def merge_contiguous(indices: list[int], max_gap: int = 2) -> list[tuple[int, int]]:
    """Group sorted integer indices into contiguous runs.

    >>> merge_contiguous([1, 2, 3, 10, 11])
    [(1, 3), (10, 11)]
    """
    if len(indices) == 0:
        return []

    indices = sorted(int(i) for i in indices)
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]

    for idx in indices[1:]:
        if idx <= prev + max_gap:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx

    groups.append((start, prev))
    return groups


def box_area(p: dict) -> int:
    """Return the area (in pixels²) of a panel candidate dict."""
    return max(0, p["x1"] - p["x0"]) * max(0, p["y_bottom"] - p["y_top"])


def box_iou(a: dict, b: dict) -> float:
    """Compute intersection-over-union of two panel candidate boxes."""
    x0 = max(a["x0"], b["x0"])
    x1 = min(a["x1"], b["x1"])
    y0 = max(a["y_top"], b["y_top"])
    y1 = min(a["y_bottom"], b["y_bottom"])

    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = box_area(a) + box_area(b) - inter

    return inter / union if union > 0 else 0


def contains_box(outer: dict, inner: dict, margin: int = 8) -> bool:
    """Return True if *outer* substantially contains *inner*."""
    return (
        outer["x0"] <= inner["x0"] + margin
        and outer["x1"] >= inner["x1"] - margin
        and outer["y_top"] <= inner["y_top"] + margin
        and outer["y_bottom"] >= inner["y_bottom"] - margin
    )
