"""Summarise digitised traces by detected x-label intervals.

Converts dense column-level trace data into a compact table with one
row per (panel, trace, interval_label).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarise_by_detected_labels(
    trace_data: pd.DataFrame,
    panels: list[dict],
) -> pd.DataFrame:
    """Summarise digitised trace data by detected x-label intervals.

    Interval boundaries are placed at the midpoint between adjacent
    label centres.
    """
    rows: list[dict] = []
    panel_lookup = {p["panel_id"]: p for p in panels}

    if trace_data.empty:
        return pd.DataFrame()

    for (panel_id, trace_id), sub in trace_data.groupby(["panel_id", "trace_id"]):
        p = panel_lookup[panel_id]
        labels = p.get("x_labels", [])

        if len(labels) < 2:
            continue

        centers = np.array([lab["x"] for lab in labels], dtype=float)
        texts = [lab["text"] for lab in labels]

        boundaries: list[float] = [p["x0"]]
        boundaries.extend(((centers[:-1] + centers[1:]) / 2).tolist())
        boundaries.append(p["x1"])

        for i, text in enumerate(texts):
            left = boundaries[i]
            right = boundaries[i + 1]
            ss = sub[(sub["x_pixel"] >= left) & (sub["x_pixel"] < right)]

            if ss.empty:
                continue

            rows.append(
                {
                    "panel_id": panel_id,
                    "trace_id": trace_id,
                    "interval_label": text,
                    "x_left_pixel": left,
                    "x_right_pixel": right,
                    "center_mean": ss["center_estimate"].mean(),
                    "center_median": ss["center_estimate"].median(),
                    "center_q05": ss["center_estimate"].quantile(0.05),
                    "center_q95": ss["center_estimate"].quantile(0.95),
                    "upper_envelope_max": ss["upper_envelope"].max(),
                    "lower_envelope_min": ss["lower_envelope"].min(),
                    "n_columns": len(ss),
                    "scale_type": ss["scale_type"].iloc[0],
                }
            )

    return pd.DataFrame(rows)
