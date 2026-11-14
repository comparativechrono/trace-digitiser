"""Sampling interval detection and trace resampling.

When the underlying time series was plotted without smoothing and the
data density is lower than the pixel resolution, the trace shows
visible "steps" at regular intervals.  This module detects those steps
and optionally resamples the per-pixel trace data to one row per
detected sample.

If the trace is dense (every pixel differs) or smoothed, no interval
is detected and the per-pixel output is the best available resolution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ===================================================================
# Interval detection
# ===================================================================

def detect_sampling_interval(
    y_values: np.ndarray,
    flat_threshold: float = 1.5,
    min_run_length: int = 2,
    min_consistency: float = 0.30,
) -> Optional[float]:
    """Detect the apparent sampling interval from a 1-D trace.

    Looks for regular "steps" — flat runs where consecutive y-values
    are nearly identical, separated by transitions.

    Parameters
    ----------
    y_values : ndarray
        Per-column y-values (e.g. ``center_estimate``).
    flat_threshold : float
        Maximum absolute difference between consecutive values to
        count as "flat" (same sample).
    min_run_length : int
        Minimum flat run length to count as a step.
    min_consistency : float
        Minimum fraction of the trace that must be covered by runs of
        the detected length (otherwise the trace is considered too
        noisy or smoothed for interval detection).

    Returns
    -------
    float or None
        Detected interval in pixel columns, or None if no regular
        stepping pattern is found.
    """
    y = np.asarray(y_values, dtype=float)
    if len(y) < 10:
        return None

    # Remove NaNs for analysis.
    valid = ~np.isnan(y)
    if valid.sum() < 10:
        return None

    diffs = np.abs(np.diff(y))
    is_flat = diffs < flat_threshold

    # Extract run lengths of flat segments.
    runs: list[int] = []
    current = 0
    for f in is_flat:
        if f:
            current += 1
        else:
            if current >= min_run_length:
                runs.append(current)
            current = 0
    if current >= min_run_length:
        runs.append(current)

    if len(runs) < 3:
        return None

    run_arr = np.array(runs, dtype=float)
    median_run = float(np.median(run_arr))

    # Check consistency: do the runs cover enough of the trace?
    total_covered = run_arr.sum()
    coverage = total_covered / len(y)
    if coverage < min_consistency:
        return None

    # Check regularity: coefficient of variation should be low.
    cv = float(np.std(run_arr) / max(0.01, np.mean(run_arr)))
    if cv > 0.8:
        return None

    # The interval is median_run + 1 (the run is N flat diffs = N+1 pixels).
    return median_run + 1


# ===================================================================
# Resampling
# ===================================================================

def resample_trace(
    trace_data: pd.DataFrame,
    interval_pixels: Optional[float] = None,
    x_units: str = "",
) -> pd.DataFrame:
    """Resample per-pixel trace data to detected or user-specified intervals.

    Parameters
    ----------
    trace_data : pd.DataFrame
        Per-pixel trace data from :func:`digitise`.
    interval_pixels : float, optional
        Sampling interval in pixel columns.  If None, auto-detects
        per panel/trace.  If detection fails, returns the data as-is.
    x_units : str
        Label for the x-axis units (e.g. ``"hours"``, ``"days"``).
        Used in the ``x_units`` output column.

    Returns
    -------
    pd.DataFrame
        Resampled data with one row per sample interval per trace,
        or the original data unchanged if no interval is detected.
    """
    if trace_data.empty:
        return trace_data

    has_xval = "x_value" in trace_data.columns

    all_resampled: list[pd.DataFrame] = []

    for (pid, tid), sub in trace_data.groupby(["panel_id", "trace_id"]):
        sub = sub.sort_values("x_pixel").reset_index(drop=True)
        y = sub["center_estimate"].values

        # Determine interval.
        ivl = interval_pixels
        if ivl is None:
            ivl = detect_sampling_interval(y)

        if ivl is None or ivl < 2:
            # No stepping detected — keep per-pixel rows.
            out = sub.copy()
            out["sample_index"] = range(len(out))
            out["sampling_interval_pixels"] = 1
            out["sampling_detected"] = False
            if x_units:
                out["x_units"] = x_units
            all_resampled.append(out)
            continue

        # Bin columns into intervals.
        n_cols = len(sub)
        n_bins = max(1, int(round(n_cols / ivl)))
        bin_edges = np.linspace(0, n_cols, n_bins + 1).astype(int)

        rows: list[dict] = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if lo >= hi:
                continue
            chunk = sub.iloc[lo:hi]

            row: dict = {
                "panel_id": pid,
                "trace_id": tid,
                "sample_index": b,
                "x_pixel": int(chunk["x_pixel"].median()),
                "x_norm_0_to_1": float(chunk["x_norm_0_to_1"].median()),
                "center_estimate": float(chunk["center_estimate"].median()),
                "mean_estimate": float(chunk["mean_estimate"].mean()),
                "upper_envelope": float(chunk["upper_envelope"].max()),
                "lower_envelope": float(chunk["lower_envelope"].min()),
                "scale_type": chunk["scale_type"].iloc[0],
                "trace_median_rgb": chunk["trace_median_rgb"].iloc[0],
                "trace_pixel_count": int(chunk["trace_pixel_count"].iloc[0]),
                "sampling_interval_pixels": float(ivl),
                "sampling_detected": True,
                "n_columns_in_bin": hi - lo,
            }

            if has_xval:
                row["x_value"] = float(chunk["x_value"].median())

            if x_units:
                row["x_units"] = x_units

            rows.append(row)

        all_resampled.append(pd.DataFrame(rows))

    if not all_resampled:
        return trace_data

    result = pd.concat(all_resampled, ignore_index=True)

    # Compute sampling interval in x-axis units if calibrated.
    if has_xval and "sampling_interval_pixels" in result.columns:
        # Approximate: x-range / n-pixels * interval_in_pixels.
        for (pid, tid), sub in result.groupby(["panel_id", "trace_id"]):
            if len(sub) < 2:
                continue
            x_range = sub["x_value"].max() - sub["x_value"].min()
            n_pix = trace_data[
                (trace_data["panel_id"] == pid) & (trace_data["trace_id"] == tid)
            ]["x_pixel"].nunique()
            if n_pix > 1:
                pix_to_x = x_range / (n_pix - 1)
                mask = (result["panel_id"] == pid) & (result["trace_id"] == tid)
                result.loc[mask, "sampling_interval_x"] = (
                    result.loc[mask, "sampling_interval_pixels"] * pix_to_x
                )

    return result
