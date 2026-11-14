"""Column-by-column trace digitisation.

Converts a detected boolean trace mask into a table with one row per pixel
column: where the trace is in that column, in data units.

What the estimates mean depends on the kind of trace:

* For an ordinary **thin line** (one y value per x), ``center_estimate`` is
  the centre of the line and the envelopes hug it.
* For a **dense record** drawn at many samples per pixel column (a
  week of minute-by-minute activity in a 350-pixel plot, say), each column
  of the drawing is a vertical stroke from the smallest to the largest
  sample in it.  ``upper_envelope`` and ``lower_envelope`` then recover
  that maximum and minimum, and ``center_estimate`` is the middle of the
  stroke.  It is *not* the mean of the underlying samples, which a drawn
  stroke does not record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .x_calibration import apply_x

VALUE_COLUMNS = ["center_estimate", "mean_estimate", "upper_envelope", "lower_envelope"]


def digitise_trace_mask(panel: dict, trace: dict) -> pd.DataFrame:
    """Digitise one trace mask column-by-column.

    Parameters
    ----------
    panel : dict
        Panel dict with ``y_top``, ``x0``, ``panel_id``, and
        ``y_calibration`` (containing ``a``, ``b``, ``scale_type``).
    trace : dict
        Trace dict with ``mask`` (bool array), ``trace_id``,
        ``median_rgb``, ``pixel_count``.

    Returns
    -------
    pd.DataFrame
        One row per x-column, with these columns:

        ``x_pixel``
            image column;
        ``x_norm_0_to_1``
            position between the left (0) and right (1) edges of the plot area;
        ``x_value``
            calibrated x value (only if the x axis is calibrated);
        ``center_estimate``
            median of the trace pixels in the column (see the module notes);
        ``mean_estimate``
            mean of the trace pixels in the column;
        ``upper_envelope``, ``lower_envelope``
            highest and lowest trace pixel in the column;
        ``n_pixels``
            number of trace pixels found in the column;
        ``interpolated``
            True where the column had no trace pixels (hidden behind another
            trace, a gridline or the frame) and the values were filled in
            from neighbouring columns.
    """
    mask = trace["mask"]
    calib = panel["y_calibration"]
    is_log = str(calib.get("scale_type", "")).startswith("log")
    x_calib = panel.get("x_calibration")

    n_rows, n_cols = mask.shape
    edges = panel.get("edges") or {"left": float(panel["x0"]), "right": float(panel["x0"] + n_cols - 1)}
    width = max(1e-9, edges["right"] - edges["left"])

    center = np.full(n_cols, np.nan)
    mean = np.full(n_cols, np.nan)
    upper = np.full(n_cols, np.nan)
    lower = np.full(n_cols, np.nan)
    counts = mask.sum(axis=0).astype(int)

    for x_local in np.flatnonzero(counts):
        y_abs = np.flatnonzero(mask[:, x_local]).astype(float) + panel["y_top"]
        values = calib["a"] * y_abs + calib["b"]
        if is_log:
            values = np.power(10.0, values)
        center[x_local] = np.median(values)
        mean[x_local] = values.mean()
        upper[x_local] = values.max()
        lower[x_local] = values.min()

    x_pixel = panel["x0"] + np.arange(n_cols)
    out = pd.DataFrame(
        {
            "panel_id": panel["panel_id"],
            "trace_id": trace["trace_id"],
            "x_pixel": x_pixel,
            "x_norm_0_to_1": (x_pixel - edges["left"]) / width,
            "center_estimate": center,
            "mean_estimate": mean,
            "upper_envelope": upper,
            "lower_envelope": lower,
            "scale_type": calib["scale_type"],
            "trace_median_rgb": str(trace["median_rgb"]),
            "trace_pixel_count": trace["pixel_count"],
            "n_pixels": counts,
            "interpolated": counts == 0,
        }
    )
    if x_calib is not None:
        out["x_value"] = apply_x(x_calib, x_pixel)

    out[VALUE_COLUMNS] = out[VALUE_COLUMNS].interpolate(limit_direction="both")
    return out
