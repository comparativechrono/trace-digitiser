"""Column-by-column trace digitisation.

Converts a detected boolean trace mask into a tabular series of
y-values (center, mean, upper envelope, lower envelope) at each
x-pixel column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        One row per x-column with digitised y-value estimates.
    """
    mask = trace["mask"]
    calib = panel["y_calibration"]

    # Determine if log scale.
    is_log = calib.get("scale_type", "") == "log_ocr_value"

    # Optional x calibration.
    x_calib = panel.get("x_calibration")

    rows: list[dict] = []
    n_rows, n_cols = mask.shape

    for x_local in range(n_cols):
        y_local = np.where(mask[:, x_local])[0]

        if len(y_local) == 0:
            center = mean = upper = lower = np.nan
        else:
            y_abs = y_local + panel["y_top"]
            y_arr = np.asarray(y_abs, dtype=float)
            if is_log:
                values = np.power(10.0, calib["a"] * y_arr + calib["b"])
            else:
                values = calib["a"] * y_arr + calib["b"]
            center = float(np.nanmedian(values))
            mean = float(np.nanmean(values))
            upper = float(np.nanmax(values))
            lower = float(np.nanmin(values))

        x_pixel = panel["x0"] + x_local

        row_dict: dict = {
            "panel_id": panel["panel_id"],
            "trace_id": trace["trace_id"],
            "x_pixel": x_pixel,
            "x_norm_0_to_1": x_local / max(1, n_cols - 1),
            "center_estimate": center,
            "mean_estimate": mean,
            "upper_envelope": upper,
            "lower_envelope": lower,
            "scale_type": calib["scale_type"],
            "trace_median_rgb": str(trace["median_rgb"]),
            "trace_pixel_count": trace["pixel_count"],
        }

        # Add calibrated x-value if available.
        if x_calib is not None:
            row_dict["x_value"] = x_calib["a"] * x_pixel + x_calib["b"]

        rows.append(row_dict)

    out = pd.DataFrame(rows)
    value_cols = ["center_estimate", "mean_estimate", "upper_envelope", "lower_envelope"]
    out[value_cols] = out[value_cols].interpolate(limit_direction="both")
    return out
