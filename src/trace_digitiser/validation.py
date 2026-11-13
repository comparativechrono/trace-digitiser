"""Validation: how close is a digitised trace to the data that was plotted?

A digitiser can only be trusted as far as it has been tested against data
whose true values are known.  This module does that test end to end:

1. take a known series,
2. draw it as a figure (recording exactly where the axes fall in the image),
3. digitise the image, and
4. compare what comes back with what went in.

Two entry points:

:func:`round_trip`
    Validate with **your own data** - for example a raw recording of the
    same kind as the published trace you want to digitise, drawn at the
    same size and resolution as the published figure.  This answers "how
    accurately could this figure have been digitised?" directly.

:func:`run_validation`
    Run a built-in suite of synthetic figures covering axis styles,
    colours, resolutions, JPEG compression, noise, line widths, layouts,
    log axes, dense records and tilted scans, and tabulate the errors.

:func:`run_axis_validation`
    Test the one step that depends on OCR - reading the scale of an axis
    from its tick labels - on several hundred axes with different ranges,
    steps, label sizes and typefaces, and count how often the scale is
    read correctly, declined, or read wrongly.

From the command line::

    trace-digitiser --validate results_dir/
    python -m trace_digitiser.validation results_dir/ --axes

What the numbers mean
---------------------
Errors are in **percent of the y-axis range**, so 1.0 means one hundredth
of the height of the plot.  For each pixel column the true series is
reduced to what a drawing of it can show: the lowest and highest value
the line reaches within that column (``truth_lower``, ``truth_upper``) and
their midpoint (``truth_center``).  For an ordinary thin line these three
coincide with the plotted value.  For a dense record, where many samples
share a column, they do not, and neither equals the mean of the samples
(``truth_bin_mean``): the comparison with the bin mean is reported
separately, because it is the quantity most analyses actually want and a
drawing does not preserve it.
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "FigureSpec",
    "axis_cases",
    "compare",
    "default_cases",
    "render_figure",
    "round_trip",
    "run_axis_validation",
    "run_validation",
    "summarise",
    "summarise_axes",
    "synthetic_activity",
]

# A trace counts as "recovered" when its error is below this (percent of the y range) ...
ACCURATE_RMSE_PCT = 2.0
# ... and the x axis is placed to within this (percent of the x range).
ACCURATE_X_PCT = 1.0


# ---------------------------------------------------------------------------
# Known data
# ---------------------------------------------------------------------------

def _series_smooth(seed: int, n: int = 400, xlim=(0.0, 10.0)) -> tuple[np.ndarray, np.ndarray]:
    """A smooth single-valued curve occupying 6-94 % of the unit interval."""
    rng = np.random.default_rng(seed)
    x = np.linspace(xlim[0], xlim[1], n)
    u = (x - xlim[0]) / (xlim[1] - xlim[0])
    y = (
        0.5
        + 0.22 * np.sin(2 * np.pi * (1.5 * u + rng.uniform()))
        + 0.12 * np.sin(2 * np.pi * (4.0 * u + rng.uniform()))
        + 0.10 * (u - 0.5) * rng.choice([-1, 1])
    )
    return x, np.clip(y, 0.06, 0.94)


def _series_noisy(seed: int, n: int = 900, xlim=(0.0, 14.0)) -> tuple[np.ndarray, np.ndarray]:
    """A jagged line: daily structure, noise and occasional spikes (unit interval)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(xlim[0], xlim[1], n)
    t = np.linspace(0, 14, n)
    ph = rng.uniform(0, 2 * np.pi)
    y = 55 + 40 * np.maximum(0, np.sin(2 * np.pi * t + ph)) + 28 * np.maximum(0, np.sin(4 * np.pi * t + ph / 2))
    y = y + 15 * np.sin(2 * np.pi * t / 7 + ph) + rng.normal(0, 10, n)
    for pos in rng.choice(np.arange(20, n - 20), size=16, replace=False):
        w = rng.integers(3, 16)
        lo, hi = max(0, pos - w), min(n, pos + w)
        y[lo:hi] += rng.uniform(35, 120) * np.exp(-0.5 * ((np.arange(lo, hi) - pos) / max(1, w / 3)) ** 2)
    return x, np.clip(y, 0, 300) / 300.0


def synthetic_activity(
    days: float = 7.0,
    step_minutes: float = 1.0,
    seed: int = 0,
    scale: float = 1.0,
    phase_hours: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A made-up locomotor-activity record: bursty, right-skewed and nocturnal.

    Returns ``(time_in_days, activity)`` with activity between 0 and
    *scale*.  Time starts at dark onset and activity is concentrated in the
    first half of each day.

    This is a stand-in with the right general character for exercising the
    digitiser on dense records.  It is **not** a model of any real animal
    and must not be used as data.
    """
    rng = np.random.default_rng(seed)
    n = int(round(days * 24 * 60 / step_minutes))
    t_h = np.arange(n) * step_minutes / 60.0
    circ = 0.5 * (1 + np.cos(2 * np.pi * (t_h - 5.0 - phase_hours) / 24.0))
    drive = 0.12 + 0.88 * circ ** 1.5
    arousal = np.zeros(n)
    s = 0.0
    kicks = rng.normal(0, 0.25, n)
    for i in range(n):
        s = 0.97 * s + kicks[i]
        arousal[i] = s
    bout = 1 / (1 + np.exp(-(arousal + 3.2 * (drive - 0.5))))
    y = rng.gamma(shape=1.2, scale=1.0, size=n) * bout * drive
    y = np.clip(y / np.quantile(y, 0.999) * 0.82, 0, 0.98) * scale
    return t_h / 24.0, y


# ---------------------------------------------------------------------------
# Figure specification and rendering
# ---------------------------------------------------------------------------

@dataclass
class FigureSpec:
    """How to draw a test figure.

    ``colours`` holds one entry per panel (cycled); an entry that is itself
    a tuple puts several traces in that panel.
    """

    name: str
    series: str = "smooth"  # "smooth", "noisy" or "dense" (built-in data), or "custom"
    rows: int = 1
    cols: int = 1
    panel_w_px: int = 600  # width of each plot area in pixels
    panel_h_px: int = 300
    dpi: int = 150
    style: str = "box_grid"  # "box_grid", "box", "L" or "paper"
    colours: tuple = ("#1f77b4",)
    linewidth: float = 1.2  # points
    xlim: tuple = (0.0, 10.0)
    ylim: tuple = (0.0, 100.0)
    yticks: Optional[tuple] = None
    xticks: Optional[tuple] = None
    yscale: str = "linear"
    fmt: str = "png"  # "png" or "jpg"
    jpeg_quality: int = 90
    noise_sigma: float = 0.0  # Gaussian pixel noise (grey levels)
    blur_radius: float = 0.0  # Gaussian blur (pixels)
    fontsize: float = 9.0
    seed: int = 0
    tick_dir: str = "out"
    group: str = ""
    font: str = "DejaVu Sans"  # ships with matplotlib, so figures are the same everywhere
    tilt: float = 0.0  # degrees anticlockwise, as if the page lay askew on a scanner

    def traces_for_panel(self, i: int) -> tuple:
        c = self.colours[i % len(self.colours)]
        return tuple(c) if isinstance(c, (tuple, list)) else (c,)


def _builtin_series(spec: FigureSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = spec.ylim
    if spec.series == "dense":
        odd = (seed // 10) % 2 == 1
        x, y = synthetic_activity(spec.xlim[1] - spec.xlim[0], seed=seed, scale=0.45 if odd else 1.0,
                                  phase_hours=-3.0 if odd else 0.0)
        return x + spec.xlim[0], lo + y * (hi - lo)
    if spec.series == "noisy":
        x, y = _series_noisy(seed, xlim=spec.xlim)
    else:
        x, y = _series_smooth(seed, xlim=spec.xlim)
    if spec.yscale == "log":
        return x, 10 ** (np.log10(lo) + y * (np.log10(hi) - np.log10(lo)))
    return x, lo + y * (hi - lo)


def render_figure(
    spec: FigureSpec,
    out_dir: str | Path,
    data: Optional[Sequence[Sequence[tuple[np.ndarray, np.ndarray]]]] = None,
) -> dict:
    """Draw *spec* as an image and return the ground truth.

    Parameters
    ----------
    data : optional
        ``data[panel][trace] = (x, y)``.  If omitted, built-in series are
        generated from ``spec.series`` and ``spec.seed``.

    Returns
    -------
    dict
        The truth record, also saved as ``<name>.truth.json`` beside the
        image: for each panel the exact sub-pixel position of the plot area
        (``left``, ``right``, ``top``, ``bottom``; pixel centres at
        integers), the axis limits, and each trace's data.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from PIL import Image, ImageFilter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi, fs = spec.dpi, spec.fontsize
    em = dpi * fs / 72
    ml, mr, mt, mb = int(em * 5.2) + 8, int(em * 1.2) + 6, int(em * 1.8) + 6, int(em * 3.0) + 6
    gap_x, gap_y = int(em * 5.5) + 8, int(em * 4.2) + 8
    W = ml + spec.cols * spec.panel_w_px + (spec.cols - 1) * gap_x + mr
    H = mt + spec.rows * spec.panel_h_px + (spec.rows - 1) * gap_y + mb

    with plt.rc_context({"font.family": spec.font}):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        panels = []
        for r in range(spec.rows):
            for c in range(spec.cols):
                i = r * spec.cols + c
                left = ml + c * (spec.panel_w_px + gap_x)
                top = mt + r * (spec.panel_h_px + gap_y)
                ax = fig.add_axes(
                    [left / W, 1 - (top + spec.panel_h_px) / H, spec.panel_w_px / W, spec.panel_h_px / H]
                )
                traces = []
                for k, colour in enumerate(spec.traces_for_panel(i)):
                    if data is not None:
                        x, y = (np.asarray(v, dtype=float) for v in data[i][k])
                    else:
                        x, y = _builtin_series(spec, spec.seed * 1000 + 10 * i + k)
                    ax.plot(x, y, color=colour, linewidth=spec.linewidth, solid_joinstyle="round")
                    traces.append(
                        {"colour": colour, "rgb": [int(round(255 * v)) for v in to_rgb(colour)],
                         "x": x.tolist(), "y": y.tolist()}
                    )
                ax.set_xlim(*spec.xlim)
                if spec.yscale == "log":
                    ax.set_yscale("log")
                ax.set_ylim(*spec.ylim)
                if spec.yticks is not None:
                    ax.set_yticks(list(spec.yticks))
                if spec.xticks is not None:
                    ax.set_xticks(list(spec.xticks))
                ax.tick_params(labelsize=fs, direction=spec.tick_dir, width=0.8, length=3.5)
                if spec.style == "box_grid":
                    ax.grid(True, color="0.82", linewidth=0.7)
                elif spec.style == "L":
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                elif spec.style == "paper":
                    # A small spreadsheet-style chart: grey frame and gridlines,
                    # no tick marks, interval labels inside along the top.
                    ax.grid(True, axis="y", color="0.78", linewidth=0.9)
                    for spine in ax.spines.values():
                        spine.set_color("0.72")
                        spine.set_linewidth(1.2)
                    ax.tick_params(length=0, labelsize=fs)
                    ax.set_xticks([])
                    n_seg = int(round((spec.xlim[1] - spec.xlim[0]) * 2))
                    for seg in range(n_seg):
                        ax.text(
                            spec.xlim[0] + (seg + 0.5) / 2.0, spec.ylim[1] * 0.985,
                            f"{seg // 2 + 1}{'N' if seg % 2 == 0 else 'D'}",
                            ha="center", va="top", fontsize=fs + 1, fontweight="bold",
                        )
                    ax.set_ylabel("Activity (cm/sec)", fontsize=fs - 1)
                elif spec.style != "box":
                    raise ValueError(f"unknown style {spec.style!r}")
                fig.canvas.draw()
                bb = ax.get_window_extent()
                panels.append(
                    {
                        "index": i + 1, "row": r, "col": c,
                        # Continuous pixel coordinates with pixel centres at integers.
                        "left": bb.x0 - 0.5, "right": bb.x1 - 0.5,
                        "top": (H - bb.y1) - 0.5, "bottom": (H - bb.y0) - 0.5,
                        "xlim": list(spec.xlim), "ylim": list(spec.ylim), "yscale": spec.yscale,
                        "traces": traces,
                    }
                )
        png = out_dir / f"{spec.name}.png"
        fig.savefig(png, dpi=dpi, facecolor="white")
        plt.close(fig)

    img = Image.open(png).convert("RGB")
    if img.size != (W, H):
        raise RuntimeError(f"rendered {img.size}, expected {(W, H)}")
    if spec.tilt:
        # The truth record keeps the positions of the upright figure, which
        # is what the digitiser reports after straightening the image.
        img = img.rotate(spec.tilt, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if spec.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=spec.blur_radius))
    if spec.noise_sigma > 0:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(spec.seed + 99).normal(0, spec.noise_sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if spec.fmt == "jpg":
        path = out_dir / f"{spec.name}.jpg"
        img.save(path, quality=spec.jpeg_quality, subsampling=1)
        png.unlink()
    elif spec.fmt == "png":
        path = png
        img.save(path)
    else:
        raise ValueError("fmt must be 'png' or 'jpg'")

    truth = {"spec": asdict(spec), "image": path.name, "width": W, "height": H, "panels": panels}
    (out_dir / f"{spec.name}.truth.json").write_text(json.dumps(truth))
    return truth


# ---------------------------------------------------------------------------
# What a drawing of a series can show
# ---------------------------------------------------------------------------

def drawn_envelope(
    x: np.ndarray,
    y: np.ndarray,
    centres: np.ndarray,
    half_width: float,
    mean_half_width: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lowest and highest value a polyline reaches near each of *centres*.

    For each centre the polyline through ``(x, y)`` is cut to the window
    ``centre ± half_width`` and its minimum and maximum taken - the vertical
    extent of the drawn line within one pixel column.  Also returns the mean
    of the samples within ``centre ± mean_half_width`` (default: the same
    window), NaN where there are none.

    Returns ``(lower, upper, sample_mean)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if mean_half_width is None:
        mean_half_width = half_width
    a, b = centres - half_width, centres + half_width
    ya, yb = np.interp(a, x, y), np.interp(b, x, y)
    lower, upper = np.minimum(ya, yb), np.maximum(ya, yb)
    i0 = np.searchsorted(x, a, side="left")
    i1 = np.searchsorted(x, b, side="right")
    m0 = np.searchsorted(x, centres - mean_half_width, side="left")
    m1 = np.searchsorted(x, centres + mean_half_width, side="left")
    mean = np.full(len(centres), np.nan)
    for j in range(len(centres)):
        if i1[j] > i0[j]:
            seg = y[i0[j] : i1[j]]
            lower[j] = min(lower[j], seg.min())
            upper[j] = max(upper[j], seg.max())
        if m1[j] > m0[j]:
            mean[j] = y[m0[j] : m1[j]].mean()
    return lower, upper, mean


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _fraction(values, ylim, yscale) -> np.ndarray:
    """Position of values on the y axis as a fraction of its length."""
    values = np.asarray(values, dtype=float)
    if yscale == "log":
        with np.errstate(invalid="ignore", divide="ignore"):
            lo, hi = np.log10(ylim[0]), np.log10(ylim[1])
            return (np.log10(np.clip(values, 1e-300, None)) - lo) / (hi - lo)
    return (values - ylim[0]) / (ylim[1] - ylim[0])


def _rmse(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def _panel_box(panel) -> tuple[float, float, float, float]:
    edges = getattr(panel, "edges", None)
    if edges:
        return edges["left"], edges["right"], edges["top"], edges["bottom"]
    return float(panel.x0), float(panel.x1), float(panel.y_top), float(panel.y_bottom)


def compare(result, truth: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare a digitisation with the ground truth of a rendered figure.

    Parameters
    ----------
    result : DigitiserResult
        What :func:`trace_digitiser.digitise` returned for the image.
    truth : dict
        The record returned by :func:`render_figure`.

    Returns
    -------
    (metrics, columns)
        *metrics* has one row per true trace:

        ``status``
            ``"ok"``, ``"panel_missed"`` or ``"no_trace"``;
        ``d_left, d_right, d_top, d_bottom``
            error of the detected plot area, in pixels;
        ``ycal_err_top_pct, ycal_err_bottom_pct``
            error of the y calibration at the ends of the axis;
        ``x_err_left_pct, x_err_right_pct``
            error of the x positions at the ends of the trace;
        ``rmse_pct, bias_pct, max_abs_pct``
            error of ``center_estimate`` against ``truth_center``, using the
            x positions the digitiser reported (so that an x error counts);
        ``rmse_at_true_x_pct``
            the same with each column placed at its true x (y error alone);
        ``rmse_upper_pct, rmse_lower_pct``
            error of the envelopes;
        ``corr``
            correlation of ``center_estimate`` with ``truth_center``;
        ``corr_bin_mean, bias_bin_mean_pct``
            comparison with the mean of the samples in each column;
        ``n_traces_found``
            traces the digitiser reported in the panel.

        All ``_pct`` columns are percent of the axis range.

        *columns* is the digitiser's per-column table for the matched traces
        with the truth alongside (``truth_center``, ``truth_lower``,
        ``truth_upper``, ``truth_bin_mean``), ready for plotting.
    """
    spec = truth["spec"]
    detected = list(result.panels) if result is not None else []
    trace_data = result.trace_data if result is not None else pd.DataFrame()
    lw_px = spec["linewidth"] * spec["dpi"] / 72.0
    rows: list[dict] = []
    column_frames: list[pd.DataFrame] = []

    for tp in truth["panels"]:
        # Match the detected panel by overlap.
        best, best_iou = None, 0.0
        for p in detected:
            pl, pr, pt, pb = _panel_box(p)
            ix = max(0.0, min(pr, tp["right"]) - max(pl, tp["left"]))
            iy = max(0.0, min(pb, tp["bottom"]) - max(pt, tp["top"]))
            union = (pr - pl) * (pb - pt) + (tp["right"] - tp["left"]) * (tp["bottom"] - tp["top"]) - ix * iy
            iou = ix * iy / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = p, iou

        px_to_x = (tp["xlim"][1] - tp["xlim"][0]) / (tp["right"] - tp["left"])
        x_range = tp["xlim"][1] - tp["xlim"][0]
        used: set[int] = set()

        for k, tr in enumerate(tp["traces"]):
            row: dict = {
                "case": spec["name"], "group": spec["group"], "panel": tp["index"], "trace": k + 1,
                "colour": tr["colour"], "n_panels_true": len(truth["panels"]), "n_panels_found": len(detected),
                "n_traces_true": len(tp["traces"]), "panel_iou": round(best_iou, 3),
            }
            if best is None or best_iou < 0.5:
                row["status"] = "panel_missed"
                rows.append(row)
                continue
            pl, pr, pt, pb = _panel_box(best)
            row.update(
                d_left=pl - tp["left"], d_right=pr - tp["right"], d_top=pt - tp["top"], d_bottom=pb - tp["bottom"]
            )
            cal = best.calibration
            row["scale_type"] = cal.scale_type if cal is not None else "none"
            row["y_confidence"] = getattr(cal, "confidence", "") if cal is not None else ""
            if cal is not None:
                row["ycal_err_top_pct"] = 100 * (
                    float(_fraction(cal.y_to_value(tp["top"]), tp["ylim"], tp["yscale"])) - 1.0
                )
                row["ycal_err_bottom_pct"] = 100 * float(
                    _fraction(cal.y_to_value(tp["bottom"]), tp["ylim"], tp["yscale"])
                )
            sub = trace_data[trace_data["panel_id"] == best.panel_id] if not trace_data.empty else trace_data
            row["n_traces_found"] = int(sub["trace_id"].nunique()) if not sub.empty else 0
            if sub.empty:
                row["status"] = "no_trace"
                rows.append(row)
                continue

            tx, ty = np.asarray(tr["x"]), np.asarray(tr["y"])
            half = (0.5 + 0.5 * lw_px) * px_to_x
            thick_pct = 100 * 0.5 * lw_px / (tp["bottom"] - tp["top"])
            chosen = None
            for tid, g in sub.groupby("trace_id"):
                if tid in used:
                    continue
                g = g.sort_values("x_pixel")
                cols = g["x_pixel"].to_numpy(dtype=float)
                inside = (cols >= tp["left"] + 2) & (cols <= tp["right"] - 2)
                if inside.sum() < 10:
                    continue
                g = g[inside]
                cols = cols[inside]
                x_true = tp["xlim"][0] + (cols - tp["left"]) * px_to_x
                if "x_value" in g.columns and g["x_value"].notna().all():
                    x_tool = g["x_value"].to_numpy(dtype=float)
                else:
                    x_tool = tp["xlim"][0] + g["x_norm_0_to_1"].to_numpy(dtype=float) * x_range
                lo_t, hi_t, mean_t = drawn_envelope(tx, ty, x_true, half, 0.5 * px_to_x)
                lo_e, hi_e, _ = drawn_envelope(tx, ty, x_tool, half)

                def f(v, ylim=tp["ylim"], yscale=tp["yscale"]):
                    return _fraction(v, ylim, yscale)

                centre = g["center_estimate"].to_numpy(dtype=float)
                mid_t = (
                    10 ** (0.5 * (np.log10(lo_t) + np.log10(hi_t))) if tp["yscale"] == "log" else 0.5 * (lo_t + hi_t)
                )
                mid_e = (
                    10 ** (0.5 * (np.log10(lo_e) + np.log10(hi_e))) if tp["yscale"] == "log" else 0.5 * (lo_e + hi_e)
                )
                err = 100 * (f(centre) - f(mid_e))
                err_true_x = 100 * (f(centre) - f(mid_t))
                ok = np.isfinite(err)
                if ok.sum() < 10:
                    continue
                with np.errstate(invalid="ignore"):
                    corr = (
                        float(np.corrcoef(centre[ok], mid_e[ok])[0, 1])
                        if np.std(centre[ok]) > 0 and np.std(mid_e[ok]) > 0
                        else float("nan")
                    )
                m = {
                    "trace_id_found": int(tid),
                    "rmse_pct": _rmse(err),
                    "bias_pct": float(np.nanmean(err)),
                    "max_abs_pct": float(np.nanmax(np.abs(err))),
                    "rmse_at_true_x_pct": _rmse(err_true_x),
                    # The drawn line is lw_px thick, so its edges lie half that beyond the data.
                    "rmse_upper_pct": _rmse(100 * (f(g["upper_envelope"].to_numpy(dtype=float)) - f(hi_t)) - thick_pct),
                    "rmse_lower_pct": _rmse(100 * (f(g["lower_envelope"].to_numpy(dtype=float)) - f(lo_t)) + thick_pct),
                    "x_err_left_pct": float(100 * (x_tool[0] - x_true[0]) / x_range),
                    "x_err_right_pct": float(100 * (x_tool[-1] - x_true[-1]) / x_range),
                    "corr": corr,
                    "columns_compared": int(ok.sum()),
                    "columns_interpolated": int(g["interpolated"].sum()) if "interpolated" in g else 0,
                }
                has_mean = np.isfinite(mean_t)
                if has_mean.sum() >= 10 and np.std(mean_t[has_mean]) > 0:
                    m["corr_bin_mean"] = float(np.corrcoef(centre[has_mean], mean_t[has_mean])[0, 1])
                    m["bias_bin_mean_pct"] = float(np.mean(100 * (f(centre[has_mean]) - f(mean_t[has_mean]))))
                frame = g.copy()
                frame["x_true"] = x_true
                frame["truth_center"], frame["truth_lower"], frame["truth_upper"], frame["truth_bin_mean"] = (
                    mid_t,
                    lo_t,
                    hi_t,
                    mean_t,
                )
                frame.insert(0, "case", spec["name"])
                if chosen is None or m["rmse_pct"] < chosen[0]["rmse_pct"]:
                    chosen = (m, frame)
            if chosen is None:
                row["status"] = "no_trace"
            else:
                used.add(chosen[0]["trace_id_found"])
                row.update(chosen[0])
                row["status"] = "ok"
                column_frames.append(chosen[1])
            rows.append(row)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics["recovered"] = _recovered(metrics)
    columns = pd.concat(column_frames, ignore_index=True) if column_frames else pd.DataFrame()
    return metrics, columns


def _declined(metrics: pd.DataFrame) -> pd.Series:
    """Traces that were found but whose y axis the digitiser left uncalibrated, saying so."""
    if "scale_type" not in metrics:
        return pd.Series(False, index=metrics.index)
    return metrics["status"].eq("ok") & metrics["scale_type"].eq("normalised_0_to_1")


def _recovered(metrics: pd.DataFrame) -> pd.Series:
    """A trace is recovered if it was found, placed correctly in x, and accurate in y."""
    def col(name: str, fill: float) -> pd.Series:
        return metrics[name] if name in metrics else pd.Series(fill, index=metrics.index)

    return (
        metrics["status"].eq("ok")
        & ~_declined(metrics)
        & (col("rmse_pct", np.inf) <= ACCURATE_RMSE_PCT)
        & (col("x_err_left_pct", np.inf).abs() <= ACCURATE_X_PCT)
        & (col("x_err_right_pct", np.inf).abs() <= ACCURATE_X_PCT)
        & (col("n_traces_found", -1) == col("n_traces_true", -2))
    )


# ---------------------------------------------------------------------------
# Round trip with your own data
# ---------------------------------------------------------------------------

def round_trip(
    x: Sequence[float],
    y: Sequence[float],
    out_dir: str | Path,
    name: str = "round_trip",
    *,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    panel_px: tuple[int, int] = (600, 300),
    dpi: int = 150,
    linewidth: float = 1.0,
    colour: str = "#1f77b4",
    style: str = "box_grid",
    yticks: Optional[tuple] = None,
    fmt: str = "png",
    jpeg_quality: int = 90,
    fontsize: float = 9.0,
    supply_axis_limits: bool = True,
    digitise_kwargs: Optional[dict] = None,
) -> dict:
    """Draw a known series as a figure, digitise the figure, and compare.

    Use this with a raw recording of the same kind as a published trace to
    find out how well that trace can be recovered from its figure: set
    *panel_px*, *dpi*, *linewidth*, *style* and *fmt* to resemble the
    published figure.

    Parameters
    ----------
    x, y : sequences of float
        The known series (x increasing).
    xlim, ylim : (low, high), optional
        Axis limits; default to the range of the data (y padded by 5 %).
    panel_px : (width, height)
        Size of the plot area in pixels.
    supply_axis_limits : bool
        If True the digitiser is told the axis limits, as a careful user
        would; if False it must read them from the tick labels.
    digitise_kwargs : dict, optional
        Extra arguments for :func:`trace_digitiser.digitise`.

    Returns
    -------
    dict with keys

        ``metrics``  one-row DataFrame (see :func:`compare`),
        ``columns``  per-column DataFrame with digitised values and truth,
        ``result``   the :class:`~trace_digitiser.DigitiserResult`,
        ``image``    path of the rendered figure,
        ``overlay``  path of the overlay image (digitised trace on figure),
        ``truth``    the truth record.
    """
    from . import digitise

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("x and y must be 1-D sequences of the same length (at least 2)")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must not contain NaN or infinite values")
    if xlim is None:
        xlim = (float(x[0]), float(x[-1]))
    if ylim is None:
        pad = 0.05 * (float(y.max()) - float(y.min()) or 1.0)
        ylim = (float(y.min()) - pad, float(y.max()) + pad)

    out_dir = Path(out_dir)
    spec = FigureSpec(
        name=name, series="custom", panel_w_px=int(panel_px[0]), panel_h_px=int(panel_px[1]), dpi=dpi,
        style=style, colours=(colour,), linewidth=linewidth, xlim=tuple(xlim), ylim=tuple(ylim),
        yticks=yticks, fmt=fmt, jpeg_quality=jpeg_quality, fontsize=fontsize, group="round_trip",
    )
    truth = render_figure(spec, out_dir, data=[[(x, y)]])
    image = out_dir / truth["image"]

    kwargs = dict(digitise_kwargs or {})
    if supply_axis_limits:
        kwargs.setdefault("y_min", ylim[0])
        kwargs.setdefault("y_max", ylim[1])
        kwargs.setdefault("x_min", xlim[0])
        kwargs.setdefault("x_max", xlim[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = digitise(image, output_dir=out_dir, output_prefix=name, save_overlay=True, **kwargs)
    metrics, columns = compare(result, truth)
    columns.to_csv(out_dir / f"{name}_digitised_vs_truth.csv", index=False)
    metrics.to_csv(out_dir / f"{name}_metrics.csv", index=False)
    return {
        "metrics": metrics, "columns": columns, "result": result,
        "image": image, "overlay": result.overlay_path, "truth": truth,
    }


# ---------------------------------------------------------------------------
# Built-in suite
# ---------------------------------------------------------------------------

def default_cases() -> list[FigureSpec]:
    """The built-in validation suite: 44 figures, 57 traces."""
    tab = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#bcbd22")
    cases = [
        # A. Framed plot with grey gridlines and one coloured line; varied y axes.
        FigureSpec("A1_box_grid_blue", group="A_reference", yticks=(0, 25, 50, 75, 100)),
        FigureSpec(
            "A2_box_grid_red_0_300",
            group="A_reference",
            colours=("#d62728",),
            ylim=(0, 300),
            yticks=(0, 100, 200, 300),
            seed=1,
        ),
        FigureSpec(
            "A3_box_grid_decimal_0_1",
            group="A_reference",
            colours=("#2ca02c",),
            ylim=(0, 1.0),
            yticks=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
            seed=2,
        ),
        FigureSpec(
            "A4_box_grid_negative",
            group="A_reference",
            colours=("#9467bd",),
            ylim=(-50, 50),
            yticks=(-50, -25, 0, 25, 50),
            seed=3,
        ),
        FigureSpec(
            "A5_box_grid_thousands",
            group="A_reference",
            colours=("#ff7f0e",),
            ylim=(0, 5000),
            yticks=(0, 1000, 2000, 3000, 4000, 5000),
            seed=4,
        ),
        # B. Axis styles common in older papers.
        FigureSpec("B1_box_nogrid_ticks_out", group="B_axes_style", style="box", seed=5),
        FigureSpec("B2_L_axes_ticks_out", group="B_axes_style", style="L", seed=6),
        FigureSpec("B3_box_nogrid_ticks_in", group="B_axes_style", style="box", tick_dir="in", seed=7),
        FigureSpec("B4_box_grid_ticks_in", group="B_axes_style", style="box_grid", tick_dir="in", seed=8),
    ]
    # C. Trace colour, including black and grey.
    cases += [
        FigureSpec(f"C{i + 1}_colour_{c[1:]}", group="C_colour", colours=(c,), seed=10 + i) for i, c in enumerate(tab)
    ]
    cases += [
        FigureSpec("C9_black_line_grid", group="C_colour", colours=("#000000",), seed=20),
        FigureSpec("C10_black_line_nogrid", group="C_colour", colours=("#000000",), style="box", seed=21),
        FigureSpec("C11_grey_line", group="C_colour", colours=("#555555",), seed=22),
        FigureSpec("C12_pastel_pink", group="C_colour", colours=("#da9693",), seed=23),
    ]
    # D. Resolution.
    cases += [
        FigureSpec(f"D_dpi{dpi}", group="D_resolution", dpi=dpi, panel_w_px=w, panel_h_px=h, seed=30)
        for dpi, w, h in ((60, 240, 120), (100, 400, 200), (200, 800, 400), (300, 1200, 600))
    ]
    # E. Compression, noise and blur.
    cases += [
        FigureSpec(f"E_jpeg_q{q}", group="E_degradation", fmt="jpg", jpeg_quality=q, seed=40) for q in (95, 75, 50, 30)
    ]
    cases += [
        FigureSpec("E_noise_sigma8_jpeg85", group="E_degradation", fmt="jpg", jpeg_quality=85, noise_sigma=8, seed=41),
        FigureSpec("E_blur1_jpeg80", group="E_degradation", fmt="jpg", jpeg_quality=80, blur_radius=1.0, seed=42),
    ]
    # F. Line width.
    cases += [
        FigureSpec(f"F_linewidth_{str(lw).replace('.', 'p')}", group="F_linewidth", linewidth=lw, seed=50)
        for lw in (0.5, 2.5)
    ]
    # G. Several panels.
    cases += [
        FigureSpec("G1_stacked_2", group="G_layout", rows=2, colours=("#1f77b4", "#d62728"), panel_h_px=220, seed=60),
        FigureSpec(
            "G2_horizontal_3", group="G_layout", cols=3, colours=tab[:3], panel_w_px=360, panel_h_px=260, seed=61
        ),
        FigureSpec(
            "G3_grid_2x2", group="G_layout", rows=2, cols=2, colours=tab[:4], panel_w_px=420, panel_h_px=240, seed=62
        ),
    ]
    # H. Several traces in one panel.
    cases += [
        FigureSpec("H1_two_traces_blue_red", group="H_multi_trace", colours=(("#1f77b4", "#d62728"),), seed=70),
        FigureSpec("H2_three_traces", group="H_multi_trace", colours=(("#1f77b4", "#d62728", "#2ca02c"),), seed=71),
    ]
    # I. Logarithmic y axis.
    cases.append(FigureSpec("I1_log_y", group="I_log", yscale="log", ylim=(1, 1000), seed=80))
    # J. A jagged line with pixel noise.
    cases.append(
        FigureSpec("J1_noisy_stacked_2", group="J_jagged", series="noisy", rows=2, colours=("#1f77b4", "#d62728"),
                   xlim=(0, 14), ylim=(0, 300), yticks=(0, 100, 200, 300), panel_w_px=1100, panel_h_px=380,
                   dpi=170, linewidth=1.15, seed=90)
    )
    # K. A dense record at low resolution: a week of minute-by-minute activity
    #    in a plot 348 pixels wide, as in a small journal figure.
    paper = dict(group="K_dense", series="dense", rows=2, style="paper", colours=("#578dd2", "#da9693"),
                 xlim=(0, 7), ylim=(0, 300), yticks=(0, 50, 100, 150, 200, 250, 300), linewidth=0.6,
                 fontsize=6.5, fmt="jpg", jpeg_quality=85, seed=100)
    cases += [
        FigureSpec("K1_dense_72dpi", panel_w_px=348, panel_h_px=137, dpi=72, **paper),
        FigureSpec("K2_dense_144dpi", panel_w_px=696, panel_h_px=274, dpi=144, **paper),
    ]
    # L. A figure that is askew, as a page laid on a scanner by hand is.
    cases += [
        FigureSpec("L1_tilted_half_a_degree", group="L_tilted", style="box", tilt=0.5, seed=110),
        FigureSpec("L2_tilted_2_degrees_stacked_2", group="L_tilted", rows=2, colours=("#1f77b4", "#d62728"),
                   panel_h_px=220, tilt=-2.0, fmt="jpg", jpeg_quality=85, seed=111),
    ]
    return cases


def _layout_hint(spec: dict) -> dict:
    rows, cols = spec["rows"], spec["cols"]
    if rows == 1 and cols == 1:
        return {"layout_mode": "single"}
    if cols == 1:
        return {"layout_mode": "stacked", "expected_rows": rows, "expected_cols": 1}
    if rows == 1:
        return {"layout_mode": "horizontal", "expected_rows": 1, "expected_cols": cols}
    return {"layout_mode": "grid", "expected_rows": rows, "expected_cols": cols}


def run_validation(
    out_dir: str | Path,
    cases: Optional[Sequence[FigureSpec]] = None,
    modes: Sequence[str] = ("auto", "limits"),
    digitise_fn: Optional[Callable] = None,
    layout_hints: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Render the validation figures, digitise them, and tabulate the errors.

    Parameters
    ----------
    out_dir : directory for the figures, their truth records and ``results.csv``.
    cases : figures to use; default :func:`default_cases`.
    modes : how the digitiser is run on each figure:

        ``"auto"``    nothing is supplied - axes are read from the tick labels;
        ``"limits"``  the axis limits are supplied, as a careful user would.
    digitise_fn : the function to test; default :func:`trace_digitiser.digitise`.
    layout_hints : also pass the figure's panel layout (needed by versions 1.x).

    Returns
    -------
    DataFrame
        One row per true trace per mode (see :func:`compare`), also written
        to ``out_dir/results.csv``.  Summarise it with :func:`summarise`.
    """
    if digitise_fn is None:
        from . import digitise as digitise_fn  # type: ignore[no-redef]
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    all_rows: list[pd.DataFrame] = []
    started = time.time()

    for spec in (cases if cases is not None else default_cases()):
        truth = render_figure(spec, fig_dir)
        for mode in modes:
            kwargs: dict = _layout_hint(truth["spec"]) if layout_hints else {}
            if mode == "limits":
                kwargs.update(x_min=spec.xlim[0], x_max=spec.xlim[1])
                if spec.yscale == "linear":
                    kwargs.update(y_min=spec.ylim[0], y_max=spec.ylim[1])
            elif mode != "auto":
                raise ValueError("modes must be 'auto' and/or 'limits'")
            t0 = time.time()
            error = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = digitise_fn(
                        fig_dir / truth["image"], output_dir=out_dir / "digitised" / mode / spec.name, **kwargs
                    )
            except Exception as exc:  # a crash is a result too
                result, error = None, f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            metrics, _ = compare(result, truth)
            metrics.insert(2, "mode", mode)
            metrics["seconds"] = round(time.time() - t0, 2)
            if error is not None:
                metrics["status"] = "crash"
                metrics["error"] = error[:200]
                metrics["recovered"] = False
            all_rows.append(metrics)
            if progress:
                declined = _declined(metrics)
                states = ", ".join(
                    "y axis not read" if off else r.status + (f" {r.rmse_pct:.2f}%" if r.status == "ok" else "")
                    for r, off in zip(metrics.itertuples(), declined)
                )
                elapsed, took = time.time() - started, metrics["seconds"].iloc[0]
                print(f"[{elapsed:5.0f}s] {spec.name:26s} {mode:6s} {took:5.1f}s  {states}", flush=True)

    results = pd.concat(all_rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    return results


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Condense :func:`run_validation` results to one row per mode and group, plus totals.

    ``found``
        traces whose panel and curve were located;
    ``recovered``
        of those, the ones placed correctly in x (within 1 % of the axis)
        and accurate in y (RMSE within 2 % of the axis);
    ``declined``
        found, but the y axis was left uncalibrated because its labels
        could not be read with any assurance - the digitiser says so and
        asks for the axis limits;
    ``inaccurate``
        the rest of those found: values were returned, and they are outside
        those tolerances.

    The error columns leave out the declined traces, which have no values
    to be in error.
    """
    results = results.copy()
    results["found"] = results["status"].eq("ok")
    results["recovered"] = _recovered(results)
    results["declined"] = _declined(results)
    results["inaccurate"] = results["found"] & ~results["recovered"] & ~results["declined"]
    results["error"] = results["rmse_pct"].where(~results["declined"]) if "rmse_pct" in results else np.nan
    per_figure = results.groupby(["mode", "group", "case"])["seconds"].first().groupby(["mode", "group"]).median()

    def table(keys: list[str]) -> pd.DataFrame:
        g = results.groupby(keys)
        out = g.agg(
            traces=("found", "size"), found=("found", "sum"), recovered=("recovered", "sum"),
            declined=("declined", "sum"), inaccurate=("inaccurate", "sum"),
            median_rmse_pct=("error", "median"), worst_rmse_pct=("error", "max"),
        )
        return out.reset_index()

    by_group = table(["mode", "group"])
    by_group["seconds_per_figure"] = [
        per_figure.get((m, g), np.nan) for m, g in zip(by_group["mode"], by_group["group"])
    ]
    totals = table(["mode"])
    totals["group"] = "ALL"
    totals["seconds_per_figure"] = [
        results[results["mode"] == m].groupby("case")["seconds"].first().median() for m in totals["mode"]
    ]
    out = pd.concat([by_group, totals], ignore_index=True)
    for name in ("median_rmse_pct", "worst_rmse_pct", "seconds_per_figure"):
        out[name] = out[name].round(3)
    return out[
        ["mode", "group", "traces", "found", "recovered", "declined", "inaccurate",
         "median_rmse_pct", "worst_rmse_pct", "seconds_per_figure"]
    ]


# ---------------------------------------------------------------------------
# Reading axis scales
# ---------------------------------------------------------------------------

_Y_AXES: tuple[tuple[str, tuple, Optional[tuple], str], ...] = (
    ("0_100", (0, 100), (0, 20, 40, 60, 80, 100), "linear"),
    ("0_300", (0, 300), (0, 100, 200, 300), "linear"),
    ("0_300_by_50", (0, 300), (0, 50, 100, 150, 200, 250, 300), "linear"),
    ("0_1", (0, 1), (0, 0.2, 0.4, 0.6, 0.8, 1.0), "linear"),
    ("m50_50", (-50, 50), (-50, -25, 0, 25, 50), "linear"),
    ("0_5000", (0, 5000), (0, 1000, 2000, 3000, 4000, 5000), "linear"),
    ("0_24", (0, 24), (0, 6, 12, 18, 24), "linear"),
    ("0_360", (0, 360), (0, 90, 180, 270, 360), "linear"),
    ("0_0p05", (0, 0.05), (0, 0.01, 0.02, 0.03, 0.04, 0.05), "linear"),
    ("m1p5_1p5", (-1.5, 1.5), (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5), "linear"),
    ("20_80", (20, 80), (20, 30, 40, 50, 60, 70, 80), "linear"),
    ("36_39", (36, 39), (36, 36.5, 37, 37.5, 38, 38.5, 39), "linear"),
    ("0_2p5", (0, 2.5), (0, 0.5, 1.0, 1.5, 2.0, 2.5), "linear"),
    ("100_200", (100, 200), (100, 125, 150, 175, 200), "linear"),
    ("0_1200", (0, 1200), (0, 200, 400, 600, 800, 1000, 1200), "linear"),
    ("0_48", (0, 48), (0, 12, 24, 36, 48), "linear"),
    ("0_14", (0, 14), (0, 2, 4, 6, 8, 10, 12, 14), "linear"),
    ("0_7", (0, 7), (0, 1, 2, 3, 4, 5, 6, 7), "linear"),
    ("0_30000", (0, 30000), (0, 10000, 20000, 30000), "linear"),
    ("log_1_1000", (1, 1000), None, "log"),
    ("log_0p01_100", (0.01, 100), None, "log"),
    ("0_150", (0, 150), (0, 50, 100, 150), "linear"),
    ("m200_0", (-200, 0), (-200, -150, -100, -50, 0), "linear"),
    ("0_0p8", (0, 0.8), (0, 0.2, 0.4, 0.6, 0.8), "linear"),
    ("0_10_by_2p5", (0, 10), (0, 2.5, 5.0, 7.5, 10.0), "linear"),
)
_X_AXES: tuple[tuple[tuple, tuple], ...] = (
    ((0, 10), (0, 2, 4, 6, 8, 10)),
    ((0, 24), (0, 4, 8, 12, 16, 20, 24)),
    ((0, 72), (0, 12, 24, 36, 48, 60, 72)),
    ((0, 14), tuple(range(0, 15, 2))),
    ((0, 1), (0, 0.25, 0.5, 0.75, 1.0)),
    ((-5, 5), (-5, -2.5, 0, 2.5, 5)),
    ((1990, 2020), (1990, 2000, 2010, 2020)),
    ((0, 168), tuple(range(0, 169, 24))),
    ((0, 500), (0, 100, 200, 300, 400, 500)),
)
# (dots per inch, type size in points): print from about 5 to 24 pixels high.
_PRINT_SIZES = ((72, 6.5), (60, 9.0), (72, 9.0), (100, 9.0), (150, 9.0), (300, 8.0))
# Typefaces that ship with matplotlib, so the figures are the same everywhere:
# a wide sans-serif, a serif, and a Times-like serif.
_TYPEFACES = (("sans", "DejaVu Sans"), ("serif", "DejaVu Serif"), ("times", "STIXGeneral"))


def axis_cases() -> list[FigureSpec]:
    """Figures for testing how reliably axis scales are read from tick labels.

    Twenty-five y axes (whole numbers, decimals, negatives, thousands,
    clock and angle steps, two logarithmic) paired with nine x axes, each
    drawn with print from about 5 to 24 pixels high, in three typefaces at
    the middle sizes, and degraded by JPEG compression and blur: 375
    figures, 750 axes.
    """
    cases: list[FigureSpec] = []
    for i, (name, ylim, yticks, yscale) in enumerate(_Y_AXES):
        xlim, xticks = _X_AXES[i % len(_X_AXES)]

        def add(group: str, dpi: int, points: float, i=i, name=name, ylim=ylim, yticks=yticks, yscale=yscale,
                xlim=xlim, xticks=xticks, **kwargs) -> None:
            cases.append(
                FigureSpec(
                    f"axis_{name}__{group}", ylim=ylim, yticks=yticks, yscale=yscale, xlim=xlim, xticks=xticks,
                    dpi=dpi, fontsize=points, panel_w_px=int(4.2 * dpi), panel_h_px=int(2.6 * dpi),
                    seed=i, group=group, **kwargs,
                )
            )

        for dpi, points in _PRINT_SIZES:
            add(f"sans_{dpi}dpi_{points:g}pt", dpi, points)
        for tag, font in _TYPEFACES[1:]:
            for dpi, points in _PRINT_SIZES[2:5]:
                add(f"{tag}_{dpi}dpi_{points:g}pt", dpi, points, font=font)
        add("sans_72dpi_9pt_jpeg60", 72, 9.0, fmt="jpg", jpeg_quality=60)
        add("sans_100dpi_9pt_jpeg60", 100, 9.0, fmt="jpg", jpeg_quality=60)
        add("sans_100dpi_9pt_blurred", 100, 9.0, fmt="jpg", jpeg_quality=80, blur_radius=0.6)
    return cases


def _read_axes(fig_dir: Path, truth: dict) -> list[dict]:
    """Read both axis scales of a single-panel validation figure and score them."""
    from .axis_labels import SMALL_PRINT_PX
    from .calibration import calibrate_y
    from .frames import darkness_map
    from .io import load_image
    from .panel_detection import find_plot_panels
    from .x_calibration import apply_x, calibrate_x

    spec, tp = truth["spec"], truth["panels"][0]
    rgb = load_image(fig_dir / truth["image"])
    base = {"case": spec["name"], "group": spec["group"], "font": spec["font"]}
    panels = find_plot_panels(rgb)[0]
    if len(panels) != 1:
        return [dict(base, axis=axis, outcome="no_plot_area") for axis in ("y", "x")]
    panel, dark = panels[0], darkness_map(rgb)
    rows = []
    for axis in ("y", "x"):
        t0 = time.time()
        row = dict(base, axis=axis)
        if axis == "y":
            cal = calibrate_y(rgb, dark, panel)
            got = None
            if cal.is_calibrated:
                got = (float(cal.y_to_value(tp["bottom"])), float(cal.y_to_value(tp["top"])))
            want, log = tuple(tp["ylim"]), tp["yscale"] == "log"
            row.update(confidence=cal.confidence, n_agree=len(cal.inliers), n_labels=cal.n_labels,
                       label_px=cal.label_px)
        else:
            x_cal = calibrate_x(rgb, dark, panel)
            got = None
            if x_cal is not None:
                got = (float(apply_x(x_cal, tp["left"])), float(apply_x(x_cal, tp["right"])))
                row.update(confidence=x_cal["confidence"], n_agree=len(x_cal["inliers"]),
                           n_labels=x_cal["n_labels"], label_px=x_cal["label_px"])
            want, log = tuple(tp["xlim"]), False
        row.update(true_low=want[0], true_high=want[1], seconds=round(time.time() - t0, 2))
        if got is None:
            row["outcome"] = "declined"
        else:
            if log:
                span = np.log10(want[1] / want[0])
                ok = min(got) > 0 and max(abs(np.log10(g / w)) for g, w in zip(got, want)) <= 0.01 * span
            else:
                ok = max(abs(g - w) for g, w in zip(got, want)) <= 0.01 * (want[1] - want[0])
            row.update(outcome="right" if ok else "wrong", read_low=got[0], read_high=got[1])
        height = row.get("label_px")
        if height is None:
            # No numeric labels were made out: go by the size the digits were drawn at.
            height = row["label_px"] = round(0.72 * spec["dpi"] * spec["fontsize"] / 72.0, 1)
        row["small_print"] = bool(height < SMALL_PRINT_PX)
        rows.append(row)
    return rows


def run_axis_validation(
    out_dir: str | Path,
    cases: Optional[Sequence[FigureSpec]] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Read the axis scales of the :func:`axis_cases` figures and say how each came out.

    Needs Tesseract.  For each axis the scale is read exactly as
    :func:`trace_digitiser.digitise` reads it, and the outcome is one of

    ``"right"``
        the values read for both ends of the axis are within 1 % of its length;
    ``"declined"``
        no scale was offered: the labels could not be read with any assurance;
    ``"wrong"``
        a scale was offered, and it is off.

    Returns one row per axis (``case``, ``group``, ``axis``, ``outcome``,
    ``confidence``, ``n_agree``, ``n_labels``, ``label_px``, ...), also
    written to ``out_dir/axis_results.csv``.  Summarise it with
    :func:`summarise_axes`.
    """
    from .axis_labels import tesseract_available

    if not tesseract_available():
        raise RuntimeError("reading axis scales needs Tesseract, which was not found")
    out_dir = Path(out_dir)
    fig_dir = out_dir / "axis_figures"
    rows: list[dict] = []
    todo = list(cases if cases is not None else axis_cases())
    started = time.time()
    for k, spec in enumerate(todo, start=1):
        truth = render_figure(spec, fig_dir)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows += _read_axes(fig_dir, truth)
        except Exception as exc:  # a crash is a result too
            traceback.print_exc()
            rows += [
                {"case": spec.name, "group": spec.group, "font": spec.font, "axis": axis, "outcome": "crash",
                 "error": f"{type(exc).__name__}: {exc}"[:200]}
                for axis in ("y", "x")
            ]
        if progress and (k % 25 == 0 or k == len(todo)):
            print(f"[{time.time() - started:5.0f}s] {k} of {len(todo)} figures", flush=True)
    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "axis_results.csv", index=False)
    return results


def summarise_axes(results: pd.DataFrame) -> pd.DataFrame:
    """Count the outcomes of :func:`run_axis_validation` by the size of the print.

    One row for small print (labels under 8 pixels high, which are never
    given high confidence), one for larger print, and one for all axes.
    ``right_high`` and ``right_flagged`` split the correct readings by
    whether a warning asking the user to check them was attached;
    ``wrong_high`` counts wrong scales given with high confidence, the
    failure that matters most, and ``wrong_flagged`` those given with a
    warning.
    """
    r = results.copy()
    for name in ("confidence", "small_print"):
        if name not in r:
            r[name] = None
    high = r["confidence"].eq("high")
    r["right_high"] = r["outcome"].eq("right") & high
    r["right_flagged"] = r["outcome"].eq("right") & ~high
    r["declined"] = r["outcome"].eq("declined")
    r["wrong_flagged"] = r["outcome"].eq("wrong") & ~high
    r["wrong_high"] = r["outcome"].eq("wrong") & high
    r["failed"] = ~r["outcome"].isin(["right", "declined", "wrong"])
    r["print"] = np.where(r["small_print"].fillna(False).astype(bool), "under 8 px", "8 px or more")
    columns = ["right_high", "right_flagged", "declined", "wrong_flagged", "wrong_high", "failed"]
    by_size = r.groupby("print")[columns].sum()
    by_size.insert(0, "axes", r.groupby("print").size())
    total = r[columns].sum().to_frame().T
    total.insert(0, "axes", len(r))
    total.index = ["all"]
    out = pd.concat([by_size.reindex([k for k in ("8 px or more", "under 8 px") if k in by_size.index]), total])
    out.index.name = "label height"
    return out.astype(int).reset_index()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m trace_digitiser.validation OUT_DIR``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m trace_digitiser.validation",
        description="Digitise figures drawn from known data and report the errors.",
    )
    parser.add_argument("out_dir", type=Path, help="Directory for figures and results.")
    parser.add_argument("--only", default="", help="Comma-separated prefixes of case or group names to run.")
    parser.add_argument(
        "--axes", action="store_true",
        help="Test the reading of axis scales from tick labels (750 axes) instead of the digitisation of traces.",
    )
    args = parser.parse_args(argv)
    wanted = tuple(p.strip() for p in args.only.split(",") if p.strip())
    if args.axes:
        axes = [c for c in axis_cases() if not wanted or c.name.startswith(wanted) or c.group.startswith(wanted)]
        axis_results = run_axis_validation(args.out_dir, cases=axes, progress=True)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print()
            print(summarise_axes(axis_results).to_string(index=False))
        print(f"\nPer-axis results: {args.out_dir / 'axis_results.csv'}")
        return 0
    cases = default_cases()
    if wanted:
        cases = [c for c in cases if c.name.startswith(wanted) or c.group.startswith(wanted)]
    results = run_validation(args.out_dir, cases=cases, progress=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print()
        print(summarise(results).to_string(index=False))
    print(f"\nPer-trace results: {args.out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
