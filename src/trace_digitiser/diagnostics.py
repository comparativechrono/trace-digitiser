"""Diagnostic overlays and optional file output.

All diagnostic functions accept an optional ``output_dir``.  When
provided, overlays and CSVs are written there; otherwise they are
displayed inline via matplotlib (for notebook / interactive use).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_panel_overlay(
    rgb: np.ndarray,
    panels: list[dict],
    h_lines: list[dict],
    v_lines: list[dict],
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> np.ndarray:
    """Draw detected lines and panel boxes on the image.

    Returns the overlay image.
    """
    overlay = rgb.copy()

    for line in h_lines:
        cv2.line(overlay, (line["x0"], line["y"]), (line["x1"], line["y"]), (255, 0, 0), 2)

    for line in v_lines:
        cv2.line(overlay, (line["x"], line["y0"]), (line["x"], line["y1"]), (0, 0, 255), 2)

    for p in panels:
        cv2.rectangle(overlay, (p["x0"], p["y_top"]), (p["x1"], p["y_bottom"]), (0, 255, 0), 4)
        cv2.putText(
            overlay,
            str(p["panel_id"]),
            (p["x0"] + 5, max(25, p["y_top"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 128, 0),
            3,
            cv2.LINE_AA,
        )

    if output_dir is not None:
        out_path = Path(output_dir) / "panel_detection_overlay.png"
        plt.imsave(str(out_path), overlay)

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title("Detected primary horizontals, axis-like verticals, and final panels")
        plt.show()

    return overlay


def draw_trace_mask(
    rgb: np.ndarray,
    panel: dict,
    trace: dict,
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Highlight detected trace pixels inside a panel ROI."""
    roi = rgb[panel["y_top"] : panel["y_bottom"] + 1, panel["x0"] : panel["x1"] + 1].copy()
    overlay = roi.copy()
    overlay[trace["mask"]] = [255, 0, 0]

    if output_dir is not None:
        out_path = Path(output_dir) / f"trace_mask_panel_{panel['panel_id']}_trace_{trace['trace_id']}.png"
        plt.imsave(str(out_path), overlay)

    if show:
        plt.figure(figsize=(9, 3))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"Panel {panel['panel_id']} trace {trace['trace_id']} detected pixels")
        plt.show()


def draw_digitised_trace(
    trace_data: pd.DataFrame,
    panel_id: int,
    trace_id: int,
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot the digitised center/upper/lower envelopes for one trace."""
    sub = trace_data[(trace_data["panel_id"] == panel_id) & (trace_data["trace_id"] == trace_id)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sub["x_norm_0_to_1"], sub["center_estimate"], label="center estimate")
    ax.plot(sub["x_norm_0_to_1"], sub["upper_envelope"], alpha=0.5, label="upper envelope")
    ax.plot(sub["x_norm_0_to_1"], sub["lower_envelope"], alpha=0.5, label="lower envelope")
    ax.set_title(f"Digitised panel {panel_id}, trace {trace_id}")
    ax.set_xlabel("Normalised x position")
    ax.set_ylabel("Digitised y value")
    ax.legend()

    if output_dir is not None:
        out_path = Path(output_dir) / f"digitised_trace_panel_{panel_id}_trace_{trace_id}.png"
        fig.savefig(str(out_path), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_panel_summary(panels: list[dict], h_lines: list[dict], v_lines: list[dict], layout_mode: str) -> None:
    """Print a text summary of panel detection results."""
    # Count lines from primary detector only.
    legacy_h = [line for line in h_lines if line.get("source") == "legacy_row_projection_segmented"]
    axis_h = [line for line in h_lines if line.get("source") == "axis_low_saturation"]

    print(f"Detected {len(legacy_h)} primary horizontal lines")
    print(f"Detected {len(axis_h)} axis-like horizontal lines")
    print(f"Detected {len(v_lines)} axis-like vertical lines")
    print(f"Detected {len(panels)} plot panel(s) after layout hint: {layout_mode}")

    for p in panels:
        print(
            {
                "panel_id": p["panel_id"],
                "x0": p["x0"],
                "x1": p["x1"],
                "y_top": p["y_top"],
                "y_bottom": p["y_bottom"],
                "source": p["source"],
                "score": round(p["score"], 2),
                "gridlines": len(p["gridline_y"]),
            }
        )


def plot_traces(
    trace_data: pd.DataFrame,
    panels: list,
    output_path: Optional[Path] = None,
    show: bool = True,
    show_envelope: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: str = "Digitised value",
) -> Optional[plt.Figure]:
    """Plot digitised traces, recreating the original figure layout.

    Produces one subplot per panel, stacked vertically, with traces
    drawn in their detected colour.

    Parameters
    ----------
    trace_data : pd.DataFrame
        The ``trace_data`` DataFrame from :class:`DigitiserResult`,
        or a resampled DataFrame.
    panels : list
        The ``panels`` list from :class:`DigitiserResult`.
    output_path : Path, optional
        If given, save the figure to this file.
    show : bool
        If True, display the figure interactively.
    show_envelope : bool
        If True, shade the area between upper and lower envelopes.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
    title : str, optional
        Overall figure title.
    x_label : str, optional
        Override the x-axis label (e.g. ``"Time (hours)"``).
        If None, auto-detected from the data.
    y_label : str
        The y-axis label.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if trace_data.empty:
        return None

    panel_ids = sorted(trace_data["panel_id"].unique())
    n_panels = len(panel_ids)

    if figsize is None:
        figsize = (10, 3.0 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False, sharex=True)

    # Build a lookup for panel x-labels.
    panel_lookup = {}
    for p in panels:
        pid = p.panel_id if hasattr(p, "panel_id") else p.get("panel_id")
        panel_lookup[pid] = p

    for row_idx, pid in enumerate(panel_ids):
        ax = axes[row_idx, 0]
        panel_traces = trace_data[trace_data["panel_id"] == pid]

        for tid, sub in panel_traces.groupby("trace_id"):
            # Choose x-axis: prefer calibrated x values if available.
            if "x_value" in sub.columns and sub["x_value"].notna().any():
                x = sub["x_value"].values
                if x_label is None:
                    x_label = "x value"
            else:
                x = sub["x_norm_0_to_1"].values
                if x_label is None:
                    x_label = "Normalised x position"

            y = sub["center_estimate"].values

            # Use detected trace colour.
            rgb_str = sub["trace_median_rgb"].iloc[0]
            try:
                rgb_vals = ast.literal_eval(rgb_str) if isinstance(rgb_str, str) else rgb_str
                colour = tuple(float(v) / 255.0 for v in rgb_vals[:3])
            except (ValueError, SyntaxError, TypeError):
                colour = None

            ax.plot(x, y, linewidth=0.9, alpha=0.9, color=colour, label=f"Trace {tid}")

            if show_envelope and "upper_envelope" in sub.columns:
                upper = sub["upper_envelope"].values
                lower = sub["lower_envelope"].values
                ax.fill_between(x, lower, upper, alpha=0.15, color=colour)

        # Label the x axis with the tick labels read from the figure, if any.
        p = panel_lookup.get(pid)
        if p is not None and "x_value" in panel_traces.columns:
            x_labels = p.x_labels if hasattr(p, "x_labels") else p.get("x_labels", [])
            # One row per pixel column, in order, so that positions can be interpolated.
            columns = panel_traces[panel_traces["x_value"].notna()].drop_duplicates("x_pixel").sort_values("x_pixel")
            if x_labels and len(columns) >= 2:
                positions, texts = [], []
                for lab in x_labels:
                    lx = lab.x if hasattr(lab, "x") else lab.get("x")
                    positions.append(float(np.interp(lx, columns["x_pixel"], columns["x_value"])))
                    texts.append(lab.text if hasattr(lab, "text") else lab.get("text"))
                ax.set_xticks(positions)
                ax.set_xticklabels(texts, fontsize=7)

        ax.set_ylabel(y_label)
        ax.set_title(f"Panel {pid}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(x_label)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Result overlay: the digitised traces drawn back onto the figure
# ---------------------------------------------------------------------------

def _get(obj, name, default=None):
    """Read a field from a Panel dataclass or a panel dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _y_calibration(panel):
    """Return ``(a, b, is_log)`` for a Panel dataclass or panel dict, or None."""
    cal = _get(panel, "calibration")
    if cal is None:
        cal = _get(panel, "y_calibration")
    if cal is None:
        return None
    a, b = _get(cal, "a"), _get(cal, "b")
    return float(a), float(b), str(_get(cal, "scale_type", "")).startswith("log")


def draw_result_overlay(
    rgb: np.ndarray,
    panels: list,
    trace_data: pd.DataFrame,
    output_path: Optional[Path] = None,
    show: bool = False,
    min_long_side: int = 1400,
) -> np.ndarray:
    """Draw what was digitised on top of the original figure.

    The overlay shows, for each panel, the plot area that was used (green
    box), the tick labels the y calibration rests on (magenta marks with
    the value read), and for each trace its centre estimate (a solid line)
    with the upper and lower envelopes (thin lines).  Columns where the
    trace was not seen and the values were interpolated are drawn in grey.

    If the digitisation is right, every drawn line lies on the trace it
    belongs to.  Small figures are enlarged (without smoothing) so that
    single pixels can be checked.

    Returns the overlay as an RGB array, and writes it to *output_path* if
    one is given.
    """
    h, w = rgb.shape[:2]
    scale = int(np.clip(np.ceil(min_long_side / max(h, w)), 1, 6))
    canvas = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    shift = 4  # sub-pixel precision for OpenCV drawing
    one = 1 << shift

    def to_canvas(x, y) -> np.ndarray:
        """Image coordinates (pixel centres at integers) to fixed-point canvas coordinates."""
        xs = (np.asarray(x, dtype=float) + 0.5) * scale - 0.5
        ys = (np.asarray(y, dtype=float) + 0.5) * scale - 0.5
        return np.round(np.stack([xs, ys], axis=-1) * one).astype(np.int32)

    def polyline(xs, ys, colour, thickness) -> None:
        ok = np.isfinite(xs) & np.isfinite(ys)
        # Draw each unbroken stretch separately.
        idx = np.flatnonzero(ok)
        if idx.size < 2:
            return
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if run.size >= 2:
                cv2.polylines(canvas, [to_canvas(xs[run], ys[run])], False, colour, thickness, cv2.LINE_AA, shift)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, 0.38 * scale * max(0.6, min(1.0, max(h, w) / 800)))
    thick = max(1, int(round(scale * 0.5)))

    for panel in panels:
        pid = _get(panel, "panel_id")
        edges = _get(panel, "edges") or {
            "left": _get(panel, "x0"), "right": _get(panel, "x1"),
            "top": _get(panel, "y_top"), "bottom": _get(panel, "y_bottom"),
        }
        corners = to_canvas(
            [edges["left"], edges["right"], edges["right"], edges["left"]],
            [edges["top"], edges["top"], edges["bottom"], edges["bottom"]],
        )
        cv2.polylines(canvas, [corners], True, (0, 170, 0), thick, cv2.LINE_AA, shift)
        text = f"panel {pid}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, max(1, thick))
        origin = (
            int((edges["right"] + 0.5) * scale) - text_w - 2 * scale,
            int((edges["top"] + 0.5) * scale) - 2 * scale,
        )
        cv2.putText(canvas, text, origin, font, font_scale, (0, 120, 0), max(1, thick), cv2.LINE_AA)

        cal = _get(panel, "calibration") or _get(panel, "y_calibration")
        for row, value in (_get(cal, "inliers", []) if cal is not None else []):
            a_pt = to_canvas(edges["left"] - 6, row)
            b_pt = to_canvas(edges["left"] + 6, row)
            cv2.line(
                canvas,
                tuple(int(v) for v in a_pt),
                tuple(int(v) for v in b_pt),
                (200, 0, 200),
                thick,
                cv2.LINE_AA,
                shift,
            )
            cv2.putText(
                canvas, f"{value:g}",
                (int((edges["left"] + 8) * scale), int((row + 0.5) * scale) + int(4 * font_scale)),
                font, font_scale * 0.85, (200, 0, 200), max(1, thick), cv2.LINE_AA,
            )

        ycal = _y_calibration(panel)
        if ycal is None or trace_data.empty:
            continue
        a, b, is_log = ycal
        sub_panel = trace_data[trace_data["panel_id"] == pid]
        for _, sub in sub_panel.groupby("trace_id"):
            sub = sub.sort_values("x_pixel")
            xs = sub["x_pixel"].to_numpy(dtype=float)

            def rows(column: str, sub=sub, a=a, b=b, is_log=is_log) -> np.ndarray:
                """Image rows at which the values in *column* were read."""
                v = sub[column].to_numpy(dtype=float)
                if is_log:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        v = np.log10(v)
                return (v - b) / a

            try:
                r, g, bl = ast.literal_eval(str(sub["trace_median_rgb"].iloc[0]))[:3]
                dark_trace = (max(r, g, bl) - min(r, g, bl) < 40) and (r + g + bl) / 3 < 140
            except (ValueError, SyntaxError, TypeError):
                dark_trace = False
            main = (255, 90, 0) if dark_trace else (0, 0, 0)
            seen = ~sub["interpolated"].to_numpy(dtype=bool) if "interpolated" in sub else np.ones(len(sub), bool)
            centre = rows("center_estimate")
            upper, lower = rows("upper_envelope"), rows("lower_envelope")
            polyline(xs, np.where(seen, upper, np.nan), (255, 0, 255), max(1, thick // 2))
            polyline(xs, np.where(seen, lower, np.nan), (0, 200, 255), max(1, thick // 2))
            polyline(xs, centre, (150, 150, 150), thick)
            polyline(xs, np.where(seen, centre, np.nan), main, thick)

    if output_path is not None:
        from PIL import Image

        Image.fromarray(canvas).save(str(output_path))
    if show:
        plt.figure(figsize=(12, 12 * h / max(1, w)))
        plt.imshow(canvas)
        plt.axis("off")
        plt.title("Digitised traces drawn on the original figure")
        plt.show()
    return canvas
