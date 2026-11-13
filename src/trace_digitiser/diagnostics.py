"""Diagnostic overlays and optional file output.

All diagnostic functions accept an optional ``output_dir``.  When
provided, overlays and CSVs are written there; otherwise they are
displayed inline via matplotlib (for notebook / interactive use).
"""

from __future__ import annotations

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
    legacy_h = [l for l in h_lines if l.get("source") == "legacy_row_projection_segmented"]
    axis_h = [l for l in h_lines if l.get("source") == "axis_low_saturation"]

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
            # Choose x-axis: prefer x_value (days) if available.
            if "x_value" in sub.columns and sub["x_value"].notna().any():
                x = sub["x_value"].values
                if x_label is None:
                    x_label = "Time (days)"
            else:
                x = sub["x_norm_0_to_1"].values
                if x_label is None:
                    x_label = "Normalised x position"

            y = sub["center_estimate"].values

            # Use detected trace colour.
            rgb_str = sub["trace_median_rgb"].iloc[0]
            try:
                rgb_vals = eval(rgb_str) if isinstance(rgb_str, str) else rgb_str
                colour = tuple(v / 255.0 for v in rgb_vals[:3])
            except Exception:
                colour = None

            ax.plot(x, y, linewidth=0.9, alpha=0.9, color=colour, label=f"Trace {tid}")

            if show_envelope and "upper_envelope" in sub.columns:
                upper = sub["upper_envelope"].values
                lower = sub["lower_envelope"].values
                ax.fill_between(x, lower, upper, alpha=0.15, color=colour)

        # Add x-tick labels from OCR if available.
        p = panel_lookup.get(pid)
        if p is not None:
            x_labels = p.x_labels if hasattr(p, "x_labels") else p.get("x_labels", [])
            if x_labels and "x_value" in panel_traces.columns:
                x_cal = None
                if hasattr(p, "calibration"):
                    # It's a Panel dataclass — check for x_calibration in the raw dict
                    pass
                # Try to get x_calibration from trace data to map label pixel positions
                sub0 = panel_traces.iloc[0:1]
                if "x_value" in sub0.columns:
                    # Map label x-pixel to x_value using linear interpolation
                    all_xpix = panel_traces["x_pixel"].values
                    all_xval = panel_traces["x_value"].values
                    label_texts = []
                    label_positions = []
                    for lab in x_labels:
                        lx = lab.x if hasattr(lab, "x") else lab.get("x")
                        lt = lab.text if hasattr(lab, "text") else lab.get("text")
                        xval = np.interp(lx, all_xpix, all_xval)
                        label_texts.append(lt)
                        label_positions.append(xval)
                    if label_texts:
                        ax.set_xticks(label_positions)
                        ax.set_xticklabels(label_texts, fontsize=7)

        ax.set_ylabel("Activity")
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
