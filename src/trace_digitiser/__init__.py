"""trace_digitiser — template-free digitisation of raster scientific line plots.

Quick start::

    from trace_digitiser import digitise

    result = digitise(
        "figure.jpg",
        layout_mode="stacked",
        expected_rows=2,
        expected_cols=1,
        output_dir="outputs",
    )

    print(result.trace_data.head())
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .calibration import propagate_y_calibration_across_rows, robust_y_calibration
from .diagnostics import draw_digitised_trace, draw_panel_overlay, draw_trace_mask, print_panel_summary
from .digitise import digitise_trace_mask
from .io import build_panel_metadata, load_image, save_outputs
from .models import Calibration, DigitiserResult, Panel, ResampledResult, Trace, XLabel
from .ocr import detect_x_labels
from .panel_detection import find_plot_panels
from .summarise import summarise_by_detected_labels
from .trace_detection import detect_trace_masks
from .x_calibration import calibrate_x_axis

__all__ = [
    "digitise",
    "Calibration",
    "DigitiserResult",
    "Panel",
    "Trace",
    "XLabel",
]


def digitise(
    image_path: str | Path,
    *,
    layout_mode: str = "auto",
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    expected_panels: Optional[int] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    total_days: Optional[float] = None,
    output_dir: Optional[str | Path] = None,
    output_prefix: Optional[str] = None,
    show_debug: bool = False,
    save_diagnostics: bool = False,
) -> DigitiserResult:
    """End-to-end chart digitisation pipeline.

    Parameters
    ----------
    image_path : str or Path
        Path to the input raster image.
    layout_mode : str
        ``"auto"``, ``"single"``, ``"stacked"``, ``"horizontal"``, or
        ``"grid"``.
    expected_rows, expected_cols, expected_panels : int, optional
        Layout hints that constrain panel selection.
    y_min, y_max : float, optional
        If both are supplied, bypass OCR y-axis calibration and use a
        linear mapping from the panel's bottom pixel to *y_min* and
        top pixel to *y_max*.  This is recommended for low-resolution
        images where OCR cannot reliably read tick labels.
    x_min, x_max : float, optional
        If both are supplied, bypass OCR x-axis calibration and use a
        linear mapping from the panel's left pixel to *x_min* and
        right pixel to *x_max*.  Works the same way as *y_min*/*y_max*.
    total_days : float, optional
        Convenience shorthand for ``x_min=0, x_max=total_days``.
        If *x_min*/*x_max* are also given, they take precedence.
    output_dir : str or Path, optional
        Directory for CSV outputs.  Defaults to current directory.
    output_prefix : str, optional
        Prefix for output filenames.  Defaults to the image stem.
    show_debug : bool
        If True, display inline diagnostic plots (for interactive use).
    save_diagnostics : bool
        If True, write diagnostic overlay PNGs to *output_dir*.

    Returns
    -------
    DigitiserResult
        Structured result with panels, traces, DataFrames, and paths.
    """
    # Resolve total_days into x_min/x_max if neither is set explicitly.
    if x_min is None and x_max is None and total_days is not None:
        x_min = 0.0
        x_max = float(total_days)
    image_path = Path(image_path)
    rgb = load_image(image_path)

    if output_prefix is None:
        output_prefix = image_path.stem

    diag_dir: Optional[Path] = None
    if save_diagnostics:
        diag_dir = Path(output_dir or ".") / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

    if show_debug:
        print("Processing:", image_path)
        print("Image size:", rgb.shape[1], "×", rgb.shape[0])

    # ------------------------------------------------------------------
    # 1. Detect panels
    # ------------------------------------------------------------------
    panels, h_lines, v_lines = find_plot_panels(
        rgb,
        layout_mode=layout_mode,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        expected_panels=expected_panels,
    )

    if show_debug or save_diagnostics:
        print_panel_summary(panels, h_lines, v_lines, layout_mode)
        draw_panel_overlay(rgb, panels, h_lines, v_lines, output_dir=diag_dir, show=show_debug)

    # ------------------------------------------------------------------
    # 2. Y-axis calibration
    # ------------------------------------------------------------------
    user_y_range = y_min is not None and y_max is not None
    for p in panels:
        if user_y_range:
            # User-supplied axis range — bypass OCR.
            a = (float(y_min) - float(y_max)) / max(1, p["y_bottom"] - p["y_top"])
            b = float(y_max) - a * p["y_top"]
            p["y_calibration"] = Calibration(
                a=a, b=b, scale_type="user_supplied",
            ).to_dict()
            if show_debug:
                print(f"Panel {p['panel_id']}: user-supplied y calibration "
                      f"[{y_min}, {y_max}] → a={a:.6g}, b={b:.4g}")
        else:
            calib = robust_y_calibration(rgb, p, verbose=show_debug)
            p["y_calibration"] = calib.to_dict()

    if not user_y_range:
        panels = propagate_y_calibration_across_rows(panels, verbose=show_debug)

    # ------------------------------------------------------------------
    # 2b. X-axis calibration (user-supplied range or OCR numeric ticks)
    # ------------------------------------------------------------------
    user_x_range = x_min is not None and x_max is not None
    for p in panels:
        if user_x_range:
            # User-supplied x range — linear mapping across panel width.
            pw = max(1, p["x1"] - p["x0"])
            a_x = (float(x_max) - float(x_min)) / pw
            b_x = float(x_min) - a_x * p["x0"]
            p["x_calibration"] = {
                "a": a_x, "b": b_x,
                "scale_type": "user_supplied",
                "inliers": [],
                "ocr_candidates": {},
            }
            if show_debug:
                print(f"Panel {p['panel_id']}: user-supplied x calibration "
                      f"[{x_min}, {x_max}]")
        else:
            x_cal = calibrate_x_axis(rgb, p, verbose=show_debug)
            if x_cal is not None:
                p["x_calibration"] = x_cal

    # ------------------------------------------------------------------
    # 3. Trace detection and digitisation
    # ------------------------------------------------------------------
    all_trace_frames: list[pd.DataFrame] = []
    trace_debug: list[tuple[dict, dict]] = []

    for p in panels:
        masks = detect_trace_masks(rgb, p)
        if show_debug:
            print(f"Panel {p['panel_id']}: detected {len(masks)} coloured trace(s)")

        for tr in masks:
            if show_debug:
                print(" ", {k: v for k, v in tr.items() if k != "mask"})
            all_trace_frames.append(digitise_trace_mask(p, tr))
            trace_debug.append((p, tr))

    trace_data = pd.concat(all_trace_frames, ignore_index=True) if all_trace_frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. X-label OCR
    # ------------------------------------------------------------------
    # Sort panels by vertical position to determine scan boundaries.
    panels_by_y = sorted(panels, key=lambda p: p["y_top"])
    for idx, p in enumerate(panels_by_y):
        # Cap the lower scan band at the top of the next panel below.
        max_y = panels_by_y[idx + 1]["y_top"] if idx + 1 < len(panels_by_y) else None
        p["x_labels"] = detect_x_labels(rgb, p, max_y=max_y)
        if show_debug:
            print(f"Panel {p['panel_id']} x labels:")
            for lab in p["x_labels"]:
                print(f"  {lab['text']:>8s}  x={lab['x']:.1f}  conf={lab['conf']:.1f}")

    # ------------------------------------------------------------------
    # 5. Interval summaries
    # ------------------------------------------------------------------
    summary_by_label = summarise_by_detected_labels(trace_data, panels)

    # ------------------------------------------------------------------
    # 6. Diagnostics
    # ------------------------------------------------------------------
    if show_debug or save_diagnostics:
        for p, tr in trace_debug:
            draw_trace_mask(rgb, p, tr, output_dir=diag_dir, show=show_debug)

        if not trace_data.empty:
            for (panel_id, trace_id), _ in trace_data.groupby(["panel_id", "trace_id"]):
                draw_digitised_trace(trace_data, panel_id, trace_id, output_dir=diag_dir, show=show_debug)

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    panel_metadata = build_panel_metadata(panels)
    trace_csv, meta_csv, summary_csv = save_outputs(
        trace_data, panel_metadata, summary_by_label, output_prefix, output_dir
    )

    if show_debug:
        print("Wrote:")
        print(" -", trace_csv)
        print(" -", meta_csv)
        if summary_csv:
            print(" -", summary_csv)
        else:
            print(" - no label summary; fewer than two labels detected")

    # ------------------------------------------------------------------
    # 8. Build structured result
    # ------------------------------------------------------------------
    return DigitiserResult(
        image_path=image_path,
        rgb=rgb,
        panels=[
            Panel(
                panel_id=p["panel_id"],
                x0=p["x0"],
                x1=p["x1"],
                y_top=p["y_top"],
                y_bottom=p["y_bottom"],
                gridline_y=p["gridline_y"],
                source=p["source"],
                score=p["score"],
                layout_mode=p.get("layout_mode", layout_mode),
                calibration=Calibration(**p["y_calibration"]) if "y_calibration" in p else None,
                x_labels=[
                    XLabel(**lab) for lab in p.get("x_labels", [])
                ],
            )
            for p in panels
        ],
        trace_data=trace_data,
        summary_by_label=summary_by_label,
        panel_metadata=panel_metadata,
        trace_csv_path=trace_csv,
        summary_csv_path=summary_csv,
        metadata_csv_path=meta_csv,
    )
