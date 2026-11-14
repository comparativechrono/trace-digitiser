"""trace_digitiser — digitise line plots from figure images, locally.

Everything runs on your own computer: the image is never uploaded to a
server or passed to an AI model.

Quick start::

    from trace_digitiser import digitise

    result = digitise("figure.jpg", output_dir="outputs")

    print(result.trace_data.head())   # one row per pixel column per trace
    for message in result.warnings:   # anything that needs checking
        print("check:", message)
    result.overlay("check.png")       # the result drawn over the figure

Other kinds of figure have readers of their own, which find and read the axes in the same way::

    digitise_points("scatter.png")        # markers, and their error bars
    digitise_bars("bars.png")             # the end of each bar, and its error bar
    digitise_rows("actogram.png", bins=48)  # a stack of rows: actograms, double plots
    digitise_heatmap("heat.png", rows=21)   # cells, by their colour on the colour bar
    digitise_polar("phase.png", r_ref=[(37.5, 1955), (98.2, 1965)])

If the automatic steps need help, each can be set by hand::

    result = digitise(
        "figure.jpg",
        plot_area=(41, 23, 388, 159),     # x_left, y_top, x_right, y_bottom
        y_min=0, y_max=300,               # axis limits ...
        x_ref=[(41, 0), (388, 7)],        # ... or (pixel, value) reference points
        trace_colours=["#578dd2"],
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ._figure import (  # noqa: F401  (the helpers stay importable from here, as before)
    Box,
    DigitiserWarning,
    Points,
    _as_boxes,
    _doubt,
    _neighbour_limits,
    _points_for_panel,
    _print_size,
    _tidy,
    prepare_figure,
)
from .diagnostics import (
    draw_digitised_trace,
    draw_result_overlay,
    draw_trace_mask,
)
from .digitise import digitise_trace_mask
from .io import build_panel_metadata, load_image, save_outputs  # noqa: F401  (load_image: as before)
from .models import Calibration, DigitiserResult, Panel, ResampledResult, Trace, XLabel
from .ocr import detect_x_labels
from .provenance import build_provenance, package_version, write_provenance
from .summarise import summarise_by_detected_labels
from .trace_detection import detect_trace_masks

__version__ = package_version()

__all__ = [
    "digitise",
    "digitise_points",
    "digitise_bars",
    "digitise_rows",
    "digitise_heatmap",
    "digitise_polar",
    "describe_figure",
    "pixels_to_values",
    "find_polar_axes",
    "FigureResult",
    "Calibration",
    "DigitiserResult",
    "DigitiserWarning",
    "Panel",
    "ResampledResult",
    "Trace",
    "XLabel",
    "__version__",
]

def digitise(
    image_path: Union[str, Path, np.ndarray],
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
    plot_area: Optional[Union[Box, Sequence[Box]]] = None,
    y_ref: Optional[Union[Points, dict]] = None,
    x_ref: Optional[Union[Points, dict]] = None,
    y_scale: str = "auto",
    x_scale: str = "auto",
    trace_colours: Optional[Sequence] = None,
    exclude: Optional[Union[Box, Sequence[Box]]] = None,
    achromatic: str = "auto",
    ocr: bool = True,
    save_overlay: bool = False,
    write_outputs: bool = True,
    straighten: Union[str, bool, float] = "auto",
) -> DigitiserResult:
    """Digitise the line traces in a figure image.

    The image is processed entirely on this computer.  Each automatic step
    (finding the plot area, reading the axes, finding the traces) can be
    replaced by an explicit setting; anything the function is unsure of is
    reported in ``result.warnings``.

    Parameters
    ----------
    image_path : str, Path or ndarray
        The figure: a file path, or an RGB image array.

    straighten : "auto", False, or a number of degrees
        What to do about a figure that is tilted, as scans often are.
        ``"auto"`` (default) measures the tilt and, if it is 0.15 degrees or
        more, turns the image upright before anything else is done; pixel
        positions in the output then refer to the straightened image, which
        is saved beside the other files and returned as ``result.rgb``.  A
        figure is not straightened automatically when *plot_area*, *y_ref*,
        *x_ref* or *exclude* give pixel positions, since those were measured
        on the image as it is.  A number turns the image clockwise by that
        many degrees (so give the anticlockwise tilt of the figure);
        ``False`` leaves it alone.

    layout_mode : str
        ``"auto"`` (default), ``"single"``, ``"stacked"``, ``"horizontal"``
        or ``"grid"``.  A hint about how panels are arranged; rarely needed.
    expected_rows, expected_cols, expected_panels : int, optional
        How many panels to expect.  Used to choose among candidates and to
        warn if a different number is found.
    plot_area : (x_left, y_top, x_right, y_bottom), or a list of them, optional
        Pixel positions of the axis lines of each panel, to use instead of
        automatic detection.

    y_min, y_max : float, optional
        The y-axis limits.  If both are supplied OCR is skipped and the
        bottom and top edges of the plot area are taken to be *y_min* and
        *y_max*.  Recommended whenever you know the limits.
    x_min, x_max : float, optional
        The same for the x axis (left and right edges).
    total_days : float, optional
        Shorthand for ``x_min=0, x_max=total_days``.
    y_ref, x_ref : list of (pixel, value), or dict of panel number -> list, optional
        Reference points for an axis: the pixel row (for y) or column (for x)
        of a tick and the value it marks.  Use when the axis runs beyond its
        last tick so that the plot-area edges are not the axis limits.
        Takes precedence over *y_min*/*y_max* and *x_min*/*x_max*.
    y_scale, x_scale : {"auto", "linear", "log"}
        Kind of axis.  ``"auto"`` lets OCR decide and is linear otherwise.

    trace_colours : list of colours, optional
        Colours of the traces to extract (``"#rrggbb"`` or ``(r, g, b)``).
        Skips automatic colour grouping; one trace per colour, in order.
    exclude : (x_left, y_top, x_right, y_bottom), or a list of them, optional
        Regions to ignore when looking for traces, e.g. a legend or an
        annotation drawn inside the plot.
    achromatic : {"auto", "always", "never"}
        Whether to look for a black or grey trace.
    ocr : bool
        Set False to never run Tesseract.  Axes not set by hand are then
        left normalised to 0-1.

    output_dir : str or Path, optional
        Directory for output files.  Defaults to the current directory.
    output_prefix : str, optional
        Prefix for output filenames.  Defaults to the image file's stem.
    write_outputs : bool
        Set False to return the result without writing any files.
    save_overlay : bool
        Also write ``<prefix>_automated_overlay.png``: the digitised traces
        drawn on the original figure, for checking by eye.
    save_diagnostics : bool
        Also write intermediate images (panel detection, trace masks) to a
        ``diagnostics`` sub-directory, and the overlay.
    show_debug : bool
        Print progress and show diagnostic plots (for interactive use).

    Returns
    -------
    DigitiserResult
        ``trace_data`` (one row per pixel column per trace), ``panels``,
        ``warnings``, ``provenance`` and the paths of the files written.
    """
    fig = prepare_figure(
        image_path, layout_mode=layout_mode, expected_rows=expected_rows, expected_cols=expected_cols,
        expected_panels=expected_panels, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max, total_days=total_days,
        output_dir=output_dir, output_prefix=output_prefix, show_debug=show_debug, save_diagnostics=save_diagnostics,
        plot_area=plot_area, y_ref=y_ref, x_ref=x_ref, y_scale=y_scale, x_scale=x_scale, exclude=exclude, ocr=ocr,
        write_outputs=write_outputs, straighten=straighten,
        extra_parameters={"trace_colours": list(trace_colours) if trace_colours else None, "achromatic": achromatic},
    )
    rgb, dark, panels, notes, note = fig.rgb, fig.dark, fig.panels, fig.notes, fig.note
    exclude_boxes, output_prefix, diag_dir = fig.exclude_boxes, fig.output_prefix, fig.diag_dir
    source_path, parameters, tilt_found, turned_by = fig.source_path, fig.parameters, fig.tilt_found, fig.turned_by
    straightened_png: Optional[Path] = None

    # ------------------------------------------------------------------
    # 4. Trace detection and digitisation
    # ------------------------------------------------------------------
    all_trace_frames: list[pd.DataFrame] = []
    trace_debug: list[tuple[dict, dict]] = []

    for p in panels:
        masks = detect_trace_masks(
            rgb, p, colours=trace_colours, exclude=exclude_boxes, achromatic=achromatic, darkness=dark,
        )
        if show_debug:
            print(f"Panel {p['panel_id']}: detected {len(masks)} trace(s)")
        if not masks:
            note(
                f"Panel {p['panel_id']}: no trace was found. If the trace is faint or an unusual colour, "
                "give trace_colours."
            )

        for tr in masks:
            if show_debug:
                print(" ", {k: v for k, v in tr.items() if k != "mask"})
            frame = digitise_trace_mask(p, tr)
            missing = float(frame["interpolated"].mean())
            if missing > 0.25:
                note(
                    f"Panel {p['panel_id']} trace {tr['trace_id']}: {missing:.0%} of columns had no trace pixels "
                    "and were interpolated."
                )
            all_trace_frames.append(frame)
            trace_debug.append((p, tr))

    trace_data = pd.concat(all_trace_frames, ignore_index=True) if all_trace_frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # 5. Interval labels along the x axis (e.g. "1N", "1D") and summaries
    # ------------------------------------------------------------------
    panels_by_y = sorted(panels, key=lambda p: p["y_top"])
    for idx, p in enumerate(panels_by_y):
        # Cap the lower scan band at the top of the next panel below.
        max_y = panels_by_y[idx + 1]["y_top"] if idx + 1 < len(panels_by_y) else None
        p["x_labels"] = detect_x_labels(rgb, p, max_y=max_y) if ocr else []
        if show_debug:
            print(f"Panel {p['panel_id']} x labels:")
            for lab in p["x_labels"]:
                print(f"  {lab['text']:>8s}  x={lab['x']:.1f}  conf={lab['conf']:.1f}")

    summary_by_label = summarise_by_detected_labels(trace_data, panels)

    # ------------------------------------------------------------------
    # 6. Structured result
    # ------------------------------------------------------------------
    result_panels = [
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
            x_labels=[XLabel(**lab) for lab in p.get("x_labels", [])],
            edges=dict(p.get("edges", {})),
            kind=p.get("kind", "box"),
            x_calibration=p.get("x_calibration"),
        )
        for p in panels
    ]
    panel_metadata = build_panel_metadata(panels)
    provenance = build_provenance(source_path, rgb, parameters, panels, trace_debug, trace_data, notes)
    provenance["straightening"] = {
        "tilt_measured_degrees": round(float(tilt_found), 3) if straighten == "auto" else None,
        "turned_clockwise_by_degrees": round(float(turned_by), 3),
    }

    # ------------------------------------------------------------------
    # 7. Files
    # ------------------------------------------------------------------
    trace_csv = meta_csv = summary_csv = overlay_png = provenance_json = None
    if write_outputs:
        trace_csv, meta_csv, summary_csv = save_outputs(
            trace_data, panel_metadata, summary_by_label, output_prefix, output_dir
        )
        base = Path(output_dir) if output_dir else Path(".")
        provenance_json = write_provenance(base / f"{output_prefix}_automated_provenance.json", provenance)
        if turned_by != 0:
            from PIL import Image

            straightened_png = base / f"{output_prefix}_automated_straightened.png"
            Image.fromarray(rgb).save(straightened_png)
        if save_overlay or save_diagnostics:
            overlay_png = base / f"{output_prefix}_automated_overlay.png"
            draw_result_overlay(rgb, result_panels, trace_data, output_path=overlay_png)

    if show_debug or diag_dir is not None:
        for p, tr in trace_debug:
            draw_trace_mask(rgb, p, tr, output_dir=diag_dir, show=show_debug)
        if not trace_data.empty:
            for (panel_id, trace_id), _ in trace_data.groupby(["panel_id", "trace_id"]):
                draw_digitised_trace(trace_data, panel_id, trace_id, output_dir=diag_dir, show=show_debug)

    if show_debug and write_outputs:
        print("Wrote:")
        for path in (trace_csv, meta_csv, summary_csv, provenance_json, overlay_png, straightened_png):
            if path:
                print(" -", path)
        if not summary_csv:
            print(" - no label summary; fewer than two labels detected")

    return DigitiserResult(
        image_path=source_path,
        rgb=rgb,
        panels=result_panels,
        trace_data=trace_data,
        summary_by_label=summary_by_label,
        panel_metadata=panel_metadata,
        trace_csv_path=trace_csv,
        summary_csv_path=summary_csv,
        metadata_csv_path=meta_csv,
        overlay_path=overlay_png,
        straightened_path=straightened_png,
        provenance_path=provenance_json,
        warnings=notes,
        provenance=provenance,
    )


# Readers for other kinds of figure, and helpers for graphical tools (version 2.1)
from ._results import FigureResult  # noqa: E402
from .bars import digitise_bars  # noqa: E402
from .explore import describe_figure, pixels_to_values  # noqa: E402
from .heatmap import digitise_heatmap  # noqa: E402
from .points import digitise_points  # noqa: E402
from .polar import digitise_polar, find_polar_axes  # noqa: E402
from .rows import digitise_rows  # noqa: E402
