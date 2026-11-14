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

import warnings as _warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .axis_labels import SMALL_PRINT_PX, tesseract_available
from .calibration import (
    calibrate_y,
    calibration_from_points,
    calibration_from_range,
    normalised_calibration,
    propagate_y_calibration_across_rows,
)
from .diagnostics import (
    draw_digitised_trace,
    draw_panel_overlay,
    draw_result_overlay,
    draw_trace_mask,
    print_panel_summary,
)
from .digitise import digitise_trace_mask
from .frames import darkness_map
from .io import build_panel_metadata, load_image, save_outputs
from .models import Calibration, DigitiserResult, Panel, ResampledResult, Trace, XLabel
from .ocr import detect_x_labels
from .panel_detection import find_plot_panels
from .provenance import build_provenance, package_version, write_provenance
from .straighten import MIN_TILT_DEGREES, estimate_tilt
from .straighten import straighten as _turn_upright
from .summarise import summarise_by_detected_labels
from .trace_detection import detect_trace_masks
from .x_calibration import calibrate_x, x_calibration_from_points, x_calibration_from_range

__version__ = package_version()

__all__ = [
    "digitise",
    "Calibration",
    "DigitiserResult",
    "DigitiserWarning",
    "Panel",
    "ResampledResult",
    "Trace",
    "XLabel",
    "__version__",
]

Box = Sequence[float]
Points = Sequence[Sequence[float]]


class DigitiserWarning(UserWarning):
    """Something about a digitisation that the user should check."""


def _print_size(cal: dict, template: str) -> str:
    """*template* filled with the height of an axis's labels if they are small print, else nothing."""
    height = cal.get("label_px")
    return template.format(f"{height:.0f}") if height is not None and height < SMALL_PRINT_PX else ""


def _doubt(cal: dict) -> str:
    """The middle of a warning about an axis scale read from tick labels with less than high confidence."""
    values = ", ".join(f"{v:g}" for _, v in sorted(cal["inliers"]))
    return (
        f"was read from {len(cal['inliers'])} of {cal.get('n_labels', '?')} tick labels ({values})"
        f"{_print_size(cal, '; the print is only {} pixels high')}. Confidence in it is "
        f"{cal.get('confidence') or 'unknown'}: check those values against the figure,"
    )


def _tidy(value: float) -> str:
    """A value read off an axis, for a message: 299.6 is written 300, 0.0497 stays 0.0497."""
    if value == 0 or not np.isfinite(value):
        return f"{value:g}"
    return f"{float(f'{value:.3g}'):g}"


def _as_boxes(value, name: str) -> Optional[list[tuple[float, float, float, float]]]:
    """Accept one ``(x_left, y_top, x_right, y_bottom)`` box or a list of them."""
    if value is None:
        return None
    message = f"{name} must be (x_left, y_top, x_right, y_bottom) or a list of such boxes"
    seq = list(value)
    if seq and all(isinstance(v, (int, float, np.integer, np.floating)) for v in seq):
        seq = [seq]  # a single box
    boxes = []
    for box in seq:
        try:
            box = tuple(float(v) for v in box)
        except TypeError:
            raise ValueError(message) from None
        if len(box) != 4:
            raise ValueError(message)
        boxes.append(box)
    return boxes


def _points_for_panel(ref, panel_id: int, n_panels: int, name: str) -> Optional[list[tuple[float, float]]]:
    """Pick the reference points that apply to one panel.

    *ref* is either a list of ``(pixel, value)`` pairs (allowed only when the
    figure has a single panel, since pixel positions belong to one panel) or
    a dict mapping panel number to such a list.
    """
    if ref is None:
        return None
    if isinstance(ref, dict):
        points = ref.get(panel_id)
        return [tuple(float(v) for v in pt) for pt in points] if points else None
    if n_panels > 1:
        raise ValueError(
            f"{name} was given as one list of points but the figure has {n_panels} panels; "
            f"give a dict keyed by panel number, e.g. {name}={{1: [...], 2: [...]}}"
        )
    return [tuple(float(v) for v in pt) for pt in ref]


def _neighbour_limits(panels: list[dict]) -> dict[int, tuple[Optional[float], Optional[float]]]:
    """For each panel, how far its tick labels can extend before meeting another panel.

    Returns ``{panel_id: (column limit for y labels, row limit for x labels)}``.
    """
    limits: dict[int, tuple[Optional[float], Optional[float]]] = {}
    for p in panels:
        e = p["edges"]
        left_of = [
            q["edges"]["right"] for q in panels
            if q is not p and q["edges"]["right"] < e["left"]
            and q["edges"]["top"] < e["bottom"] and q["edges"]["bottom"] > e["top"]
        ]
        below = [
            q["edges"]["top"] for q in panels
            if q is not p and q["edges"]["top"] > e["bottom"]
            and q["edges"]["left"] < e["right"] and q["edges"]["right"] > e["left"]
        ]
        limits[p["panel_id"]] = (max(left_of) + 3 if left_of else None, min(below) - 3 if below else None)
    return limits


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
    # ------------------------------------------------------------------
    # 0. Inputs
    # ------------------------------------------------------------------
    if x_min is None and x_max is None and total_days is not None:
        x_min, x_max = 0.0, float(total_days)
    for name, value in (("y_scale", y_scale), ("x_scale", x_scale)):
        if value not in ("auto", "linear", "log"):
            raise ValueError(f"{name} must be 'auto', 'linear' or 'log'")
    if (y_min is None) != (y_max is None):
        raise ValueError("give both y_min and y_max, or neither")
    if (x_min is None) != (x_max is None):
        raise ValueError("give both x_min and x_max, or neither")
    plot_boxes = _as_boxes(plot_area, "plot_area")
    exclude_boxes = _as_boxes(exclude, "exclude")
    if straighten is True:
        straighten = "auto"
    elif straighten is None or straighten is False or straighten == "never":
        straighten = False
    elif straighten != "auto":
        try:
            straighten = float(straighten)
        except (TypeError, ValueError):
            raise ValueError("straighten must be 'auto', False or a number of degrees") from None
        if not abs(straighten) <= 45:
            raise ValueError("straighten must be between -45 and 45 degrees")

    if isinstance(image_path, np.ndarray):
        rgb = np.ascontiguousarray(image_path)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("an image array must be RGB, shape (height, width, 3), dtype uint8")
        source_path: Optional[Path] = None
    else:
        source_path = Path(image_path)
        rgb = load_image(source_path)
    if output_prefix is None:
        output_prefix = source_path.stem if source_path is not None else "figure"

    parameters = {
        "layout_mode": layout_mode, "expected_rows": expected_rows, "expected_cols": expected_cols,
        "expected_panels": expected_panels, "plot_area": plot_boxes,
        "y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max,
        "y_ref": y_ref, "x_ref": x_ref, "y_scale": y_scale, "x_scale": x_scale,
        "trace_colours": list(trace_colours) if trace_colours else None,
        "exclude": exclude_boxes, "achromatic": achromatic, "ocr": ocr, "straighten": straighten,
    }

    notes: list[str] = []

    def note(message: str) -> None:
        notes.append(message)
        _warnings.warn(message, DigitiserWarning, stacklevel=3)

    diag_dir: Optional[Path] = None
    if save_diagnostics and write_outputs:
        diag_dir = Path(output_dir or ".") / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

    if show_debug:
        print("Processing:", source_path if source_path is not None else "<image array>")
        print("Image size:", rgb.shape[1], "×", rgb.shape[0])

    # ------------------------------------------------------------------
    # 0b. A figure that is askew is turned upright first
    # ------------------------------------------------------------------
    tilt_found, turned_by = 0.0, 0.0
    if straighten == "auto":
        tilt_found = estimate_tilt(rgb)
        if abs(tilt_found) >= MIN_TILT_DEGREES:
            which = "anticlockwise" if tilt_found > 0 else "clockwise"
            measured = [
                name for name, given in
                (("plot_area", plot_boxes), ("y_ref", y_ref), ("x_ref", x_ref), ("exclude", exclude_boxes)) if given
            ]
            if measured:
                note(
                    f"The figure is tilted {abs(tilt_found):.2f} degrees {which}. It has not been straightened, "
                    f"because {' and '.join(measured)} give{'s' if len(measured) == 1 else ''} pixel positions in "
                    "the image as it is; values read from a tilted figure are less accurate. Straighten the "
                    f"image first, or pass straighten={tilt_found:.2f} and positions measured on the result."
                )
            else:
                turned_by = tilt_found
    elif straighten is not False:
        turned_by = float(straighten)
    straightened_png: Optional[Path] = None
    if turned_by != 0:
        rgb = _turn_upright(rgb, turned_by)
        if straighten == "auto":
            which = "anticlockwise" if turned_by > 0 else "clockwise"
            note(
                f"The figure was tilted {abs(turned_by):.2f} degrees {which} and has been straightened. Pixel "
                "positions in the output refer to the straightened image (result.rgb"
                + (f", saved as {output_prefix}_automated_straightened.png" if write_outputs else "")
                + ")."
            )

    dark = darkness_map(rgb)

    # ------------------------------------------------------------------
    # 1. Plot areas
    # ------------------------------------------------------------------
    panel_notes: list[str] = []
    panels, h_lines, v_lines = find_plot_panels(
        rgb,
        layout_mode=layout_mode,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        expected_panels=expected_panels,
        plot_area=plot_boxes,
        notes=panel_notes,
    )
    for message in panel_notes:
        note(message)

    if show_debug or diag_dir is not None:
        if show_debug:
            print_panel_summary(panels, h_lines, v_lines, layout_mode)
        draw_panel_overlay(rgb, panels, h_lines, v_lines, output_dir=diag_dir, show=show_debug)

    limits = _neighbour_limits(panels)
    n_panels = len(panels)

    # ------------------------------------------------------------------
    # 2. Y-axis calibration: reference points, else limits, else OCR
    # ------------------------------------------------------------------
    for p in panels:
        pid = p["panel_id"]
        points = _points_for_panel(y_ref, pid, n_panels, "y_ref")
        if points is not None:
            cal = calibration_from_points(points, scale="log" if y_scale == "log" else "linear")
        elif y_min is not None:
            cal = calibration_from_range(p, y_min, y_max, scale="log" if y_scale == "log" else "linear")
            if not p.get("has_top", True):
                note(
                    f"Panel {pid}: the plot has no line along the top, so y_max={y_max:g} was placed at the "
                    "upper end of the y-axis line. If the axis runs past its last tick, give y_ref instead."
                )
        elif ocr:
            cal = calibrate_y(rgb, dark, p, limit=limits[pid][0], scale=y_scale, verbose=show_debug)
        else:
            cal = normalised_calibration(p)
        p["y_calibration"] = cal.to_dict()

    if y_ref is None and y_min is None:
        panels = propagate_y_calibration_across_rows(panels, verbose=show_debug)

    for p in panels:
        cal = p["y_calibration"]
        pid = p["panel_id"]
        if cal["scale_type"] == "normalised_0_to_1":
            if not ocr:
                why = "OCR is switched off"
            elif not tesseract_available():
                why = "Tesseract is not installed, so its tick labels cannot be read"
            elif cal.get("n_labels", 0) >= 2:
                why = (
                    f"{cal['n_labels']} tick labels were found, but too few of them could be read for "
                    f"their values to agree on a scale{_print_size(cal, ' (the print is only {} pixels high)')}"
                )
            else:
                why = "no tick labels were found beside it"
            note(
                f"Panel {pid}: the y axis is not calibrated - {why}. Values run from 0 at the bottom of "
                "the plot area to 1 at the top. Give y_min and y_max, or y_ref."
            )
        elif cal.get("method", "").startswith("tick_labels"):
            if cal.get("confidence") != "high":
                note(f"Panel {pid}: the y-axis scale {_doubt(cal)} or give y_min and y_max.")
            if cal.get("orientation_flipped"):
                y_cal = Calibration(**cal)
                top, bottom = (_tidy(float(y_cal.y_to_value(p["edges"][k]))) for k in ("top", "bottom"))
                note(
                    f"Panel {pid}: the y axis was read as running downwards, from {top} at the top of the plot "
                    f"area to {bottom} at the bottom. If its labels are negative numbers, their minus signs were "
                    "not read: give y_min and y_max."
                )

    # ------------------------------------------------------------------
    # 3. X-axis calibration: reference points, else limits, else OCR
    # ------------------------------------------------------------------
    for p in panels:
        pid = p["panel_id"]
        points = _points_for_panel(x_ref, pid, n_panels, "x_ref")
        if points is not None:
            p["x_calibration"] = x_calibration_from_points(points, scale="log" if x_scale == "log" else "linear")
        elif x_min is not None:
            p["x_calibration"] = x_calibration_from_range(
                p, x_min, x_max, scale="log" if x_scale == "log" else "linear"
            )
            if not p.get("has_right", True):
                note(
                    f"Panel {pid}: the plot has no line down the right-hand side, so x_max={x_max:g} was placed "
                    "at the right-hand end of the x-axis line. If the axis runs past its last tick, give x_ref instead."
                )
        elif ocr:
            x_cal = calibrate_x(rgb, dark, p, limit=limits[pid][1], scale=x_scale)
            if x_cal is not None:
                p["x_calibration"] = x_cal
                if x_cal.get("confidence") != "high":
                    note(f"Panel {pid}: the x-axis scale {_doubt(x_cal)} or give x_min and x_max.")

    uncalibrated_x = [p["panel_id"] for p in panels if "x_calibration" not in p]
    if uncalibrated_x:
        if n_panels == 1:
            which = "the plot"
        elif len(uncalibrated_x) == 1:
            which = f"panel {uncalibrated_x[0]}"
        else:
            which = "panels " + ", ".join(str(i) for i in uncalibrated_x[:-1]) + f" and {uncalibrated_x[-1]}"
        note(
            f"The x axis of {which} is not calibrated (no numeric scale could be read from its labels); the "
            "output has x_pixel and x_norm_0_to_1 but no x_value. Give x_min and x_max, or x_ref."
        )

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
