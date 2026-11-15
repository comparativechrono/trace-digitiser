"""What every way of digitising a figure does first: load the image, straighten it, find the plot areas and
calibrate their axes (from reference points, axis limits or the tick labels).

:func:`trace_digitiser.digitise` (lines) and the readers for points, bars, rows and heat maps all start here, so
that a figure's axes are read, and can be set by hand, in the same way whatever is drawn in it.
"""

from __future__ import annotations

import warnings as _warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .axis_labels import SMALL_PRINT_PX, tesseract_available
from .calibration import (
    calibrate_y,
    calibration_from_points,
    calibration_from_range,
    normalised_calibration,
    propagate_y_calibration_across_rows,
)
from .diagnostics import draw_panel_overlay, print_panel_summary
from .frames import darkness_map
from .io import load_image
from .legend import find_legend
from .models import Calibration
from .panel_detection import find_plot_panels
from .straighten import MIN_TILT_DEGREES, estimate_tilt
from .straighten import straighten as _turn_upright
from .x_calibration import calibrate_x, x_calibration_from_points, x_calibration_from_range


class DigitiserWarning(UserWarning):
    """Something about a digitisation that the user should check."""


Box = Sequence[float]
Points = Sequence[Sequence[float]]


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


def _neighbour_limits(panels: list[dict], right: bool = False) -> dict[int, tuple[Optional[float], Optional[float]]]:
    """For each panel, how far its tick labels can extend before meeting another panel.

    Returns ``{panel_id: (column limit for y labels, row limit for x labels)}``; with *right*, the column limit is
    for labels to the right of the panel (a second y axis).
    """
    limits: dict[int, tuple[Optional[float], Optional[float]]] = {}
    for p in panels:
        e = p["edges"]
        beside = [
            q["edges"]["left"] if right else q["edges"]["right"]
            for q in panels
            if q is not p
            and (q["edges"]["left"] > e["right"] if right else q["edges"]["right"] < e["left"])
            and q["edges"]["top"] < e["bottom"]
            and q["edges"]["bottom"] > e["top"]
        ]
        below = [
            q["edges"]["top"]
            for q in panels
            if q is not p
            and q["edges"]["top"] > e["bottom"]
            and q["edges"]["left"] < e["right"]
            and q["edges"]["right"] > e["left"]
        ]
        column = (min(beside) - 3 if right else max(beside) + 3) if beside else None
        limits[p["panel_id"]] = (column, min(below) - 3 if below else None)
    return limits


def _share_x_calibration(panels: list[dict], note) -> None:
    """Give a panel without x-axis labels the x scale of a panel above or below it that has the same left and right
    edges: plots stacked on one shared x axis (matplotlib's ``sharex``) label only the bottom one."""
    for p in panels:
        if "x_calibration" in p:
            continue
        e = p["edges"]
        same = [
            q
            for q in panels
            if q is not p
            and q.get("x_calibration")
            and str(q["x_calibration"].get("method", "")).startswith("tick_labels")
            and abs(q["edges"]["left"] - e["left"]) <= 2.0
            and abs(q["edges"]["right"] - e["right"]) <= 2.0
        ]
        if not same:
            continue
        src = min(same, key=lambda q: abs(q["edges"]["top"] - e["top"]))
        cal = dict(src["x_calibration"])
        cal["method"] = "propagated"
        cal["propagated_from"] = src["panel_id"]
        p["x_calibration"] = cal
        note(
            f"Panel {p['panel_id']}: its x axis has no labels that could be read, so it was given the x scale of "
            f"panel {src['panel_id']}, which has the same left and right edges (a shared x axis). If the two do not "
            "share an axis, give x_min and x_max, or x_ref."
        )


@dataclass
class PreparedFigure:
    """A figure ready to be read: the image (straightened if it was tilted), its plot areas and their axes."""

    rgb: np.ndarray
    dark: np.ndarray
    panels: list[dict]  # each with "y_calibration" (dict) and, if the x axis is calibrated, "x_calibration"
    notes: list[str]
    note: Callable[[str], None]
    parameters: dict[str, Any]
    source_path: Optional[Path]
    output_prefix: str
    exclude_boxes: Optional[list[tuple[float, float, float, float]]]
    diag_dir: Optional[Path] = None
    tilt_found: float = 0.0
    turned_by: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def prepare_figure(
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
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: Optional[str] = None,
    show_debug: bool = False,
    save_diagnostics: bool = False,
    plot_area=None,
    y_ref=None,
    x_ref=None,
    y_scale: str = "auto",
    x_scale: str = "auto",
    exclude=None,
    ocr: bool = True,
    write_outputs: bool = True,
    straighten: Union[str, bool, float] = "auto",
    y_axis: str = "left",
    legend: Union[str, bool] = "auto",
    extra_parameters: Optional[dict] = None,
    stacklevel: int = 4,
) -> PreparedFigure:
    """Load a figure, straighten it, find its plot areas and calibrate their axes.

    The arguments are those of :func:`trace_digitiser.digitise` that concern the figure and its axes. Messages
    for the user are warned (:class:`DigitiserWarning`) and collected in ``notes``.
    """
    # ------------------------------------------------------------------
    # 0. Inputs
    # ------------------------------------------------------------------
    if x_min is None and x_max is None and total_days is not None:
        x_min, x_max = 0.0, float(total_days)
    for name, value in (("y_scale", y_scale), ("x_scale", x_scale)):
        if value not in ("auto", "linear", "log"):
            raise ValueError(f"{name} must be 'auto', 'linear' or 'log'")
    if y_axis not in ("left", "right"):
        raise ValueError("y_axis must be 'left' or 'right'")
    if legend not in ("auto", "keep", True, False):
        raise ValueError("legend must be 'auto' (find legends drawn inside plots and leave them out) or 'keep'")
    find_legends = legend in ("auto", True)
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
        "layout_mode": layout_mode,
        "expected_rows": expected_rows,
        "expected_cols": expected_cols,
        "expected_panels": expected_panels,
        "plot_area": plot_boxes,
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "y_ref": y_ref,
        "x_ref": x_ref,
        "y_scale": y_scale,
        "x_scale": x_scale,
        "exclude": exclude_boxes,
        "ocr": ocr,
        "straighten": straighten,
    }
    if y_axis != "left":
        parameters["y_axis"] = y_axis
    if not find_legends:
        parameters["legend"] = "keep"
    parameters.update(extra_parameters or {})

    notes: list[str] = []

    def note(message: str) -> None:
        notes.append(message)
        _warnings.warn(message, DigitiserWarning, stacklevel=stacklevel)

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
                name
                for name, given in (
                    ("plot_area", plot_boxes),
                    ("y_ref", y_ref),
                    ("x_ref", x_ref),
                    ("exclude", exclude_boxes),
                )
                if given
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

    limits = _neighbour_limits(panels, right=(y_axis == "right"))
    n_panels = len(panels)
    which_y = "right-hand y axis" if y_axis == "right" else "y axis"

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
            cal = calibrate_y(
                rgb,
                dark,
                p,
                limit=limits[pid][0],
                scale=y_scale,
                verbose=show_debug,
                axis="y2" if y_axis == "right" else "y",
            )
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
                f"Panel {pid}: the {which_y} is not calibrated - {why}. Values run from 0 at the bottom of "
                "the plot area to 1 at the top. Give y_min and y_max, or y_ref."
            )
        elif cal.get("method", "").startswith("tick_labels"):
            if cal.get("confidence") != "high":
                scale_name = "right-hand y-axis scale" if y_axis == "right" else "y-axis scale"
                note(f"Panel {pid}: the {scale_name} {_doubt(cal)} or give y_min and y_max.")
            if cal.get("orientation_flipped"):
                y_cal = Calibration(**cal)
                top, bottom = (_tidy(float(y_cal.y_to_value(p["edges"][k]))) for k in ("top", "bottom"))
                note(
                    f"Panel {pid}: the {which_y} was read as running downwards, from {top} at the top of the plot "
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

    if x_ref is None and x_min is None:
        _share_x_calibration(panels, note)

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
    # 4. A legend drawn inside a plot is left out, as if it had been excluded
    # ------------------------------------------------------------------
    if find_legends:
        found_legends = []
        for p in panels:
            e = p["edges"]
            half = int(p.get("line_thickness", 1)) // 2 + 1
            box = (
                int(np.ceil(e["left"])) + half,
                int(np.ceil(e["top"])) + half,
                int(np.floor(e["right"])) - half,
                int(np.floor(e["bottom"])) - half,
            )
            boxes = [tuple(float(v) for v in b) for b in find_legend(rgb, box)]
            if boxes:
                found_legends.extend(boxes)
                x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
                x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
                note(
                    f"Panel {p['panel_id']}: a legend was found inside the plot area (x {x0:.0f}-{x1:.0f}, y "
                    f"{y0:.0f}-{y1:.0f}) and left out. If that is not a legend, pass legend='keep'."
                )
        if found_legends:
            exclude_boxes = list(exclude_boxes or []) + found_legends
            parameters["legends_found"] = found_legends

    return PreparedFigure(
        rgb=rgb,
        dark=dark,
        panels=panels,
        notes=notes,
        note=note,
        parameters=parameters,
        source_path=source_path,
        output_prefix=output_prefix,
        exclude_boxes=exclude_boxes,
        diag_dir=diag_dir,
        tilt_found=tilt_found,
        turned_by=turned_by,
    )
