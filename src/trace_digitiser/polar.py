"""Reading polar plots: points at an angle and a distance from the centre.

The polar axes are found from their circles: the outer frame and the gridline rings, which share a centre. Each
point's angle is measured from a zero direction (east, to the right, as matplotlib draws by default; or north, as
clock-face plots of phase do) and its distance from the centre is put into the radial axis's units from two or
more rings whose values you give (or the outer circle's).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd

from ._figure import PreparedFigure
from ._results import FigureResult, finish, hex_colour
from .frames import darkness_map
from .io import load_image
from .markers import find_markers
from .provenance import package_version  # noqa: F401  (kept with the other readers' imports)
from .straighten import straighten as _turn_upright
from .trace_detection import parse_colour


def find_polar_axes(rgb: np.ndarray) -> Optional[dict]:
    """The polar axes of a figure: ``{"centre": (x, y), "radius": outer radius, "rings": [radii of the gridline
    circles, in pixels, inner to outer]}``, or None if no circle is found."""
    dark = darkness_map(rgb)
    h, w = dark.shape
    ink = (dark > 60).astype(np.uint8) * 255
    # The circles are proposed on a copy no larger than 1000 pixels a side, and then fitted to the picture itself:
    # on a whole page of a scanned paper (all text), the transform at full size took a minute and a half.
    scale = min(1.0, 1000.0 / max(h, w))
    small = ink if scale >= 1.0 else cv2.resize(ink, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                                                interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5, 5), 1.2)
    sh, sw = small.shape
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=5,
        param1=120,
        param2=40,
        minRadius=int(0.15 * min(sh, sw)),
        maxRadius=int(0.75 * min(sh, sw)),
    )
    if circles is None:
        return None
    circles = circles / scale
    # the frame: of the circles proposed (each first fitted to the dark pixels near it, as the transform places
    # them only roughly), the one drawn darkest all the way round (the larger, of equals)
    angles0 = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ys, xs = np.nonzero(dark > 100)

    def refine(x, y, rad):
        for band in (max(8.0, 0.05 * rad), max(4.0, 0.02 * rad), max(2.5, 0.01 * rad)):
            d = np.hypot(xs - x, ys - y)
            near = np.abs(d - rad) < band
            if near.sum() <= 50:
                break
            px, py = xs[near].astype(float), ys[near].astype(float)
            a = np.column_stack([px, py, np.ones_like(px)])
            sol, *_ = np.linalg.lstsq(a, px * px + py * py, rcond=None)
            x, y = sol[0] / 2, sol[1] / 2
            rad = math.sqrt(max(1.0, sol[2] + x * x + y * y))
        return x, y, rad

    def support(x, y, rad):
        px = np.round(x + rad * np.cos(angles0)).astype(int)
        py = np.round(y + rad * np.sin(angles0)).astype(int)
        if ((px >= 0) & (px < w) & (py >= 0) & (py < h)).mean() < 0.95:
            return 0.0
        darkest = np.zeros(len(angles0))
        for d in (-1, 0, 1):
            qx = np.clip(np.round(x + (rad + d) * np.cos(angles0)).astype(int), 0, w - 1)
            qy = np.clip(np.round(y + (rad + d) * np.sin(angles0)).astype(int), 0, h - 1)
            darkest = np.maximum(darkest, dark[qy, qx])
        return float((darkest > 100).mean())

    scored = []
    for c in circles[0][:25]:
        x, y, rad = refine(*(float(v) for v in c))
        scored.append((support(x, y, rad), rad, (x, y, rad)))
    good = [t for t in scored if t[0] >= 0.6]
    if not good:
        return None
    cx, cy, r = max(good, key=lambda t: (round(t[0], 1), t[1]))[2]
    # rings: radii at which grey ink (gridlines, not the coloured data) runs all round the centre
    mx = rgb.max(axis=2).astype(np.float32)
    grey = (mx - rgb.min(axis=2)) / np.maximum(mx, 1.0) < 0.2
    rr = np.arange(2, int(r) - 1)
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    cover = []
    for radius in rr:
        px = np.clip(np.round(cx + radius * np.cos(angles)).astype(int), 0, w - 1)
        py = np.clip(np.round(cy + radius * np.sin(angles)).astype(int), 0, h - 1)
        cover.append(float(((dark[py, px] > 25) & grey[py, px]).mean()))
    cover = np.array(cover)
    rings = []
    for i in range(1, len(cover) - 1):
        if cover[i] >= 0.5 and cover[i] >= cover[i - 1] and cover[i] >= cover[i + 1]:
            if not rings or rr[i] - rings[-1] > 3:
                rings.append(float(rr[i]))
            elif cover[i] > cover[list(rr).index(int(rings[-1]))]:
                rings[-1] = float(rr[i])
    return {"centre": (float(cx), float(cy)), "radius": float(r), "rings": rings}


def digitise_polar(
    image_path: Union[str, Path, np.ndarray],
    *,
    centre: Optional[Sequence[float]] = None,
    r_ref: Optional[Sequence[Sequence[float]]] = None,
    outer_value: Optional[float] = None,
    inner_value: float = 0.0,
    zero: str = "east",
    clockwise: bool = False,
    colours: Optional[Sequence] = None,
    colour_tolerance: float = 60.0,
    marker_size: Optional[float] = None,
    straighten: Union[str, bool, float] = False,
    output_dir=None,
    output_prefix: Optional[str] = None,
    write_outputs: bool = True,
    save_overlay: bool = False,
) -> FigureResult:
    """Read the points of a polar plot: the angle of each and its distance from the centre.

    Parameters
    ----------
    centre : (x, y), optional
        The centre of the plot, in pixels. Found from its circles if not given.
    r_ref : list of (radius in pixels, value), optional
        What two or more distances from the centre stand for: a ring and its label, say ``[(37.5, 1955),
        (98.2, 1965)]``. The rings found are listed in the result (``provenance["polar"]["rings"]``) to help.
    outer_value, inner_value : float, optional
        Instead of *r_ref*: the value at the outer circle (and at the centre, 0 by default).
    zero : {"east", "north", "west", "south"}
        Where an angle of 0 is: east (to the right) is matplotlib's default; north (up) is a clock face.
    clockwise : bool
        Angles grow clockwise (a clock face) rather than anticlockwise (matplotlib's default).
    colours : list of colours, optional
        The colours of the points. Every colour drawn inside the plot is tried otherwise.

    Returns
    -------
    FigureResult
        ``data`` has a row for each point: ``series``, ``colour``, ``point``, its pixel position, ``radius_px``,
        ``r_value`` (in the radial axis's units if they were given, else as a fraction of the outer radius),
        ``angle_degrees`` (0 to 360) and ``angle_radians``.
    """
    if zero not in ("east", "north", "west", "south"):
        raise ValueError("zero must be 'east', 'north', 'west' or 'south'")
    source = None
    if isinstance(image_path, np.ndarray):
        rgb = np.ascontiguousarray(image_path)
    else:
        source = Path(image_path)
        rgb = load_image(source)
    if straighten not in (False, None, "never") and straighten != "auto":
        rgb = _turn_upright(rgb, float(straighten))
    notes: list[str] = []
    found = find_polar_axes(rgb)
    if centre is None:
        if found is None:
            raise ValueError("no polar axes were found: give centre (and r_ref or outer_value)")
        cx, cy = found["centre"]
    else:
        cx, cy = (float(v) for v in centre)
    outer = found["radius"] if found else None
    # the radial scale
    if r_ref is not None:
        pts = [(float(a), float(b)) for a, b in r_ref]
        if len(pts) < 2:
            raise ValueError("r_ref needs at least two (radius, value) pairs")
        ra, rb = np.polyfit([q[0] for q in pts], [q[1] for q in pts], 1)
        radial = {"method": "reference_points", "points": pts}
    elif outer_value is not None and outer is not None:
        ra, rb = (float(outer_value) - inner_value) / outer, inner_value
        radial = {"method": "outer_circle", "outer_value": outer_value, "inner_value": inner_value}
    elif outer is not None:
        ra, rb = 1.0 / outer, 0.0
        radial = {"method": "fraction_of_outer_radius"}
        notes.append(
            "The radial axis is not calibrated: r_value is the distance from the centre as a fraction of the "
            "outer circle's radius. Give r_ref (rings and their values) or outer_value."
        )
    else:
        ra, rb = 1.0, 0.0
        radial = {"method": "pixels"}
        notes.append("The radial axis is not calibrated: r_value is in pixels. Give r_ref.")
    # (r_value = a * distance from the centre in pixels + b: for a program that places points of its own)
    radial.update(a=float(ra), b=float(rb))
    wanted = [parse_colour(c) for c in colours] if colours else None
    reach = outer if outer is not None else 0.45 * min(rgb.shape[:2])
    box = (
        int(max(0, cx - reach)),
        int(max(0, cy - reach)),
        int(min(rgb.shape[1] - 1, cx + reach)),
        int(min(rgb.shape[0] - 1, cy + reach)),
    )
    if wanted is None:
        # the saturated colours inside the circle: each a candidate for points
        roi = rgb[box[1] : box[3] + 1, box[0] : box[2] + 1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1] > 90
        if sat.sum() < 10:
            wanted = [(0, 0, 0)]
        else:
            px = roi[sat].reshape(-1, 3).astype(np.float32)
            k = min(4, max(1, len(px) // 50))
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _, lab, ctr = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(lab.ravel(), minlength=k)
            wanted = []
            for i in np.argsort(-counts):
                c = tuple(int(v) for v in ctr[i])
                # the paler edges of one colour's markers make a cluster of their own: keep one colour for both
                if counts[i] >= 20 and all(math.dist(c, q) > 70 for q in wanted):
                    wanted.append(c)
    radius = 0.5 * marker_size if marker_size else None
    markers = find_markers(
        rgb, box, wanted, tolerance=colour_tolerance, radius=radius, min_coverage=0.0 if colours else 0.3
    )
    rows = []
    for m in markers:
        dx, dy = m.x - cx, cy - m.y  # y up
        dist = math.hypot(dx, dy)
        if outer is not None and dist > outer * 1.02:
            continue
        theta = math.degrees(math.atan2(dy, dx))  # from east, anticlockwise
        offset = {"east": 0.0, "north": 90.0, "west": 180.0, "south": 270.0}[zero]
        ang = (theta - offset) % 360.0
        if clockwise:
            ang = (-ang) % 360.0
        rows.append(
            {
                "series": m.series,
                "colour": hex_colour(wanted[m.series - 1]),
                "x_pixel": round(m.x, 2),
                "y_pixel": round(m.y, 2),
                "radius_px": round(dist, 2),
                "r_value": ra * dist + rb,
                "angle_degrees": round(ang, 3),
                "angle_radians": math.radians(ang),
                "shared": m.shared,
            }
        )
    data = pd.DataFrame(rows)
    if not data.empty:
        data = data.sort_values(["series", "r_value"]).reset_index(drop=True)
        data.insert(2, "point", data.groupby("series").cumcount() + 1)
    else:
        notes.append("No points were found. Give colours (and marker_size) if the points are an unusual colour.")
    for n in notes:
        import warnings

        from ._figure import DigitiserWarning

        warnings.warn(n, DigitiserWarning, stacklevel=2)
    fig = PreparedFigure(
        rgb=rgb,
        dark=darkness_map(rgb),
        panels=[],
        notes=notes,
        note=notes.append,
        parameters={
            "kind": "polar",
            "centre": centre,
            "r_ref": r_ref,
            "outer_value": outer_value,
            "inner_value": inner_value,
            "zero": zero,
            "clockwise": clockwise,
            "colours": [list(c) for c in wanted],
            "straighten": straighten,
        },
        source_path=source,
        output_prefix=output_prefix or (source.stem if source else "figure"),
        exclude_boxes=None,
        turned_by=float(straighten) if straighten not in (False, None, "never", "auto") else 0.0,
    )

    def draw(canvas: np.ndarray) -> np.ndarray:
        c = (int(round(cx)), int(round(cy)))
        cv2.drawMarker(canvas, c, (0, 170, 0), cv2.MARKER_CROSS, 14, 1)
        if outer is not None:
            cv2.circle(canvas, c, int(round(outer)), (0, 170, 0), 1, lineType=cv2.LINE_AA)
        for ring in (found or {}).get("rings", []):
            cv2.circle(canvas, c, int(round(ring)), (120, 200, 120), 1, lineType=cv2.LINE_AA)
        for m in markers:
            cv2.circle(
                canvas,
                (int(round(m.x)), int(round(m.y))),
                int(round(m.radius)) + 2,
                (255, 0, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        return canvas

    extra = {
        "polar": {
            "centre": [cx, cy],
            "outer_radius": outer,
            "rings": (found or {}).get("rings", []),
            "radial_scale": radial,
            "zero": zero,
            "clockwise": clockwise,
        }
    }
    return finish(
        fig,
        "polar",
        data,
        draw,
        output_dir=output_dir,
        write_outputs=write_outputs,
        save_overlay=save_overlay,
        extra=extra,
    )
