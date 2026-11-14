"""Reading heat maps: the value of each cell, from its colour and the colour bar.

The colour bar is read like an axis: its colours are sampled along its length, and its tick labels (or the values
you give for its ends) say what each colour stands for. Each pixel of the heat map is then given the value of the
nearest colour on the bar (in CIELAB, where distances match what the eye sees), and the pixels are averaged over
the cells of the map.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd

from ._figure import _as_boxes, prepare_figure
from ._results import FigureResult, finish, interior_box, x_values, y_values
from .axis_labels import calibrate_axis


def _lab(rgb: np.ndarray) -> np.ndarray:
    return (
        cv2.cvtColor(np.ascontiguousarray(rgb.reshape(-1, 1, 3).astype(np.uint8)), cv2.COLOR_RGB2LAB)
        .reshape(-1, 3)
        .astype(np.float32)
    )


def _bar_candidates(rgb: np.ndarray) -> list[dict]:
    """Boxes between two parallel lines of the same length close together: the outline of a colour bar (its sides
    are found as the edges of a filled strip)."""
    from .frames import darkness_map, detect_axis_lines
    from .panel_detection import panels_from_boxes

    h_lines, v_lines = detect_axis_lines(rgb, darkness_map(rgb))
    out = []
    for lines, vertical in ((v_lines, True), (h_lines, False)):
        for i, a in enumerate(lines):
            for b in lines[i + 1 :]:
                gap = b.pos - a.pos
                if not 6 <= gap <= 0.25 * a.length or abs(a.start - b.start) > 3 or abs(a.end - b.end) > 3:
                    continue
                box = (a.pos, a.start, b.pos, a.end) if vertical else (a.start, a.pos, a.end, b.pos)
                try:
                    q = panels_from_boxes(rgb, [box])[0]
                except ValueError:
                    continue
                out.append(q)
    return out


def find_colourbar(rgb: np.ndarray, panels: list[dict]) -> Optional[dict]:
    """The colour bar: among the plot areas found, and boxes between pairs of parallel lines, the one that is long
    and narrow, its colour changing steadily along its length and hardly at all across it. Returns it as a panel,
    with ``"vertical"`` set."""
    best = None
    for p in list(panels) + _bar_candidates(rgb):
        e = p["edges"]
        w, h = e["right"] - e["left"], e["bottom"] - e["top"]
        if min(w, h) < 4 or max(w, h) < 40:
            continue
        vertical = h >= 4 * w
        if not (vertical or w >= 4 * h):
            continue
        x0, y0, x1, y1 = interior_box(p, inset=1)
        roi = rgb[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        if roi.size == 0:
            continue
        if vertical:
            mid = roi[:, int(0.2 * roi.shape[1]) : max(int(0.2 * roi.shape[1]) + 1, int(np.ceil(0.8 * roi.shape[1])))]
        else:
            mid = roi[int(0.2 * roi.shape[0]) : max(int(0.2 * roi.shape[0]) + 1, int(np.ceil(0.8 * roi.shape[0])))]
        along = mid.mean(axis=1) if vertical else mid.mean(axis=0)
        across = mid.std(axis=1 if vertical else 0).mean()
        change = np.abs(np.diff(along, axis=0)).sum(axis=1)
        span = float(np.abs(along[-1] - along[0]).sum())
        if span < 60 or across > 25 or change.max() > 0.35 * span:
            continue
        score = span / (1.0 + across)
        if best is None or score > best[0]:
            best = (score, dict(p, vertical=vertical))
    return best[1] if best else None


def colourbar_scale(
    rgb: np.ndarray,
    dark: np.ndarray,
    bar: dict,
    values: Optional[Sequence] = None,
    ocr: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The colour bar's colours (CIELAB, one per pixel along it) and their values, with how the values were found.

    *values*: ``(value at the bottom (or left) end, value at the top (or right) end)``, or a list of
    ``(pixel, value)`` along the bar; otherwise the bar's tick labels are read.
    """
    x0, y0, x1, y1 = interior_box(bar, inset=1)
    vertical = bar.get("vertical", True)
    roi = rgb[y0 : y1 + 1, x0 : x1 + 1]
    if vertical:
        c0, c1 = int(0.2 * roi.shape[1]), max(int(0.2 * roi.shape[1]) + 1, int(np.ceil(0.8 * roi.shape[1])))
        colours = np.median(roi[:, c0:c1], axis=1)
        pixels = np.arange(y0, y1 + 1, dtype=float)
    else:
        r0, r1 = int(0.2 * roi.shape[0]), max(int(0.2 * roi.shape[0]) + 1, int(np.ceil(0.8 * roi.shape[0])))
        colours = np.median(roi[r0:r1], axis=0)
        pixels = np.arange(x0, x1 + 1, dtype=float)
    how: dict = {}
    if values is not None and len(values) == 2 and not hasattr(values[0], "__len__"):
        e = bar["edges"]
        lo_px, hi_px = (e["bottom"], e["top"]) if vertical else (e["left"], e["right"])
        pts = [(lo_px, float(values[0])), (hi_px, float(values[1]))]
        how = {"method": "ends", "points": pts}
    elif values is not None:
        pts = [(float(a), float(b)) for a, b in values]
        how = {"method": "reference_points", "points": pts}
    else:
        pts = []
        if ocr:
            fit, labels = calibrate_axis(rgb, dark, bar, "y2" if vertical else "x")
            if fit is None and vertical:
                fit, labels = calibrate_axis(rgb, dark, bar, "y")  # labels on the left
            if fit is not None and fit.usable:
                pts = list(fit.inliers)
                how = {"method": "tick_labels", "points": pts, "confidence": fit.confidence, "scale": fit.scale}
        if len(pts) < 2:
            raise ValueError("the colour bar's values could not be read from its labels: give colourbar_values")
    px = np.array([q[0] for q in pts])
    vv = np.array([q[1] for q in pts])
    a, b = np.polyfit(px, vv, 1)
    return _lab(colours), a * pixels + b, how


def _nearest_values(lab: np.ndarray, bar_lab: np.ndarray, bar_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The value of the nearest colour on the bar for each colour in *lab* (interpolating between the bar's
    neighbouring colours), and how far (CIELAB) each colour is from the bar."""
    # distances to every colour of the bar, in chunks
    out_v = np.empty(len(lab), dtype=np.float32)
    out_d = np.empty(len(lab), dtype=np.float32)
    seg_a, seg_b = bar_lab[:-1], bar_lab[1:]
    seg = seg_b - seg_a
    seg_len2 = np.maximum((seg**2).sum(axis=1), 1e-6)
    va, vb = bar_values[:-1], bar_values[1:]
    for i in range(0, len(lab), 20000):
        q = lab[i : i + 20000]
        rel = q[:, None, :] - seg_a[None, :, :]
        t = np.clip((rel * seg[None, :, :]).sum(axis=2) / seg_len2[None, :], 0.0, 1.0)
        nearest = seg_a[None, :, :] + t[:, :, None] * seg[None, :, :]
        d = np.sqrt(((q[:, None, :] - nearest) ** 2).sum(axis=2))
        j = np.argmin(d, axis=1)
        k = np.arange(len(q))
        out_d[i : i + 20000] = d[k, j]
        out_v[i : i + 20000] = va[j] + t[k, j] * (vb[j] - va[j])
    return out_v, out_d


def _period(profile: np.ndarray) -> Optional[float]:
    """The size of the cells along one direction, from how often the colour changes sharply (None if the cells are
    no bigger than a pixel or two)."""
    if len(profile) < 12:
        return None
    jumps = np.abs(np.diff(profile, axis=0)).sum(axis=1) if profile.ndim == 2 else np.abs(np.diff(profile))
    thr = np.percentile(jumps, 75)
    edges = np.flatnonzero(jumps > max(thr, 0.25 * jumps.max()))
    if len(edges) < 3:
        return None
    gaps = np.diff(edges)
    gaps = gaps[gaps > 1]
    if gaps.size < 2:
        return None
    period = float(np.median(gaps))
    return period if period >= 2.5 else None


def digitise_heatmap(
    image_path: Union[str, Path, np.ndarray],
    *,
    colourbar=None,
    colourbar_values: Optional[Sequence] = None,
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    max_colour_error: float = 12.0,
    output_dir=None,
    output_prefix: Optional[str] = None,
    write_outputs: bool = True,
    save_overlay: bool = False,
    **figure,
) -> FigureResult:
    """Read a heat map (or several sharing a colour bar): the value of each cell.

    Parameters
    ----------
    colourbar : (x_left, y_top, x_right, y_bottom), optional
        Where the colour bar is. Found among the plot areas if not given (a long, narrow box whose colour changes
        steadily along it).
    colourbar_values : (bottom, top), or list of (pixel, value), optional
        The values at the ends of the colour bar (bottom and top for an upright bar, left and right for one lying
        down), or at points along it. Read from the bar's tick labels if not given.
    colormap, vmin, vmax : optional
        Instead of a colour bar: the name of the matplotlib colour map the heat map was drawn with ("viridis") and
        the values at its ends.
    rows, cols : int, optional
        How many cells the map has down and across. Without them, the cells are found from where the colour changes
        sharply, or each pixel is a cell.
    max_colour_error : float
        How far (in CIELAB units) a pixel's colour may be from every colour of the bar and still be read; further
        away is text, a gridline or a mark, and is left out.
    **figure
        The arguments of :func:`trace_digitiser.digitise` that concern the figure and its axes.

    Returns
    -------
    FigureResult
        ``data`` has a row for each cell of each map: ``panel_id``, ``row`` (from the top), ``col`` (from the
        left), the cell's centre in pixels and, where the axes are calibrated, in their units (``x_value``,
        ``y_value``), its ``value``, the ``colour_error`` of its pixels (median) and how many pixels were read.
    """
    bar_box = _as_boxes(colourbar, "colourbar")[0] if colourbar is not None else None
    fig = prepare_figure(
        image_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        write_outputs=write_outputs,
        extra_parameters={
            "kind": "heatmap",
            "colourbar": list(bar_box) if bar_box else None,
            "colourbar_values": [list(v) if hasattr(v, "__len__") else v for v in colourbar_values]
            if colourbar_values is not None
            else None,
            "colormap": colormap,
            "vmin": vmin,
            "vmax": vmax,
            "rows": rows,
            "cols": cols,
            "max_colour_error": max_colour_error,
        },
        **figure,
    )
    panels = fig.panels
    how: dict = {}
    if colormap is not None:
        if vmin is None or vmax is None:
            raise ValueError("give vmin and vmax with colormap")
        import matplotlib

        cmap = matplotlib.colormaps[colormap]
        samples = np.linspace(0.0, 1.0, 256)
        bar_lab = _lab((np.array([cmap(t)[:3] for t in samples]) * 255).round())
        bar_values = vmin + samples * (vmax - vmin)
        how = {"method": "colormap", "colormap": colormap, "vmin": vmin, "vmax": vmax}
        bar = None
    else:
        if bar_box is not None:
            from .panel_detection import panels_from_boxes

            bar = panels_from_boxes(fig.rgb, [bar_box])[0]
            e = bar["edges"]
            bar["vertical"] = (e["bottom"] - e["top"]) >= (e["right"] - e["left"])
            panels = [p for p in panels if not _same(p, bar)]
        else:
            bar = find_colourbar(fig.rgb, panels)
            if bar is None:
                raise ValueError("no colour bar was found: give colourbar (its box), or colormap with vmin and vmax")
            panels = [p for p in panels if p is not bar and not _same(p, bar)]
        bar_lab, bar_values, how = colourbar_scale(fig.rgb, fig.dark, bar, colourbar_values, ocr=fig.parameters["ocr"])
        how["box"] = [bar["edges"][k] for k in ("left", "top", "right", "bottom")]
        if how.get("method") == "tick_labels" and how.get("confidence") != "high":
            read = ", ".join(f"{v:g}" for _, v in how["points"])
            fig.note(
                f"The colour bar's scale was read from its tick labels ({read}) with {how.get('confidence')} "
                "confidence: check them, or give colourbar_values."
            )
    out = []
    drawn = []
    for p in panels:
        x0, y0, x1, y1 = interior_box(p, inset=1)
        roi = fig.rgb[y0 : y1 + 1, x0 : x1 + 1]
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            continue
        # each colour once: a heat map has far fewer colours than pixels
        flat = roi.reshape(-1, 3).astype(np.int32)
        uniq, inverse = np.unique((flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2], return_inverse=True)
        colours = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
        values, err = _nearest_values(_lab(colours), bar_lab, bar_values)
        values = values[inverse.ravel()].reshape(h, w)
        err = err[inverse.ravel()].reshape(h, w)
        good = err <= max_colour_error
        if good.mean() < 0.5:
            fig.note(
                f"Panel {p['panel_id']}: most of its colours are not on the colour bar, so it is probably not a "
                "heat map; it was left out."
            )
            continue
        n_rows = rows or (lambda per: int(round(h / per)) if per else h)(_period(roi.mean(axis=1).astype(np.float32)))
        n_cols = cols or (lambda per: int(round(w / per)) if per else w)(_period(roi.mean(axis=0).astype(np.float32)))
        e = p["edges"]
        ys = e["top"] + (np.arange(n_rows) + 0.5) * (e["bottom"] - e["top"]) / n_rows
        xs = e["left"] + (np.arange(n_cols) + 0.5) * (e["right"] - e["left"]) / n_cols
        ry = (e["bottom"] - e["top"]) / n_rows
        rx = (e["right"] - e["left"]) / n_cols
        for i, yc in enumerate(ys):
            a0 = int(np.clip(np.floor(yc - 0.3 * ry) - y0, 0, h - 1))
            a1 = int(np.clip(np.ceil(yc + 0.3 * ry) - y0, a0, h - 1))
            for j, xc in enumerate(xs):
                b0 = int(np.clip(np.floor(xc - 0.3 * rx) - x0, 0, w - 1))
                b1 = int(np.clip(np.ceil(xc + 0.3 * rx) - x0, b0, w - 1))
                cell = values[a0 : a1 + 1, b0 : b1 + 1]
                ok = good[a0 : a1 + 1, b0 : b1 + 1]
                n = int(ok.sum())
                v = float(np.median(cell[ok])) if n else float("nan")
                out.append(
                    {
                        "panel_id": p["panel_id"],
                        "row": i + 1,
                        "col": j + 1,
                        "x_pixel": round(float(xc), 2),
                        "y_pixel": round(float(yc), 2),
                        "x_value": float(x_values(p, xc)),
                        "y_value": float(y_values(p, yc)) if p.get("y_calibration") else float("nan"),
                        "value": v,
                        "colour_error": float(np.median(err[a0 : a1 + 1, b0 : b1 + 1])),
                        "pixels": n,
                    }
                )
        drawn.append((p, n_rows, n_cols))
        fig.parameters.setdefault("found", {})[p["panel_id"]] = {"rows": n_rows, "cols": n_cols}
    data = pd.DataFrame(out)
    if data.empty:
        fig.note("No heat map was read.")

    def draw(canvas: np.ndarray) -> np.ndarray:
        if bar is not None:
            e = bar["edges"]
            cv2.rectangle(canvas, (int(e["left"]), int(e["top"])), (int(e["right"]), int(e["bottom"])), (0, 170, 0), 2)
        for p, nr, nc in drawn:
            e = p["edges"]
            if nr <= 60:
                for i in range(1, nr):
                    y = int(round(e["top"] + i * (e["bottom"] - e["top"]) / nr))
                    cv2.line(canvas, (int(e["left"]), y), (int(e["right"]), y), (255, 0, 255), 1)
            if nc <= 60:
                for j in range(1, nc):
                    x = int(round(e["left"] + j * (e["right"] - e["left"]) / nc))
                    cv2.line(canvas, (x, int(e["top"])), (x, int(e["bottom"])), (255, 0, 255), 1)
        return canvas

    return finish(
        fig,
        "heatmap",
        data,
        draw,
        output_dir=output_dir,
        write_outputs=write_outputs,
        save_overlay=save_overlay,
        extra={"colour_scale": how},
    )


def _same(p: dict, q: dict) -> bool:
    a, b = p["edges"], q["edges"]
    return all(abs(a[k] - b[k]) <= 3 for k in ("left", "right", "top", "bottom"))
