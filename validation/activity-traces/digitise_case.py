"""Digitise the traces of one figure as its configuration says, and save them as one CSV.

    python digitise_case.py CONFIG.json OUT.csv [OVERLAY_DIR]

The configuration gives the image and, for each panel, what a user of the workbench would give: the plot area drawn
round it, the colour of each series clicked, and, where the axis labels cannot be read, two ticks on an axis.
Output columns: panel, series, x (in the axis's units), y, x_pixel, interpolated (True where the line reader saw
no pixel of the trace in that column and filled the value in from its neighbours). A panel's "y_scale" multiplies y,
for an axis labelled with a multiplier such as (x10^2).

A panel's series are read together, in one run of the line reader, unless the panel's "traces" gives a series its
own reading: a list of runs ("segments"), each made with the workbench's options for lines and nothing else, and
each giving the columns of one stretch of the trace:

    "traces": {"KO": [{"x_px": [339, 1200], "exclude": [[...]], "also": ["#231f20"]},
                      {"x_px": [1200, 2704], "colour": "#00b000", "peaks": "tips", "estimate": "upper"}]}

x_px (the stretch, in pixel columns; default the whole width), window_px (the columns the reader is shown, the rest
of the plot area being left out with boxes as a user of the workbench would draw them; default x_px), colour (the
colour clicked; default the series'), also (other colours clicked with it, so that the reader tells them apart; the
series is the trace nearest its own colour), exclude (boxes left out, besides the panel's), achromatic (the reader's
option), peaks ("tips": the workbench's option to read narrow peaks at their tips, dwb.lines.peaks_at_tips, from the
workbench given as the fourth argument or in $DIGITISER_WORKBENCH) and estimate (the column of the reader's table
taken: "center", the default, "mean", "upper" or "lower", which a user picks from the table). Where two stretches
overlap, the later one is taken.

Each panel's calibration (pixels to values) is saved beside the CSV, in NAME.calibration.json, for figure_check.py.
"""
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import trace_digitiser as td
from trace_digitiser.trace_detection import parse_colour

warnings.simplefilter("ignore")


def nearest(colour, wanted):
    if isinstance(colour, str) and colour.strip().startswith("["):
        colour = json.loads(colour)
    c = np.array(parse_colour(colour), float)
    d = [np.abs(c - np.array(parse_colour(w), float)).sum() for w in wanted]
    return int(np.argmin(d))


ESTIMATES = {"center": "center_estimate", "mean": "mean_estimate", "upper": "upper_envelope",
             "lower": "lower_envelope"}


def boxes(v):
    """One box or a list of boxes, as a list of tuples."""
    if not v:
        return []
    return [tuple(float(a) for a in b) for b in (v if isinstance(v[0], (list, tuple)) else [v])]


def workbench_lines(workbench=None):
    """The workbench's dwb.lines module (for peaks at their tips), loaded on its own from its file."""
    where = workbench or os.environ.get("DIGITISER_WORKBENCH")
    if not where:
        raise SystemExit("peaks: tips needs the workbench: give its folder as the third argument or in "
                         "$DIGITISER_WORKBENCH")
    spec = importlib.util.spec_from_file_location("dwb_lines", Path(where) / "python" / "dwb" / "lines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel_options(panel, extra_exclude=(), out_dir=None, prefix=None):
    opts = {"plot_area": [tuple(panel["plot_area"])]}
    for k in ("y_ref", "x_ref", "y_min", "y_max", "x_min", "x_max"):
        if k in panel:
            opts[k] = panel[k]
    excl = boxes(panel.get("exclude")) + list(extra_exclude)
    if excl:
        opts["exclude"] = excl
    if out_dir:
        opts.update(write_outputs=True, output_dir=str(out_dir), save_overlay=True,
                    output_prefix=(prefix or panel["name"]).replace(" ", "_").replace("/", "-"))
    else:
        opts["write_outputs"] = False
    return opts


def calibration(r):
    """The first panel's y calibration of a result, as {a, b, log}: value = a * row + b (log10 of it if log)."""
    for p in r.panels:
        c = p.calibration
        if c is not None and getattr(c, "is_calibrated", True):
            d = c.to_dict() if hasattr(c, "to_dict") else dict(c)
            return {"a": float(d["a"]), "b": float(d["b"]), "log": "log" in str(d.get("scale_type", ""))}
    return None


def read_segment(image, panel, name, seg, workbench=None):
    """One stretch of one series, read on its own: (rows, warnings, calibration)."""
    colour = seg.get("colour", panel["series"][name])
    also = list(seg.get("also", []))
    left, top, right, bottom = panel["plot_area"]
    x0, x1 = seg.get("x_px", [left, right])
    w0, w1 = seg.get("window_px", [x0, x1])
    extra = boxes(seg.get("exclude"))
    if w0 > left:
        extra.append((left - 3, top - 3, w0, bottom + 3))
    if w1 < right:
        extra.append((w1, top - 3, right + 3, bottom + 3))
    opts = panel_options(panel, extra)
    if seg.get("achromatic"):
        opts["achromatic"] = seg["achromatic"]
    res = td.digitise(image, trace_colours=[colour] + also, **opts)
    if seg.get("peaks") == "tips":
        res.trace_data = workbench_lines(workbench).peaks_at_tips(res)
    d = res.trace_data
    rows = []
    if len(d) and "x_value" in d:
        col = ESTIMATES[seg.get("estimate", "center")]
        best = None
        for tid, g in d.groupby("trace_id"):
            k = nearest(g["trace_median_rgb"].iloc[0], [colour] + also) if "trace_median_rgb" in g else int(tid) - 1
            if k == 0:
                best = g
                break
        if best is not None:
            g = best[(best["x_pixel"] >= x0) & (best["x_pixel"] < x1)]
            interp = g["interpolated"] if "interpolated" in g else [False] * len(g)
            for x, y, xp, ip in zip(g["x_value"], g[col], g["x_pixel"], interp):
                rows.append((panel["name"], name, x, y, xp, bool(ip)))
    return rows, list(res.warnings), calibration(res)


def run_panel(image, reader, panel, out_dir=None, workbench=None):
    names = list(panel["series"])
    colours = [panel["series"][n] for n in names]
    own = panel.get("traces", {})
    opts = panel_options(panel, out_dir=out_dir)
    rows = []
    warns = []
    cal = None
    if reader == "lines":
        together = [n for n in names if n not in own]
        if together:
            r = td.digitise(image, trace_colours=colours, **opts)
            d = r.trace_data
            warns = list(r.warnings)
            cal = calibration(r)
            if len(d) and "x_value" in d:
                for tid, g in d.groupby("trace_id"):
                    col = g["trace_median_rgb"].iloc[0] if "trace_median_rgb" in g else None
                    k = int(tid) - 1 if col is None else nearest(col, colours)
                    if names[k] not in together:
                        continue
                    interp = g["interpolated"] if "interpolated" in g else [False] * len(g)
                    for x, y, xp, ip in zip(g["x_value"], g["center_estimate"], g["x_pixel"], interp):
                        rows.append((panel["name"], names[k], x, y, xp, bool(ip)))
        for name in names:
            if name not in own:
                continue
            got = {}
            for seg in own[name]:
                seg_rows, w, c = read_segment(image, panel, name, seg, workbench)
                warns += [f"{name}: {x}" for x in w]
                cal = cal or c
                for row in seg_rows:
                    got[row[4]] = row  # by pixel column: a later stretch replaces an earlier one
            rows += [got[k] for k in sorted(got)]
    else:
        r = td.digitise_points(image, colours=colours, **opts, **panel.get("points_options", {}))
        d = r.data
        warns = list(r.warnings)
        cal = calibration(r)
        if len(d) and "x_value" in d:
            for _, row in d.iterrows():
                k = nearest(row["colour"], colours)
                rows.append((panel["name"], names[k], row["x_value"], row["y_value"], row["x_pixel"], False))
    scale = float(panel.get("y_scale", 1))
    if scale != 1:
        rows = [(a, b, x, y * scale, xp, ip) for a, b, x, y, xp, ip in rows]
    return rows, warns, cal


def main(config, out, overlay_dir=None, workbench=None):
    cfg = json.loads(Path(config).read_text())
    image = str(Path(config).parent / cfg["image"])
    rows, notes, cals = [], {}, {}
    for panel in cfg["panels"]:
        od = Path(overlay_dir) / cfg["id"] if overlay_dir else None
        if od:
            od.mkdir(parents=True, exist_ok=True)
        r, w, cal = run_panel(image, panel.get("reader", cfg["reader"]), panel, od, workbench)
        rows += r
        notes[panel["name"]] = w
        cals[panel["name"]] = {**(cal or {}), "y_scale": float(panel.get("y_scale", 1))}
        print(f"{panel['name']}: {len(r)} values; " + " | ".join(x[:110] for x in w[:3]), file=sys.stderr)
    pd.DataFrame(rows, columns=["panel", "series", "x", "y", "x_pixel", "interpolated"]).to_csv(
        out, index=False, float_format="%.6g")
    Path(out).with_suffix(".warnings.json").write_text(json.dumps(notes, indent=1))
    Path(out).with_suffix(".calibration.json").write_text(json.dumps(cals, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
