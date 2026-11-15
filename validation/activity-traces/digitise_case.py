"""Digitise the traces of one figure as its configuration says, and save them as one CSV.

    python digitise_case.py CONFIG.json OUT.csv [OVERLAY_DIR]

The configuration gives the image and, for each panel, what a user of the workbench would give: the plot area drawn
round it, the colour of each series clicked, and, where the axis labels cannot be read, two ticks on an axis.
Output columns: panel, series, x (in the axis's units), y, x_pixel, interpolated (True where the line reader saw
no pixel of the trace in that column and filled the value in from its neighbours). A panel's "y_scale" multiplies y,
for an axis labelled with a multiplier such as (x10^2).
"""
import json
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


def run_panel(image, reader, panel, out_dir=None):
    names = list(panel["series"])
    colours = [panel["series"][n] for n in names]
    opts = {"plot_area": [tuple(panel["plot_area"])]}
    for k in ("y_ref", "x_ref", "exclude", "y_min", "y_max", "x_min", "x_max"):
        if k in panel:
            opts[k] = panel[k]
    if out_dir:
        opts.update(write_outputs=True, output_dir=str(out_dir), save_overlay=True,
                    output_prefix=panel["name"].replace(" ", "_").replace("/", "-"))
    else:
        opts["write_outputs"] = False
    rows = []
    warns = []
    if reader == "lines":
        r = td.digitise(image, trace_colours=colours, **opts)
        d = r.trace_data
        warns = list(r.warnings)
        if len(d) and "x_value" in d:
            for tid, g in d.groupby("trace_id"):
                col = g["trace_median_rgb"].iloc[0] if "trace_median_rgb" in g else None
                k = int(tid) - 1 if col is None else nearest(col, colours)
                interp = g["interpolated"] if "interpolated" in g else [False] * len(g)
                for x, y, xp, ip in zip(g["x_value"], g["center_estimate"], g["x_pixel"], interp):
                    rows.append((panel["name"], names[k], x, y, xp, bool(ip)))
    else:
        r = td.digitise_points(image, colours=colours, **opts, **panel.get("points_options", {}))
        d = r.data
        warns = list(r.warnings)
        if len(d) and "x_value" in d:
            for _, row in d.iterrows():
                k = nearest(row["colour"], colours)
                rows.append((panel["name"], names[k], row["x_value"], row["y_value"], row["x_pixel"], False))
    scale = float(panel.get("y_scale", 1))
    if scale != 1:
        rows = [(a, b, x, y * scale, xp, ip) for a, b, x, y, xp, ip in rows]
    return rows, warns


def main(config, out, overlay_dir=None):
    cfg = json.loads(Path(config).read_text())
    image = str(Path(config).parent / cfg["image"])
    rows, notes = [], {}
    for panel in cfg["panels"]:
        od = Path(overlay_dir) / cfg["id"] if overlay_dir else None
        if od:
            od.mkdir(parents=True, exist_ok=True)
        r, w = run_panel(image, panel.get("reader", cfg["reader"]), panel, od)
        rows += r
        notes[panel["name"]] = w
        print(f"{panel['name']}: {len(r)} values; " + " | ".join(x[:110] for x in w[:3]), file=sys.stderr)
    pd.DataFrame(rows, columns=["panel", "series", "x", "y", "x_pixel", "interpolated"]).to_csv(
        out, index=False, float_format="%.6g")
    Path(out).with_suffix(".warnings.json").write_text(json.dumps(notes, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
