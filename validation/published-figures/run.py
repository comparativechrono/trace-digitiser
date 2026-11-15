"""Read published figures whose numbers are known, and compare what trace-digitiser reads with those numbers.

    python validation/published-figures/run.py                        # every case (about five minutes)
    python validation/published-figures/run.py foley andreani-1E      # the cases whose names start so
    python validation/published-figures/run.py --json new.json --compare validation/published-figures/results-2.2.0.json

The figures are in figures/ and the numbers behind them in known.json, which make_known.py builds from the papers' own
data (sources/). README.md says where each figure comes from, what each case tests, and how each release did.

Errors are in percent of the range the axis is labelled over (0-24 h: 24 h), so that panels in different units can
be compared. A point counts as found when it lies within 3 % (of the axes) of a known one, a bar when its height is
within 1 %, a curve when the trace that follows it covers at least half of it with a median error of 1 % or less,
and a dot of a dot plot when it is within 3 % (of the y axis) of one of its column's values. Points and dots are
paired with the known ones by the Hungarian method; bars left to right.
"""

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import trace_digitiser as td

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
POINT_TOL = 0.03  # fraction of each axis
BAR_TOL = 1.0  # percent of the axis
CURVE_TOL = 1.0  # percent of the axis (median), over at least half of the curve
DOT_TOL = 0.03  # fraction of the y axis

L1, L2, L5 = "leypunskiy-2017-fig1.jpg", "leypunskiy-2017-fig2.jpg", "leypunskiy-2017-fig5.jpg"
F2, P3 = "foley-2019-fig2.jpg", "pearl-reed-1920-fig3.png"
A1, A3 = "andreani-2022-fig1.jpg", "andreani-2022-fig3.jpg"

# Each panel's plot area (left, top, right, bottom, in pixels of its figure): the panel compared is the one whose plot
# area, as trace-digitiser found or was given it, overlaps this box most.
BOX = {
    "leypunskiy-1C": (1894.7, 905.7, 2559.0, 1631.3),
    "leypunskiy-2B": (1529, 65, 2545, 1696),  # the three stacked plots together
    "leypunskiy-2B-18:6": (1529, 65, 2545, 608.5),
    "leypunskiy-2B-12:12": (1529, 608.5, 2545, 1153),
    "leypunskiy-2B-6:18": (1529, 1153, 2545, 1696),
    "leypunskiy-2C": (294.5, 777.7, 1141.5, 1697.1),
    "leypunskiy-5A": (224.9, 730.0, 1192.6, 1799.3),
    "leypunskiy-5B": (1539.0, 734.1, 2539.6, 1797.4),
    "leypunskiy-5C": (1537.8, 2072.0, 2529.9, 3135.4),
    "foley-2C-25C": (159, 2543, 1218, 3137),
    "foley-2C-30C": (1443, 2543, 2335, 3138),
    "foley-2D": (2773, 2537, 3342, 3137),
    "foley-2E": (3812, 2537, 4388, 3137),
    "foley-2F-period": (257, 3910, 704, 4522),
    "foley-2F-rhythmic": (930, 3909, 1423, 4515),
    "foley-2G-period": (1760, 3908, 2257, 4518),
    "foley-2G-rhythmic": (2449, 3909, 2892, 4517),
    "foley-2H-period": (3236, 3910, 3732, 4520),
    "foley-2H-rhythmic": (3925, 3910, 4420, 4520),
    "andreani-1B": (1143, 160, 1853, 521),
    "andreani-1C": (95, 828, 522, 1101),
    "andreani-1E": (946, 826, 1896, 1092),
    "andreani-3A": (592, 160, 1130, 735),
    "andreani-3B": (1386, 151, 1925, 728),
    "andreani-3C": (598, 1196, 1117, 1753),
    "andreani-3D": (1392, 1184, 1911, 1756),
}

# The range each panel's axes are labelled over (x, y), in the axes' units.
SPAN = {
    "leypunskiy-1C": (24, 24), "leypunskiy-2B": (60, 75), "leypunskiy-2C": (24, 24), "leypunskiy-5A": (24, 24),
    "leypunskiy-5B": (12, 8), "leypunskiy-5C": (12, 12),
    "foley-2C-25C": (None, 2), "foley-2C-30C": (None, 2), "foley-2D": (None, 2), "foley-2E": (None, 2),
    "foley-2F-period": (None, 4), "foley-2G-period": (None, 4), "foley-2H-period": (None, 4),
    "foley-2F-rhythmic": (None, 100), "foley-2G-rhythmic": (None, 100), "foley-2H-rhythmic": (None, 100),
    "pearl-reed-3-dots": (120, 1e8), "pearl-reed-3-curve": (120, 1e8),
    "andreani-1B": (11.5, 40), "andreani-1C": (None, 50), "andreani-1E": (None, 200),
    "andreani-3A": (None, 400), "andreani-3B": (None, 400), "andreani-3C": (None, 400), "andreani-3D": (None, 400),
}

L2B_AREAS = [BOX["leypunskiy-2B-18:6"], BOX["leypunskiy-2B-12:12"], BOX["leypunskiy-2B-6:18"]]
L2B_PARTS = [(BOX[f"leypunskiy-2B-{s}"], [s]) for s in ("18:6", "12:12", "6:18")]
L2C_COLOURS = ["#e81828", "#0898d8", "#08a8e8", "black"]
L5A_TICKS = {"plot_area": [BOX["leypunskiy-5A"]], "y_ref": [(882.5, 12), (1646.5, -8)],
             "x_ref": [(293.5, 36), (1123.5, 60)]}
L5C_COLOURS = ["#e81828", "#d8d828", "#08a888", "#0868b8", "#c81888"]
P3_BOX = (0, 0, 1e9, 1e9)  # the only plot in the figure
P3_TICKS = {"y_ref": [(845.18, 0), (54.73, 100_000_000)], "x_ref": [(263.04, 1790), (1095.94, 1910)]}
A1B_COLOURS = ["#236333", "#f26930", "#542484"]
A1B_TICKS = {"plot_area": [BOX["andreani-1B"]], "x_ref": [(1165, 0.0), (1829, 11.5)]}  # its labels are slanted
A3_COLOURS = ["#636dac", "#ec4739"]


def _case(name, figure, reader, given, compare, known, options=None, parts=None, held_out=False, box=None):
    """parts: [(box, the known series in it, or None for all of them), ...], for a case that looks at several plots."""
    return {"name": name, "figure": figure, "reader": reader, "given": given, "compare": compare, "known": known,
            "options": options or {}, "parts": parts or [(box or BOX[known], None)], "held_out": held_out}


CASES = [
    # ---- Leypunskiy et al. (2017): markers with error bars; a line with markers over shaded bands
    _case("leypunskiy-1C-nothing", L1, "points", "nothing", "points", "leypunskiy-1C"),
    _case("leypunskiy-1C-colour", L1, "points", "the colour", "points", "leypunskiy-1C",
          {"colours": ["#e91f28"]}),
    _case("leypunskiy-2B-nothing", L2, "lines", "nothing", "curves", "leypunskiy-2B",
          parts=[(BOX["leypunskiy-2B"], None)]),
    _case("leypunskiy-2B-areas", L2, "lines", "the plot areas", "curves", "leypunskiy-2B",
          {"plot_area": L2B_AREAS}, L2B_PARTS),
    _case("leypunskiy-2B-areas-colour", L2, "lines", "the plot areas, the line's colour", "curves", "leypunskiy-2B",
          {"plot_area": L2B_AREAS, "trace_colours": ["#232022"]}, L2B_PARTS),
    _case("leypunskiy-2B-points", L2, "points", "the plot areas, the markers' colour", "points", "leypunskiy-2B",
          {"plot_area": L2B_AREAS, "colours": ["black"]}, L2B_PARTS),
    _case("leypunskiy-2C-nothing", L2, "points", "nothing", "points", "leypunskiy-2C"),
    _case("leypunskiy-2C-colours", L2, "points", "the colours", "points", "leypunskiy-2C", {"colours": L2C_COLOURS}),
    _case("leypunskiy-2C-colours-size", L2, "points", "the colours, the marker size", "points", "leypunskiy-2C",
          {"colours": L2C_COLOURS, "marker_size": 15}),
    _case("leypunskiy-5A-nothing", L5, "points", "nothing", "points", "leypunskiy-5A"),
    _case("leypunskiy-5A-ticks", L5, "points", "the plot area, two ticks on each axis", "points", "leypunskiy-5A",
          L5A_TICKS),
    _case("leypunskiy-5A-ticks-colour-size", L5, "points", "the plot area, ticks, the colour, the marker size",
          "points", "leypunskiy-5A", {**L5A_TICKS, "colours": ["black"], "marker_size": 15}),
    _case("leypunskiy-5B-nothing", L5, "points", "nothing", "points", "leypunskiy-5B"),
    _case("leypunskiy-5B-colour-size", L5, "points", "the colour, the marker size", "points", "leypunskiy-5B",
          {"colours": ["#383898"], "marker_size": 15}),
    _case("leypunskiy-5C-nothing", L5, "points", "nothing", "points", "leypunskiy-5C"),
    _case("leypunskiy-5C-colours", L5, "points", "the colours", "points", "leypunskiy-5C", {"colours": L5C_COLOURS}),
    _case("leypunskiy-5C-colours-size", L5, "points", "the colours, the marker size", "points", "leypunskiy-5C",
          {"colours": L5C_COLOURS, "marker_size": 15}),
    # ---- Foley et al. (2019): filled, hatched and outlined bars, with error bars
    *[_case(f"{p}-nothing", F2, "bars", "nothing", "bars", p) for p in BOX if p.startswith("foley-")],
    _case("foley-2C-25C-area", F2, "bars", "the plot area", "bars", "foley-2C-25C",
          {"plot_area": [BOX["foley-2C-25C"]]}),
    # ---- Pearl and Reed (1920), a scanned page: census dots and the fitted curve, on a grid
    _case("pearl-reed-3-curve-nothing", P3, "lines", "nothing", "curves", "pearl-reed-3-curve", box=P3_BOX),
    _case("pearl-reed-3-dots-nothing", P3, "points", "nothing", "points", "pearl-reed-3-dots", box=P3_BOX),
    _case("pearl-reed-3-curve-ticks", P3, "lines", "two ticks on each axis", "curves", "pearl-reed-3-curve",
          P3_TICKS, box=P3_BOX),
    _case("pearl-reed-3-curve-ticks-exclude", P3, "lines", "ticks, the equation excluded", "curves",
          "pearl-reed-3-curve", {**P3_TICKS, "exclude": [(280, 230, 620, 320)]}, box=P3_BOX),
    _case("pearl-reed-3-dots-ticks", P3, "points", "two ticks on each axis", "points", "pearl-reed-3-dots", P3_TICKS,
          box=P3_BOX),
    _case("pearl-reed-3-dots-ticks-colour", P3, "points", "ticks, the colour", "points", "pearl-reed-3-dots",
          {**P3_TICKS, "colours": ["#404040"]}, box=P3_BOX),
    # ---- held out: Andreani et al. (2022), looked at only after the changes they test were made
    _case("andreani-1B-area", A1, "points", "the plot area", "points", "andreani-1B",
          {"plot_area": [BOX["andreani-1B"]]}, held_out=True),
    _case("andreani-1B-area-colours", A1, "points", "the plot area, the colours", "points", "andreani-1B",
          {"plot_area": [BOX["andreani-1B"]], "colours": A1B_COLOURS}, held_out=True),
    _case("andreani-1B-area-ticks", A1, "points", "the plot area, two ticks on the x axis", "points", "andreani-1B",
          A1B_TICKS, held_out=True),
    _case("andreani-1B-area-ticks-colours", A1, "points", "the plot area, x ticks, the colours", "points",
          "andreani-1B", {**A1B_TICKS, "colours": A1B_COLOURS}, held_out=True),
    _case("andreani-1C-nothing", A1, "lines", "nothing", "curves", "andreani-1C", held_out=True),
    _case("andreani-1E-area", A1, "bars", "the plot area", "bars", "andreani-1E",
          {"plot_area": [BOX["andreani-1E"]]}, held_out=True),
    _case("andreani-1E-area-ticks", A1, "bars", "the plot area, two ticks", "bars", "andreani-1E",
          {"plot_area": [BOX["andreani-1E"]], "y_ref": [(1089, 0), (826.8, 200)]}, held_out=True),
    *[_case(f"andreani-3{p}-nothing", A3, "points", "nothing", "dots", f"andreani-3{p}", held_out=True)
      for p in "ABCD"],
    *[_case(f"andreani-3{p}-colours", A3, "points", "the colours", "dots", f"andreani-3{p}",
            {"colours": A3_COLOURS}, held_out=True) for p in "ABCD"],
]


# ----------------------------------------------------------------------------------------------- reading a figure
def read(reader, figure, options, out_dir=None):
    """Run one of trace-digitiser's readers on a figure; the data it returns, its panels and its warnings."""
    function = {"lines": td.digitise, "points": td.digitise_points, "bars": td.digitise_bars}[reader]
    kwargs = dict(options)
    if out_dir is not None:
        kwargs.update(write_outputs=True, output_dir=str(out_dir), save_overlay=True)
    else:
        kwargs["write_outputs"] = False
    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # what it warns of is in result.warnings
            result = function(str(FIGURES / figure), **kwargs)
    except Exception as exc:  # a failure is a result too
        return {"error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - start}
    data = result.trace_data if reader == "lines" else result.data
    if reader == "lines" and len(data):
        data = data.rename(columns={"center_estimate": "y_value", "trace_id": "series"})
    return {"data": data, "panels": result.provenance.get("panels", []), "warnings": list(result.warnings),
            "seconds": time.time() - start}


def pick(panels, box):
    """The panel whose plot area overlaps the box most, or None."""
    best, chosen = 0.0, None
    for p in panels:
        a = p.get("plot_area") or {}
        if not a:
            continue
        w = min(box[2], a["right"]) - max(box[0], a["left"])
        h = min(box[3], a["bottom"]) - max(box[1], a["top"])
        if w > 0 and h > 0 and w * h > best:
            best, chosen = w * h, p
    return chosen


def _calibrated(panel, axis):
    cal = panel.get(f"{axis}_calibration") or {}
    return bool(cal) and cal.get("scale_type") != "normalised_0_to_1"


# ------------------------------------------------------------------------------------------------ comparisons
def assignment(cost):
    """The rows and columns of the one-to-one pairing of least total cost (as scipy.optimize.linear_sum_assignment
    gives them): the Hungarian method, written out so that nothing beyond trace-digitiser need be installed."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    flip = cost.shape[0] > cost.shape[1]
    if flip:
        cost = cost.T
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner = np.zeros(m + 1, dtype=int)  # owner[j]: the row (counted from 1) given column j; owner[0]: the row added
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        owner[0], j0 = i, 0
        least = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            free = ~used
            reduced = np.concatenate(([np.inf], cost[i0 - 1] - u[i0] - v[1:]))
            lower = free & (reduced < least)
            least[lower] = reduced[lower]
            way[lower] = j0
            candidates = np.where(free, least, np.inf)
            j1 = int(np.argmin(candidates))
            delta = candidates[j1]
            u[owner[used]] += delta
            v[used] -= delta
            least[free] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    cols = np.nonzero(owner[1:])[0]
    rows = owner[1:][cols] - 1
    if flip:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


def _num(row, column):
    value = row.get(column) if column in row else None
    return float(value) if value is not None and pd.notna(value) else None


def compare_points(found, known, xspan, yspan):
    """Known points ({series: [[x, y, error], ...]}) against those found."""
    kn = [(s, x, y, e) for s, pts in known.items() for x, y, e in pts]
    f = found.dropna(subset=["x_value", "y_value"]).reset_index(drop=True) if len(found) else found
    out = {"known": len(kn), "found": int(len(f)), "matched": 0}
    if not len(f):
        return out
    kx, ky = np.array([k[1] for k in kn]) / xspan, np.array([k[2] for k in kn]) / yspan
    fx, fy = f["x_value"].to_numpy(float) / xspan, f["y_value"].to_numpy(float) / yspan
    cost = np.hypot(kx[:, None] - fx[None, :], ky[:, None] - fy[None, :])
    ri, ci = assignment(cost)
    good = cost[ri, ci] <= POINT_TOL
    ri, ci = ri[good], ci[good]
    out.update(matched=int(len(ri)), extra=int(len(f)) - int(len(ri)))
    if not len(ri):
        return out
    dx, dy = np.abs(kx[ri] - fx[ci]) * 100, np.abs(ky[ri] - fy[ci]) * 100
    out.update(x_median=float(np.median(dx)), x_worst=float(dx.max()), y_median=float(np.median(dy)),
               y_worst=float(dy.max()))
    errs, n_known, unexpected = [], 0, 0
    for k, j in zip(ri, ci):
        row = f.iloc[j]
        y, lo, hi = row["y_value"], _num(row, "y_error_low"), _num(row, "y_error_high")
        halves = [h for h in ((hi - y) if hi is not None else None, (y - lo) if lo is not None else None)
                  if h is not None and h > 0]
        e = kn[k][3]
        if e is None:
            unexpected += bool(halves)
            continue
        n_known += 1
        if halves:  # its half-length (the mean of the two sides) against the known one
            errs.append(abs(float(np.mean(halves)) - e) / yspan * 100)
    out.update(errbars_known=n_known, errbars_found=len(errs), errbars_unexpected=unexpected)
    if errs:
        out.update(errbar_median=float(np.median(errs)), errbar_worst=float(max(errs)))
    return out


def compare_curves(found, known, yspan):
    """For each known curve ({series: {"x": [...], "y": [...]}} or points), the found trace that follows it best."""
    f = found.dropna(subset=["x_value", "y_value"]) if len(found) else found
    groups = [g.sort_values("x_value") for _, g in f.groupby("series")] if len(f) else []
    out = {}
    for name, curve in known.items():
        if isinstance(curve, dict):
            xs, ys = np.asarray(curve["x"], float), np.asarray(curve["y"], float)
        else:
            xs, ys = np.array([p[0] for p in curve], float), np.array([p[1] for p in curve], float)
        best = None
        for g in groups:
            gx, gy = g["x_value"].to_numpy(float), g["y_value"].to_numpy(float)
            inside = (xs >= gx.min()) & (xs <= gx.max()) if len(gx) > 1 else np.zeros(len(xs), bool)
            if inside.sum() < 2:
                continue
            err = np.abs(np.interp(xs[inside], gx, gy) - ys[inside]) / yspan * 100
            cand = {"coverage": float(inside.mean()), "median": float(np.median(err)), "worst": float(err.max())}
            if best is None or cand["coverage"] > best["coverage"] + 0.2 or (
                    abs(cand["coverage"] - best["coverage"]) <= 0.2 and cand["median"] < best["median"]):
                best = cand
        out[name] = best
    return out


def compare_bars(found, known, yspan):
    """Known bars ([[label, value, error], ...], left to right) against those found, paired left to right."""
    f = found.sort_values("x_pixel").reset_index(drop=True) if len(found) else found
    out = {"known": len(known), "found": int(len(f)), "pairs": []}
    errs, ebs = [], []
    n_err = sum(e is not None for _, _, e in known)
    for (label, v, e), (_, row) in zip(known, f.iterrows()):
        errs.append(float(abs(row["value"] - v) / yspan * 100))
        out["pairs"].append([label, v, float(row["value"])])
        hi, lo = _num(row, "error_high"), _num(row, "error_low")
        if e is not None and hi is not None:
            # its half-length: the longer side, since a bar often hides one side of its error bar
            half = max(hi - row["value"], row["value"] - lo if lo is not None else 0.0)
            ebs.append(float(abs(half - e) / yspan * 100))
    out["within"] = int(sum(x <= BAR_TOL for x in errs))
    if errs:
        out.update(median=float(np.median(errs)), worst=float(max(errs)))
    out.update(errbars_known=n_err, errbars_found=len(ebs))
    if ebs:
        out.update(errbar_median=float(np.median(ebs)), errbar_worst=float(max(ebs)))
    return out


def compare_dots(found, known, yspan):
    """The dots of a dot plot ({column: [value, ...]}, columns left to right) against those found: the coloured
    marks found are split into the columns at the widest gaps between them, then paired by value in each column."""
    columns = list(known.items())
    n_known = sum(len(v) for _, v in columns)
    d = found.dropna(subset=["y_value"]) if len(found) else found
    if len(d):  # the dots are coloured; the marks of the means and error bars drawn over them are black
        rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in d["colour"]])
        d = d[(rgb.max(axis=1) - rgb.min(axis=1)) > 60].sort_values("x_pixel")
    out = {"known": n_known, "found": int(len(d)), "matched": 0}
    if len(d) < len(columns):
        return out
    xs = d["x_pixel"].to_numpy(float)
    cuts = sorted(int(g) + 1 for g in np.argsort(-np.diff(xs))[: len(columns) - 1])
    errs, per = [], {}
    for (name, values), idx in zip(columns, np.split(np.arange(len(xs)), cuts)):
        vals, got = np.asarray(values, float), d.iloc[idx]["y_value"].to_numpy(float)
        ri, ci = assignment(np.abs(vals[:, None] - got[None, :]) / yspan)
        ok = np.abs(vals[ri] - got[ci]) / yspan <= DOT_TOL
        errs += [float(v) for v in np.abs(vals[ri] - got[ci])[ok] / yspan * 100]
        per[name] = f"{int(ok.sum())} of {len(vals)} (found {len(got)})"
    out.update(matched=len(errs), extra=int(len(d)) - len(errs), columns=per)
    if errs:
        out.update(y_median=float(np.median(errs)), y_worst=float(np.max(errs)))
    return out


# ------------------------------------------------------------------------------------------------- a case
NEEDED = {"points": ["x_value", "y_value"], "curves": ["x_value", "y_value"], "bars": ["value"], "dots": ["y_value"]}


def _size(kind, part):
    if kind in ("curves", "bars"):
        return len(part)
    return sum(len(v) for v in part.values())


def evaluate(case, run, known):
    """Compare a run with the known values: (details, one-line summary, how many were read, of how many)."""
    k = known[case["known"]]
    xspan, yspan = SPAN[case["known"]]
    kind = case["compare"]
    parts = [(box, {s: k[s] for s in names} if names else k) for box, names in case["parts"]]
    total = sum(_size(kind, part) for _, part in parts)
    if "error" in run:
        return {"error": run["error"]}, f"failed: {run['error'][:100]}", 0, total
    data, details, notes, combined, score = run["data"], [], [], None, 0
    for box, part in parts:
        panel = pick(run["panels"], box)
        if panel is None:
            notes.append("plot not found")
            details.append({"error": "plot not found"})
            continue
        sub = data[data["panel_id"] == panel["panel_id"]] if len(data) and "panel_id" in data else data.iloc[0:0]
        missing = [a for a in (("y",) if kind in ("bars", "dots") else ("x", "y")) if not _calibrated(panel, a)]
        if not missing and len(sub) and any(col not in sub for col in NEEDED[kind]):
            missing = [col.split("_")[0] for col in NEEDED[kind] if col not in sub]
        if missing:
            label = " and ".join(missing) + (" axes" if len(missing) > 1 else " axis") + " not read"
            notes.append(label if kind != "curves" else "; ".join(f"{s}: {label}" for s in part))
            details.append({"error": label, "found": int(len(sub))})
            continue
        if kind == "curves":
            r = compare_curves(sub, part, yspan)
            for name, c in r.items():
                score += bool(c is not None and c["coverage"] >= 0.5 and c["median"] <= CURVE_TOL)
                if c is None:
                    notes.append(f"{name}: not found")
                else:
                    cover = f" over {c['coverage']:.0%} of it" if c["coverage"] < 0.95 else ""
                    notes.append(f"{name}: {c['median']:.2f}% (worst {c['worst']:.1f}%){cover}")
        elif kind == "points":
            r = compare_points(sub, part, xspan, yspan)
            combined = _add(combined, r)
        elif kind == "bars":
            r = combined = compare_bars(sub, part, yspan)
        else:
            r = combined = compare_dots(sub, part, yspan)
        details.append(r)
    if combined is not None:
        score = combined["within"] if kind == "bars" else combined["matched"]
        notes.insert(0, _bars_note(combined, total) if kind == "bars" else _points_note(combined, total, kind))
    return details[0] if len(details) == 1 else details, "; ".join(notes), score, total


def _add(a, b):
    """Totals of two parts' point comparisons (medians and worst errors: of the part with more points)."""
    if a is None:
        return b
    out = {k: a.get(k, 0) + b.get(k, 0) for k in ("known", "found", "matched", "extra", "errbars_known",
                                                  "errbars_found", "errbars_unexpected")}
    big = a if a.get("matched", 0) >= b.get("matched", 0) else b
    for key in ("x_median", "y_median", "errbar_median"):
        if key in big:
            out[key] = big[key]
    for key in ("x_worst", "y_worst", "errbar_worst"):
        vals = [d[key] for d in (a, b) if key in d]
        if vals:
            out[key] = max(vals)
    return out


def _points_note(r, total, what):
    s = f"{r['matched']}/{total} {what}"
    if r.get("extra"):
        s += f" (+{r['extra']} extra)"
    if r["matched"]:
        s += f", y error {r['y_median']:.2f}% (worst {r['y_worst']:.2f}%)"
    if r.get("errbars_known"):
        s += f", error bars {r['errbars_found']}/{r['errbars_known']}"
        if r.get("errbars_found"):
            s += f" ({r['errbar_median']:.2f}%)"
    if r.get("errbars_unexpected"):
        s += f", error bars on {r['errbars_unexpected']} points that have none"
    return s


def _bars_note(r, total):
    s = f"{r['within']}/{total} bars within {BAR_TOL:g}%"
    if r["found"] != r["known"]:
        s += f" ({r['found']} found)"
    if "median" in r:
        s += f", error {r['median']:.2f}% (worst {r['worst']:.2f}%)"
    if r.get("errbars_known"):
        s += f", error bars {r['errbars_found']}/{r['errbars_known']}"
        if r.get("errbars_found"):
            s += f" ({r['errbar_median']:.2f}%)"
    return s


# ------------------------------------------------------------------------------------------------- main
def _environment():
    env = {"trace_digitiser": td.__version__, "python": platform.python_version(),
           "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    for name in ("numpy", "pandas", "cv2"):
        try:
            env["opencv" if name == "cv2" else name] = __import__(name).__version__
        except ImportError:
            pass
    try:
        import pytesseract

        env["tesseract"] = str(pytesseract.get_tesseract_version())
    except Exception:  # no Tesseract: the axes are then read only where ticks are given
        env["tesseract"] = None
    return env


def _totals(cases, results):
    out = {}
    for group, held in (("development", False), ("held out", True)):
        names = [c["name"] for c in cases if c["held_out"] == held and c["name"] in results]
        if names:
            out[group] = (sum(results[n]["score"] for n in names), sum(results[n]["of"] for n in names))
    return out


def _line(c, r, version, before):
    old = (before or {}).get("cases", {}).get(c["name"])
    mark = ""
    if old is not None and old["score"] != r["score"]:
        mark = "   (better)" if r["score"] > old["score"] else "   (WORSE)"
    text = f"{c['name']:36s} given {c['given']}  [{r['seconds']:.0f} s]\n    {version}: {r['summary']}{mark}"
    if old is not None:
        text += f"\n    {before['environment']['trace_digitiser']}: {old['summary']}"
    return text


def _markdown(cases, results, version, before):
    head = ["Case", "Given"] + ([before["environment"]["trace_digitiser"]] if before else []) + [version]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in cases:
        cells = [f"`{c['name']}`", c["given"]]
        if before:
            cells.append(before.get("cases", {}).get(c["name"], {}).get("summary", "-"))
        cells.append(results[c["name"]]["summary"])
        rows.append("| " + " | ".join(x.replace("|", "\\|") for x in cells) + " |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="run only the cases whose names start with one of these")
    ap.add_argument("--json", type=Path, help="save the results in this file")
    ap.add_argument("--compare", type=Path, help="results saved before (with --json), to show beside these")
    ap.add_argument("--results", type=Path, help="show results saved before (with --json) instead of reading")
    ap.add_argument("--markdown", action="store_true", help="print the results as a Markdown table")
    ap.add_argument("--out", type=Path, help="write each run's files, and its overlay, in a folder in this one")
    ap.add_argument("--list", action="store_true", help="list the cases and stop")
    args = ap.parse_args(argv)
    cases = [c for c in CASES if not args.names or any(c["name"].startswith(n) for n in args.names)]
    if args.list or not cases:
        for c in cases or CASES:
            print(f"{c['name']:36s} {c['reader']:6s} given {c['given']}")
        return 0 if cases else 1
    before = json.loads(args.compare.read_text()) if args.compare else None
    if args.results:
        saved = json.loads(args.results.read_text())
        env = saved["environment"]
        cases = [c for c in cases if c["name"] in saved["cases"]]
        results = {c["name"]: saved["cases"][c["name"]] for c in cases}
        if not args.markdown:
            for c in cases:
                print(_line(c, results[c["name"]], env["trace_digitiser"], before))
    else:
        known = json.loads((HERE / "known.json").read_text())
        env = _environment()
        print(f"trace-digitiser {env['trace_digitiser']}, Tesseract {env['tesseract']}", file=sys.stderr)
        runs, results = {}, {}
        for c in cases:
            key = (c["reader"], c["figure"], json.dumps(c["options"], sort_keys=True))
            if key not in runs:
                print(f"  reading {c['figure']} ({c['reader']}; given {c['given']}) ...", file=sys.stderr, flush=True)
                out_dir = args.out / c["name"] if args.out else None
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                runs[key] = read(c["reader"], c["figure"], c["options"], out_dir)
            run = runs[key]
            details, summary, score, total = evaluate(c, run, known)
            results[c["name"]] = {"figure": c["figure"], "reader": c["reader"], "given": c["given"],
                                  "options": c["options"], "held_out": c["held_out"], "summary": summary,
                                  "score": int(score), "of": int(total), "seconds": round(run["seconds"], 1),
                                  "details": details, "warnings": run.get("warnings", [])[:20]}
            if not args.markdown:
                print(_line(c, results[c["name"]], env["trace_digitiser"], before), flush=True)
        print(f"{sum(r['seconds'] for r in runs.values()):.0f} s reading", file=sys.stderr)
        if args.json:
            text = json.dumps({"environment": env, "cases": results}, indent=1,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o))
            args.json.write_text(text + "\n")
    if args.markdown:
        print(_markdown(cases, results, env["trace_digitiser"], before))
    print()
    old_totals = _totals(cases, before["cases"]) if before else {}
    for group, (score, total) in _totals(cases, results).items():
        line = f"{group}: {score} of {total} read"
        if group in old_totals:
            line += f" ({old_totals[group][0]} with {before['environment']['trace_digitiser']})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
