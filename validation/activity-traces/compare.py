"""Rhythm analysis of every digitised series and, where the paper's data give it, of the series behind it.

    python compare.py CASES.json VAL_DIR OUT_DIR

CASES.json lists the figures: {id, pmcid, group, tools, digitised (CSV from digitise_case.py), truth (CSV from
truth.py, or null), time: {dark_start_h, light_dark, bin_h, fixed_period}, and optionally panel_time (per-panel
overrides of time, e.g. no dark_start_h under constant conditions), panels and series (which to analyse),
conditions (a label per panel), drawn (per panel: where the figure draws each original value, when that is not at
its own time - see drawn_at), and reference (when the figure's line is a fit to the original values rather than the
values themselves: the same fit, and where it is compared - see reference)}. For a figure with data the digitised
trace is read where the figure draws each original value (at its x, unless drawn says otherwise), and both are
analysed at the original x; without data, at regular steps of bin_h across the trace.

Writes, in OUT_DIR:
  series_metrics.csv  every analysis of every series (digitised, and original where there is one)
  comparisons.csv     one row per series: the digitised and original results side by side, and how far apart
  daily.csv           a cosinor per whole day of each series (digitised and original)
  pairs.csv           for each pair of series in a panel, the differences (dark-phase mean, light-phase mean,
                      cosinor amplitude, mean, dark-phase share, acrophase) from each source
  sampled.csv         every series as analysed: x, the digitised value and the original value
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhythm  # noqa: E402

KEYS = ("rhythmic_chi", "chi_applicable", "rhythmic_ls", "rhythmic_cos", "chi_period", "ls_period", "fft_period",
        "cos_amplitude", "cos_acrophase", "cos_mesor", "cos_p", "ls_fap", "dark_mean", "light_mean", "dark_fraction",
        "M10", "L5", "RA", "mean", "span_h")


def sample(g, xs, half):
    """The digitised trace at each x: the median of the values the reader saw (not filled in) within +-half of x,
    widened to the nearest column where the columns are further apart than that; nan where the reader saw nothing
    there, or outside the trace."""
    g = g.sort_values("x")
    if "interpolated" in g:
        seen = g[~on_floor_or_seen(g)]
    else:
        seen = g
    seen = seen.groupby("x", as_index=False)["y"].mean()
    gx, gy = seen["x"].to_numpy(float), seen["y"].to_numpy(float)
    out = np.full(len(xs), np.nan)
    if len(gx) == 0:
        return out
    allx = np.unique(g["x"].to_numpy(float))
    col = float(np.median(np.diff(allx))) if len(allx) > 1 else 0.0
    h = max(half, 0.75 * col)
    lo, hi = allx.min(), allx.max()
    for i, x in enumerate(xs):
        if x < lo - h or x > hi + h:
            continue
        w = (gx >= x - h) & (gx <= x + h)
        if w.any():
            out[i] = np.median(gy[w])
    return out


def on_floor_or_seen(g, frac=0.05):
    """True for the columns to leave out: those the reader filled in, except a gap whose ends (one end, at the
    start or end of the trace) lie on the floor of the trace (within frac of its range above its lowest seen
    value). A trace resting on the x axis is hidden by the axis line, and such a gap is the trace lying at zero
    rather than a stretch the reader lost."""
    interp = g["interpolated"].astype(bool).to_numpy()
    y = g["y"].to_numpy(float)
    seen_y = y[~interp]
    if len(seen_y) == 0:
        return interp
    floor = seen_y.min()
    tol = frac * (seen_y.max() - floor)
    drop = interp.copy()
    i, n = 0, len(interp)
    while i < n:
        if not interp[i]:
            i += 1
            continue
        j = i
        while j < n and interp[j]:
            j += 1
        left = y[i - 1] if i > 0 else None
        right = y[j] if j < n else None
        ends = [v for v in (left, right) if v is not None]
        if ends and all(v <= floor + tol for v in ends):
            drop[i:j] = False
        i = j
    return drop


def aligned(g, xs, ytrue, half, max_lag=2.0, step=0.05):
    """The time shift (h, within +-max_lag) at which the digitised trace best matches the original (largest r), and
    r and the mean absolute error there: what is left once the figure's own time placement is set aside."""
    best = (np.nan, np.nan, np.nan)
    need = max(4, 0.9 * int((np.isfinite(sample(g, xs, half)) & np.isfinite(ytrue)).sum()))
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        yd = sample(g, xs + lag, half)
        ok = np.isfinite(yd) & np.isfinite(ytrue)
        if ok.sum() < need:
            continue
        r = np.corrcoef(yd[ok], ytrue[ok])[0, 1]
        if not np.isfinite(best[0]) or r > best[0]:
            best = (r, float(lag), float(np.mean(np.abs(yd[ok] - ytrue[ok]))))
    return best


def lowess(x, y, at, k, robust=0):
    """LOWESS (Cleveland 1979), the smoother Prism draws: at each x0 a straight line fitted to the k values nearest
    x0, weighted by the tricube of their distance over that of the k-th; then, if *robust*, that many reweightings of
    the values by the bisquare of their residuals. Values that are missing are left out."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.full(len(np.atleast_1d(at)), np.nan)
    k = min(int(k), len(x))
    rw = np.ones(len(x))

    def fit(x0):
        d = np.abs(x - x0)
        h = max(float(np.sort(d)[k - 1]), 1e-9) * 1.000001
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3 * rw
        if w.sum() <= 0:
            return np.nan
        xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
        sxx = (w * (x - xm) ** 2).sum()
        b = (w * (x - xm) * (y - ym)).sum() / sxx if sxx > 0 else 0.0
        return ym + b * (x0 - xm)

    for _ in range(int(robust)):
        res = y - np.array([fit(x0) for x0 in x])
        s = np.median(np.abs(res))
        u = res / (6 * s) if s > 0 else np.zeros_like(res)
        rw = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return np.array([fit(x0) for x0 in np.atleast_1d(np.asarray(at, float))])


def reference(rule, xs, ys):
    """The authors' series as the figure's line shows it, under *rule*: None (the values themselves, at their times);
    "lowess K between" (a figure whose line is a LOWESS fit to the plotted values, drawn through markers that stand
    at those values: the same fit to the authors' values, with a window of K of them and no robustness passes, half
    way between their times, where only the line is drawn; the window is the one with which the figure's own
    markers best give its lines - lowess_window.py). Returns (times, values)."""
    if not rule:
        return xs, ys
    kind, *rest = rule.split()
    if kind == "lowess":
        mid = xs[:-1] + np.diff(xs) / 2 if "between" in rest else xs
        return mid, lowess(xs, ys, mid, int(rest[0]))
    raise ValueError(f"unknown reference rule: {rule}")


def drawn_at(rule, xs):
    """Where the figure draws each original value, by its x (h), under *rule*: None (at x itself); "shift D" (at
    x + D: a figure that draws each hour's mean at the end of its hour, where the data label it by its middle);
    "spread A B C D" (the values at A..B drawn evenly from C to D, and those outside not drawn: nan). The rule is
    read off the figure itself - where its markers, error bars or vertices lie - not fitted to the data."""
    xs = np.asarray(xs, float)
    if not rule:
        return xs
    kind, *nums = rule.split()
    v = [float(n) for n in nums]
    if kind == "shift":
        return xs + v[0]
    if kind == "spread":
        a, b, c, d = v
        return np.where((xs >= a - 1e-9) & (xs <= b + 1e-9), c + (xs - a) * (d - c) / (b - a), np.nan)
    raise ValueError(f"unknown rule for where values are drawn: {rule}")


def grid(g, tm, span=None):
    """The times at which a series without data is read: every bin_h hours (from bin_offset_h) across span, or
    across the trace as read."""
    lo, hi = (span if span else (g["x"].min(), g["x"].max()))
    b = tm["bin_h"]
    off = tm.get("bin_offset_h", 0.0)
    return np.arange(np.ceil((lo - off) / b) * b + off, hi + 1e-9, b)


def circ_diff(a, b, per=24.0):
    return float((a - b + per / 2) % per - per / 2)


def main(cases_path, val_dir, out_dir):
    cases = json.loads(Path(cases_path).read_text())
    val, out = Path(val_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, series_rows, day_rows, pair_rows, sampled_rows = [], [], [], [], []
    for c in cases:
        d = pd.read_csv(val / c["digitised"])
        truth = pd.read_csv(val / c["truth"]) if c.get("truth") else None
        results = {}
        # a case may cut one plotted series into segments (e.g. baseline, treatment), each analysed on its own
        parts = []
        for (panel, series), g in d.groupby(["panel", "series"]):
            segs = [sg for sg in c.get("segments", []) if sg["panel"] == panel and sg["series"] == series]
            if segs:
                for sg in segs:
                    lo, hi = sg["x_range"]
                    parts.append((panel, sg["label"], g[(g["x"] >= lo) & (g["x"] <= hi)], (lo, hi)))
            else:
                parts.append((panel, series, g, None))
        for panel, series, g, seg_range in parts:
            if c.get("series") and series not in c["series"]:
                continue
            if c.get("panels") and panel not in c["panels"]:
                continue
            tm = {**c["time"], **c.get("panel_time", {}).get(panel, {})}
            ds, fp = tm.get("dark_start_h"), tm.get("fixed_period")
            half = 0.2 * tm["bin_h"]
            g = g.dropna(subset=["x", "y"])
            span = seg_range or c.get("x_range", {}).get(f"{panel}|{series}") or c.get("x_range", {}).get(panel)
            if truth is not None:
                tt = truth[(truth.panel == panel) & (truth.series == series)].sort_values("x")
                xs, ytrue = reference(c.get("reference"), tt["x"].to_numpy(float), tt["y"].to_numpy(float))
            else:
                xs = grid(g, tm, span)
                ytrue = None
            # the trace is read where the figure draws each value (nan where it does not draw it)
            xd = drawn_at(c.get("drawn", {}).get(panel), xs) if ytrue is not None else xs
            ydig = np.full(len(xs), np.nan)
            if len(g) > 1:
                ok_x = np.isfinite(xd)
                ydig[ok_x] = sample(g, xd[ok_x], half)
            for i, xv in enumerate(xs):
                sampled_rows.append({"case": c["id"], "pmcid": c["pmcid"], "panel": panel, "series": series,
                                     "x": float(xv), "digitised": float(ydig[i]),
                                     "original": float(ytrue[i]) if ytrue is not None else np.nan})
            cond = c.get("conditions", {}).get(f"{panel}|{series}") or c.get("conditions", {}).get(panel) \
                or tm.get("light_dark")
            if ds is None:
                tm = {**tm}
            key = {"case": c["id"], "pmcid": c["pmcid"], "group": c["group"], "tools": c["tools"],
                   "panel": panel, "series": series, "condition": cond}
            a_dig = rhythm.analyse(xs, ydig, dark_start=ds, fixed_period=fp)
            series_rows.append({**key, "source": "digitised", **a_dig})
            for r in rhythm.daily(xs, ydig, dark_start=ds, start=xs[0] if len(xs) else None):
                day_rows.append({**key, "source": "digitised", **r})
            row = {**key, "coverage": float(np.isfinite(ydig).mean()) if len(ydig) else 0.0}
            row.update({f"dig_{k}": a_dig.get(k) for k in KEYS})
            results[(panel, series, "digitised")] = a_dig
            if ytrue is not None:
                a_true = rhythm.analyse(xs, ytrue, dark_start=ds, fixed_period=fp)
                results[(panel, series, "original")] = a_true
                series_rows.append({**key, "source": "original", **a_true})
                for r in rhythm.daily(xs, ytrue, dark_start=ds, start=xs[0]):
                    day_rows.append({**key, "source": "original", **r})
                ok = np.isfinite(ydig) & np.isfinite(ytrue)
                rng = np.nanmax(ytrue) - np.nanmin(ytrue)
                err = np.abs(ydig[ok] - ytrue[ok])
                row.update(n_values=int(ok.sum()), mae_pct=float(err.mean() / rng * 100),
                           mdae_pct=float(np.median(err) / rng * 100),
                           r=float(np.corrcoef(ydig[ok], ytrue[ok])[0, 1]) if ok.sum() > 2 else np.nan)
                drawn_ok = np.isfinite(xd)
                r_al, lag, mae_al = aligned(g, xd[drawn_ok], ytrue[drawn_ok], half)
                row.update(best_lag_h=lag, r_aligned=float(r_al), mae_aligned_pct=float(mae_al / rng * 100))
                row.update({f"true_{k}": a_true.get(k) for k in KEYS})
                per = a_true.get("cos_period", 24.0)
                row["acrophase_diff_h"] = circ_diff(a_dig["cos_acrophase"], a_true["cos_acrophase"], per)
                row["period_diff_h"] = float(a_dig["ls_period"] - a_true["ls_period"])
                row["amplitude_ratio"] = float(a_dig["cos_amplitude"] / a_true["cos_amplitude"]) \
                    if a_true["cos_amplitude"] else np.nan
                row["dark_fraction_diff"] = float(a_dig["dark_fraction"] - a_true["dark_fraction"]) \
                    if ds is not None else np.nan
            rows.append(row)
        # pairs of series within a panel
        for panel in sorted({k[0] for k in results}):
            names = sorted({k[1] for k in results if k[0] == panel})
            for s1, s2 in itertools.combinations(names, 2):
                pr = {"case": c["id"], "pmcid": c["pmcid"], "group": c["group"], "panel": panel,
                      "series_1": s1, "series_2": s2}
                for src in ("digitised", "original"):
                    a1, a2 = results.get((panel, s1, src)), results.get((panel, s2, src))
                    if not a1 or not a2:
                        continue
                    for k in ("dark_mean", "light_mean", "cos_amplitude", "mean", "dark_fraction"):
                        if k in a1 and k in a2:
                            pr[f"{src}_{k}_1_minus_2"] = float(a1[k] - a2[k])
                    pr[f"{src}_acrophase_1_minus_2"] = circ_diff(a1["cos_acrophase"], a2["cos_acrophase"])
                pair_rows.append(pr)
    pd.DataFrame(series_rows).to_csv(out / "series_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "comparisons.csv", index=False)
    pd.DataFrame(day_rows).to_csv(out / "daily.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairs.csv", index=False)
    pd.DataFrame(sampled_rows).to_csv(out / "sampled.csv", index=False)
    print(f"{len(rows)} series; {sum(1 for r in rows if 'mae_pct' in r)} with original data", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
