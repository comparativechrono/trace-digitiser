"""Real activity records drawn as dense traces, read back with trace-digitiser, and compared with their hourly means.

    python validation/dense-traces/run.py        # after records.py

A week of activity recorded every minute or every 5 min, drawn in a plot a few hundred pixels wide, puts several
samples in each pixel column, and the line through them makes each column a vertical stroke from the lowest to the
highest sample within it: Lopatina et al. (2014), Fig 4, the figure of the CD157 re-analysis, is drawn so. The line
reader gives four values for each column: the median of the trace's pixels in it (the midpoint of the stroke), their
mean, and the top and the bottom of the stroke. None is the mean of the samples, which the drawing does not record.
This tests which of them, averaged within each hour, comes closest to the record's hourly means, and how far the
rhythm measures drawn from them are from the record's own.

Each of the 9 records of records.csv is drawn as Lopatina et al.'s figure is: a line through every sample, in plot
areas the size of the two copies of that figure (7 days across 551 x 217 or 347 x 137 pixels; the 2-day record in
proportion), with grey gridlines at sixths of the axis, drawn at 4 times the size and reduced, and saved as a JPEG;
once in blue and once in red (Excel's first two series colours), the red read with the gridlines left out, as the
CD157 knockout's pale trace is. The reader is given the plot area and the axis limits, so that only the reading of
the columns is tested. Hourly values are the mean of each statistic over the columns within each hour, leaving out
columns the reader filled in from their neighbours except on the trace's floor, as cd157.py makes them.

Writes figures/*.jpg (the 36 images), results/readings.csv (one row per record, size, colour and statistic),
results/pairs.csv (the difference between two records drawn alike, as the CD157 knockout against the wild type),
results/hourly.csv and results/summary.csv, and prints the tables of README.md.
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import trace_digitiser as td

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "activity-traces"))
import compare  # noqa: E402
import rhythm  # noqa: E402

warnings.simplefilter("ignore")
SS = 4  # drawn at 4 times the size, then reduced, as a publisher's copy is
SIZES = {"PMC-sized": (551, 217), "journal-sized": (347, 137)}  # plot area (px) for 7 days, as the two copies of Fig 4
COLOURS = {"blue": "#4f81bd", "red": "#c0504d"}  # Excel's first two series colours
GRID = (175, 175, 175)
STATISTICS = {"top": "upper_envelope", "midpoint": "center_estimate", "mean of pixels": "mean_estimate",
              "bottom": "lower_envelope"}


def hourly_truth(t, v, span):
    return np.array([np.nanmean(v[(t >= h) & (t < h + 1)]) for h in range(int(span))])


def draw(t, v, span, size, colour, path):
    """The record as a line through every sample, in a plot area of the given size; returns the plot area (pixels)
    and the top of the y axis (the record's 99.9th percentile at 85% of the axis)."""
    w7, h = SIZES[size]
    w = w7 * span / 168
    m = 20
    ymax = float(np.nanpercentile(v, 99.9) / 0.85)
    W, H = int(round(w + 2 * m)), int(round(h + 2 * m))
    im = Image.new("RGB", (W * SS, H * SS), "white")
    dr = ImageDraw.Draw(im)
    for k in range(7):
        y = (m + h - k * h / 6) * SS
        dr.line([(m * SS, y), ((m + w) * SS, y)], fill=GRID, width=SS)
    dr.line([(m * SS, m * SS), (m * SS, (m + h) * SS)], fill=GRID, width=SS)
    ok = np.isfinite(v)
    pts = [((m + ti / span * w) * SS, (m + h - vi / ymax * h) * SS) for ti, vi in zip(t[ok], v[ok])]
    dr.line(pts, fill=colour, width=SS, joint="curve")
    im.resize((W, H), Image.LANCZOS).save(path, quality=85)
    return (m, m, m + w, m + h), ymax


def read(path, area, ymax, span, colour_name):
    """Read the image as a user would: the plot area drawn, the trace's colour clicked (here the median colour of the
    plot's coloured pixels), the axis limits given; for the red trace, the gridlines and the left side of the frame
    left out."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    x0, y0, x1, y1 = area
    sub = rgb[int(y0) + 2:int(y1) - 1, int(x0) + 2:int(x1) - 1].reshape(-1, 3)
    picked = np.median(sub[(sub.max(1) - sub.min(1)) > 40], axis=0).astype(int)
    opts = {"plot_area": [area], "x_ref": [(x0, 0), (x1, span)], "y_ref": [(y1, 0), (y0, ymax)],
            "write_outputs": False, "ocr": False}
    if colour_name == "red":
        hh = y1 - y0
        opts["exclude"] = [(x0 - 2, y1 - k * hh / 6 - 1.5, x1 + 2, y1 - k * hh / 6 + 1.5) for k in range(7)]
        opts["exclude"].append((x0 - 2, y0 - 2, x0 + 1.5, y1 + 2))
    return td.digitise(path, trace_colours=["#{:02x}{:02x}{:02x}".format(*picked)], **opts).trace_data, picked


def hourly_read(d, col, span):
    g = d[["x_value", col, "interpolated"]].rename(columns={"x_value": "x", col: "y"}).dropna(subset=["x", "y"])
    g = g.sort_values("x")
    seen = g[~compare.on_floor_or_seen(g)]
    x, y = seen["x"].to_numpy(float), seen["y"].to_numpy(float)
    return np.array([y[(x >= h) & (x < h + 1)].mean() if ((x >= h) & (x < h + 1)).any() else np.nan
                     for h in range(int(span))])


def dark_start_of(true):
    """Lights-off, taken as the start of the 12 h of the day holding the largest share of the record's activity (the
    clock times of the records' light-dark cycles are not all given)."""
    hrs = np.arange(len(true)) + 0.5
    return int(np.argmax([rhythm.dark_light(hrs, true, s)["dark_fraction"] for s in range(24)]))


def measures(est, true, dark_start):
    hrs = np.arange(len(true)) + 0.5
    ok = np.isfinite(est) & np.isfinite(true)
    ce, ct = rhythm.cosinor(hrs[ok], est[ok], 24.0), rhythm.cosinor(hrs[ok], true[ok], 24.0)
    de = rhythm.dark_light(hrs[ok], est[ok], dark_start)["dark_fraction"]
    dt = rhythm.dark_light(hrs[ok], true[ok], dark_start)["dark_fraction"]
    daily = []
    for k in range(len(true) // 24):
        w = ok & (hrs >= 24 * k) & (hrs < 24 * (k + 1))
        e = rhythm.cosinor(hrs[w], est[w], 24.0)["acrophase"] - rhythm.cosinor(hrs[w], true[w], 24.0)["acrophase"]
        daily.append((e + 12) % 24 - 12)
    return {"hours_read": int(ok.sum()), "r": float(np.corrcoef(est[ok], true[ok])[0, 1]),
            "level": float(est[ok].mean() / true[ok].mean()), "mean_true": float(true[ok].mean()),
            "mean_read": float(est[ok].mean()), "acrophase_true_h": ct["acrophase"],
            "acrophase_read_h": ce["acrophase"],
            "acrophase_error_h": (ce["acrophase"] - ct["acrophase"] + 12) % 24 - 12,
            "daily_acrophase_max_abs_error_h": float(np.max(np.abs(daily))),
            "dark_share_true": dt, "dark_share_read": de, "dark_share_error": de - dt}


def pairs(res):
    rows = []
    for (size, colour, stat), g in res.groupby(["size", "colour", "statistic"]):
        g = g.set_index("record")
        week = [r for r in g.index if r != "LMA per minute"]
        for a, b in itertools.combinations(week, 2):
            A, B = g.loc[a], g.loc[b]
            tdiff = (B.acrophase_true_h - A.acrophase_true_h + 12) % 24 - 12
            rdiff = (B.acrophase_read_h - A.acrophase_read_h + 12) % 24 - 12
            rows.append({"size": size, "colour": colour, "statistic": stat, "record_a": a, "record_b": b,
                         "acrophase_difference_true_h": tdiff, "acrophase_difference_error_h": rdiff - tdiff,
                         "ratio_of_means_true": B.mean_true / A.mean_true,
                         "ratio_of_means_read": B.mean_read / A.mean_read,
                         "ratio_error_factor": (B.mean_read / A.mean_read) / (B.mean_true / A.mean_true),
                         "dark_share_difference_true": B.dark_share_true - A.dark_share_true,
                         "dark_share_difference_error": (B.dark_share_read - A.dark_share_read)
                         - (B.dark_share_true - A.dark_share_true)})
    return pd.DataFrame(rows)


def summary(res, pr):
    out = []
    for stat in STATISTICS:
        a, b = res[res.statistic == stat], pr[pr.statistic == stat]
        ae, be = a.acrophase_error_h.abs(), b.acrophase_difference_error_h.abs()
        lr = np.abs(np.log(b.ratio_error_factor))
        out.append({"statistic": stat, "readings": len(a), "pairs": len(b),
                    "r_median": a.r.median(), "level_median": a.level.median(), "level_min": a.level.min(),
                    "level_max": a.level.max(), "acrophase_abs_error_median_h": ae.median(),
                    "acrophase_abs_error_p90_h": np.percentile(ae, 90), "acrophase_abs_error_max_h": ae.max(),
                    "dark_share_error_median": a.dark_share_error.median(),
                    "dark_share_error_min": a.dark_share_error.min(), "dark_share_error_max": a.dark_share_error.max(),
                    "pair_acrophase_abs_error_median_h": be.median(), "pair_acrophase_abs_error_p90_h":
                    np.percentile(be, 90), "pair_acrophase_abs_error_max_h": be.max(),
                    "pair_ratio_error_median_pct": 100 * (np.exp(lr.median()) - 1),
                    "pair_ratio_error_p90_pct": 100 * (np.exp(np.percentile(lr, 90)) - 1),
                    "pair_dark_share_abs_error_median": b.dark_share_difference_error.abs().median(),
                    "pair_dark_share_abs_error_p90": np.percentile(b.dark_share_difference_error.abs(), 90)})
    return pd.DataFrame(out)


def markdown(sm, res):
    lines = ["| Column statistic | r | Level, read ÷ true | Acrophase error | Dark share, read − true "
             "| Two records: acrophase difference error | Two records: ratio of means error "
             "| Two records: dark-share difference error |", "|---|---|---|---|---|---|---|---|"]
    for _, s in sm.iterrows():
        lines.append(
            f"| {s.statistic} | {s.r_median:.2f} | {s.level_median:.2f} ({s.level_min:.2f}–{s.level_max:.2f}) "
            f"| {s.acrophase_abs_error_median_h:.2f} h (90%: {s.acrophase_abs_error_p90_h:.1f} h) "
            f"| {s.dark_share_error_median:+.3f} ({s.dark_share_error_min:+.2f} to {s.dark_share_error_max:+.2f}) "
            f"| {s.pair_acrophase_abs_error_median_h:.2f} h (90%: {s.pair_acrophase_abs_error_p90_h:.1f} h) "
            f"| {s.pair_ratio_error_median_pct:.0f}% (90%: {s.pair_ratio_error_p90_pct:.0f}%) "
            f"| {s.pair_dark_share_abs_error_median:.3f} (90%: {s.pair_dark_share_abs_error_p90:.3f}) |")
    by = res.copy()
    by["kind"] = by.record.str.replace(r" mouse \d", " (one mouse)", regex=True)
    t = by.groupby(["kind", "statistic"], sort=False).agg(
        samples=("samples_per_column", "median"), level=("level", "median"), dark=("dark_share_error", "median"),
        acr=("acrophase_error_h", lambda x: np.median(np.abs(x)))).reset_index()
    lines += ["", "| Record | Samples per column | Column statistic | Level, read ÷ true | Dark share, read − true "
              "| Acrophase error |", "|---|---|---|---|---|---|"]
    for _, s in t.iterrows():
        lines.append(f"| {s.kind} | {s.samples:.1f} | {s.statistic} | {s.level:.2f} | {s.dark:+.3f} | {s.acr:.2f} h |")
    return "\n".join(lines)


def main():
    recs = pd.read_csv(HERE / "records.csv")
    rows, hourly = [], []
    for (rec, g), size, colour in itertools.product(recs.groupby("record", sort=False), SIZES, COLOURS):
        t, v = g.t_h.to_numpy(float), g.value.to_numpy(float)
        span = 48 if rec == "LMA per minute" else 168
        true = hourly_truth(t, v, span)
        ds = dark_start_of(true)
        stem = f"{rec}-{size}-{colour}".replace(" ", "_").replace(",", "")
        path = HERE / "figures" / f"{stem}.jpg"
        area, ymax = draw(t, v, span, size, COLOURS[colour], str(path))
        d, picked = read(str(path), area, ymax, span, colour)
        per_col = np.isfinite(v).sum() / max(1, len(d))
        for stat, col in STATISTICS.items():
            est = hourly_read(d, col, span)
            m = measures(est, true, ds)
            m.update({"record": rec, "size": size, "colour": colour, "statistic": stat, "columns": len(d),
                      "samples_per_column": per_col, "dark_start_h": ds, "y_max": ymax,
                      "colour_clicked": "#{:02x}{:02x}{:02x}".format(*picked), "image": f"figures/{stem}.jpg"})
            rows.append(m)
            hourly.append(pd.DataFrame({"record": rec, "size": size, "colour": colour, "statistic": stat,
                                        "hour": np.arange(span), "true": true, "read": est}))
    res = pd.DataFrame(rows)
    front = ["record", "size", "colour", "statistic", "columns", "samples_per_column"]
    res = res[front + [c for c in res.columns if c not in front]]
    res.to_csv(HERE / "results" / "readings.csv", index=False, float_format="%.4g")
    pd.concat(hourly).to_csv(HERE / "results" / "hourly.csv", index=False, float_format="%.5g")
    pr = pairs(res)
    pr.to_csv(HERE / "results" / "pairs.csv", index=False, float_format="%.4g")
    sm = summary(res, pr)
    sm.to_csv(HERE / "results" / "summary.csv", index=False, float_format="%.4g")
    print(markdown(sm, res))


if __name__ == "__main__":
    main()
