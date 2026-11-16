"""Which LOWESS window a figure's fitted lines were drawn with, found from the figure itself.

    python validation/activity-traces/lowess_window.py [CASE]      # default: PMC12370043-fig2

Sayler & Hammond (2025, Fig 2) draw each hourly mean as a marker and through the markers a LOWESS fit, made in Prism
(with "a 20-point smoothing window", the legend says). Prism's documentation does not say how it counts the window
where there are several values at each time, or whether it reweights the values by their residuals; and a LOWESS
over 20 of the hourly means is far smoother than the lines drawn. So the window is found from the figure: its
markers are read off the image with the points reader, LOWESS fits to them with windows of k markers (with and
without robustness passes) are compared with the line drawn, as the line reader read it (digitised/<case>.csv),
half way between the markers, where only the line is drawn; and the window that best gives the drawn lines is the
one compare.py fits to the authors' own means (cases.json, "reference"). The window is found without the authors'
data (but for their times).

Then, for the record, the markers against the authors' hourly means at the same hours: how far the means plotted
are from those of the deposited file.

The markers stand among other markers and error bars of their colour, which the points reader takes for the pieces
of a word round a letter, and so leaves out; for this reading its filter for letters is turned off. Markers hidden by
the other group's are missing, and the fits go without them.

Writes results/lowess_window.csv: for each series, window (k) and number of robustness passes, the mean absolute
difference between the fit to the markers and the drawn line, as a percentage of the range of the markers; and
results/lowess_markers.csv: for each series, the markers found at the authors' hours, their median ratio to the
authors' means there, the ratio of their 24-h cosinor amplitudes, and their correlation with the authors' means of
the hour before, the same hour and the hour after.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
import rhythm  # noqa: E402

KS = (3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
ROBUST = (0, 2)


def markers(image, panel):
    """The markers of each series of a panel, as (x, y) arrays, read with the points reader."""
    import trace_digitiser as td
    import trace_digitiser.markers as tm
    from trace_digitiser.trace_detection import parse_colour

    keep = tm._not_letters
    tm._not_letters = lambda mask, found, origin, r, rgb_roi=None: found  # see above: no filter for letters here
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = td.digitise_points(str(image), colours=list(panel["series"].values()),
                                   plot_area=[tuple(panel["plot_area"])], write_outputs=False).data
    finally:
        tm._not_letters = keep
    out = {}
    for name, colour in panel["series"].items():
        want = np.array(parse_colour(colour), float)
        got = [np.array(parse_colour(json.loads(c) if str(c).startswith("[") else c), float) for c in d.colour]
        g = d[[np.abs(c - want).sum() < 30 for c in got]].sort_values("x_value")
        out[name] = (g.x_value.to_numpy(float), g.y_value.to_numpy(float))
    return out


def versus(mx, my, times, means):
    """The markers (at mx, my) against the authors' means at their times: those within 0.3 of a time."""
    near = np.abs(mx[:, None] - times[None, :])
    hit = near.min(axis=1) < 0.3
    at = near.argmin(axis=1)[hit]
    m, f = my[hit], means[at]
    ok = np.isfinite(f)
    m, f, t = m[ok], f[ok], times[at][ok]

    def r_lag(lag):
        g = np.array([means[i + lag] if 0 <= i + lag < len(means) else np.nan for i in at[ok]])
        k = np.isfinite(g)
        return float(np.corrcoef(m[k], g[k])[0, 1])

    return {"markers_at_hours": int(len(m)), "median_ratio": float(np.median(m / f)),
            "amplitude_ratio": float(rhythm.cosinor(t, m, 24.0)["amplitude"] / rhythm.cosinor(t, f, 24.0)["amplitude"]),
            "r_hour_before": r_lag(-1), "r_same_hour": r_lag(0), "r_hour_after": r_lag(1)}


def main(case_id="PMC12370043-fig2"):
    case = next(c for c in json.loads((HERE / "cases.json").read_text()) if c["id"] == case_id)
    config = json.loads((HERE / case["config"]).read_text())
    image = (HERE / case["config"]).parent / config["image"]
    dig = pd.read_csv(HERE / case["digitised"])
    truth = pd.read_csv(HERE / case["truth"])
    half = 0.2 * case["time"]["bin_h"]
    rows, against = [], []
    for panel in config["panels"]:
        for name, (mx, my) in markers(image, panel).items():
            tt = truth[(truth.panel == panel["name"]) & (truth.series == name)].sort_values("x")
            times = tt.x.to_numpy(float)
            between = times[:-1] + np.diff(times) / 2
            against.append({"panel": panel["name"], "series": name, **versus(mx, my, times, tt.y.to_numpy(float))})
            g = dig[(dig.panel == panel["name"]) & (dig.series == name)].dropna(subset=["x", "y"])
            line = compare.sample(g, between, half)
            rng = float(np.ptp(my))
            for k in KS:
                for robust in ROBUST:
                    fit = compare.lowess(mx, my, between, k, robust)
                    ok = np.isfinite(line) & np.isfinite(fit)
                    rows.append({"panel": panel["name"], "series": name, "markers": len(mx), "k": k,
                                 "robustness_passes": robust, "n": int(ok.sum()),
                                 "mae_pct": float(np.mean(np.abs(line[ok] - fit[ok])) / rng * 100)})
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "results" / "lowess_window.csv", index=False)
    wide = t[t.robustness_passes == 0].pivot_table(index=["panel", "series", "markers"], columns="k", values="mae_pct")
    pd.set_option("display.width", 200)
    print("Fit to the markers against the line drawn, between the markers (mean abs. difference, % of range):\n")
    print(wide.round(1).to_string())
    med = t.groupby(["robustness_passes", "k"]).mae_pct.median().unstack(0).round(1)
    print("\nMedian over the series (columns: robustness passes):\n")
    print(med.to_string())
    best = t.groupby(["robustness_passes", "k"]).mae_pct.median().idxmin()
    print(f"\nBest: k = {best[1]}, {best[0]} robustness passes")
    a = pd.DataFrame(against)
    a.to_csv(HERE / "results" / "lowess_markers.csv", index=False)
    print("\nThe markers against the authors' hourly means at the same hours:\n")
    print(a.round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
