"""The CD157 re-analysis of Hearn and Lloyd-Owen: Lopatina et al. (2014) Fig 4 read with trace-digitiser and analysed
with this folder's rhythm analysis, from PMC's copy of the figure (a case of the validation set) and from the
journal's smaller copy (which the submission digitised).

    python validation/activity-traces/cd157.py      # after run.py; reads the journal's copy again if needed

Each copy is read three ways. "primary": hourly values taken as compare.py takes them from every series without
data (on the hour, the median of the columns within 0.2 h), leaving out the few columns of the pale CD157-/- trace
that the line reader took from the plot's grey frame or a gridline (their colour is within the reader's tolerance of
the trace's): a column is left out when the pixel at the value read is grey or white rather than the trace's colour.
"as read": the same with those columns kept, as the validation set reads them. "half-hour grid": as primary, on the
half hour.

Writes results/cd157.csv (one row per copy, reading and genotype), results/cd157_daily.csv (a 24-h cosinor per day),
results/cd157_summary.csv (one row per copy and reading: the knockout against the wild type), results/cd157_misread.csv
(the columns left out) and results/cd157.png (the primary reading of PMC's copy).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import trace_digitiser as td  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402
import digitise_case  # noqa: E402
import rhythm  # noqa: E402

COPIES = {"PMC copy (630 x 597 px)": "PMC4001052-fig4", "journal copy (397 x 376 px)": "cd157-journal-copy"}
GENOTYPES = (("A CD157+/+", "CD157+/+"), ("B CD157-/-", "CD157-/-"))
COLOURS = {"CD157+/+": "#2a78d6", "CD157-/-": "#e34948"}  # categorical slots 1 and 8 of the reference palette
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"


READINGS = ("primary", "as read", "half-hour grid")


def misread(case):
    """The columns read from the plot's frame or a gridline: those whose pixel at the value read is grey or white.

    The reader's calibration of the y axis (pixel row from value) comes from running it again on each panel."""
    cfg = json.loads((HERE / "configs" / f"{case}.json").read_text())
    image = str((HERE / "configs" / cfg["image"]).resolve())
    rgb = np.asarray(Image.open(image).convert("RGB")).astype(int)
    d = pd.read_csv(HERE / "digitised" / f"{case}.csv")
    bad = []
    for panel in cfg["panels"]:
        r = td.digitise(image, trace_colours=list(panel["series"].values()), plot_area=[tuple(panel["plot_area"])],
                        x_ref=panel["x_ref"], write_outputs=False)
        a, b = r.panel_metadata.loc[0, ["y_calibration_a", "y_calibration_b"]]
        for _, row in d[d.panel == panel["name"]].dropna(subset=["y"]).iterrows():
            yy = int(round((row.y - b) / a))
            px = rgb[min(max(yy, 0), rgb.shape[0] - 1), int(row.x_pixel)]
            if px.max() - px.min() < 12:  # grey or white: not the colour of either trace
                bad.append({"case": case, "panel": row.panel, "series": row.series, "x_pixel": int(row.x_pixel),
                            "x_h": row.x, "y_read": row.y, "rgb_at_value": "#{:02x}{:02x}{:02x}".format(*px)})
    return pd.DataFrame(bad, columns=["case", "panel", "series", "x_pixel", "x_h", "y_read", "rgb_at_value"])


def series(case, reading="primary", bad=None):
    path = HERE / "digitised" / f"{case}.csv"
    if not path.exists():
        digitise_case.main(str(HERE / "configs" / f"{case}.json"), str(path))
    d = pd.read_csv(path)
    tm = json.loads((HERE / "configs" / f"{case}.json").read_text())["time"]
    if reading == "half-hour grid":
        tm = dict(tm, bin_offset_h=0.5 * tm["bin_h"])
    out = {}
    for panel, name in GENOTYPES:
        g = d[(d.panel == panel) & (d.series == name)].dropna(subset=["x", "y"])
        if reading != "as read" and bad is not None and len(bad):
            drop = set(bad[(bad.panel == panel) & (bad.series == name)].x_pixel)
            g = g[~g.x_pixel.isin(drop)]
        xs = compare.grid(g, tm)  # as compare.py reads every series without data: on the hour
        out[name] = (xs, compare.sample(g, xs, 0.2 * tm["bin_h"]))
    return out


def main():
    rows, days, summary = [], [], []
    bad = pd.concat([misread(case) for case in COPIES.values()], ignore_index=True)
    bad.to_csv(HERE / "results" / "cd157_misread.csv", index=False, float_format="%.4g")
    primary = {}
    for copy, case in COPIES.items():
        for reading in READINGS:
            read = series(case, reading, bad[bad.case == case])
            if reading == "primary":
                primary[copy] = read
            for name, (x, y) in read.items():
                a = rhythm.analyse(x, y, dark_start=0, fixed_period=24)
                rows.append({"copy": copy, "reading": reading, "genotype": name, "hours_read": a["n"],
                             "mean": a["mean"], "mesor": a["cos_mesor"], "amplitude": a["cos_amplitude"],
                             "acrophase_h_after_dark_onset": a["cos_acrophase"],
                             "acrophase_ZT": (a["cos_acrophase"] + 12) % 24, "cosinor_r2": a["cos_r2"],
                             "cosinor_p": a["cos_p"], "ls_period_h": a["ls_period"], "ls_fap": a["ls_fap"],
                             "chi_square_rhythmic": a["rhythmic_chi"], "chi_square_period_h": a["chi_period"],
                             "dark_mean": a["dark_mean"], "light_mean": a["light_mean"],
                             "dark_share": a["dark_fraction"], "M10": a["M10"], "L5": a["L5"], "RA": a["RA"]})
                for r in rhythm.daily(x, y, dark_start=0, start=0):  # days from the first lights-off
                    days.append({"copy": copy, "reading": reading, "genotype": name, "day": r["day"],
                                 "acrophase_h_after_dark_onset": r["cos_acrophase"],
                                 "amplitude": r["cos_amplitude"], "mesor": r["cos_mesor"],
                                 "dark_share": r["dark_fraction"], "hours_read": r["n"]})
            t = pd.DataFrame([r for r in rows if r["copy"] == copy and r["reading"] == reading]).set_index("genotype")
            dd = pd.DataFrame([r for r in days if r["copy"] == copy and r["reading"] == reading])
            p = dd.pivot(index="day", columns="genotype", values="acrophase_h_after_dark_onset")
            per_day = (p["CD157-/-"] - p["CD157+/+"] + 12) % 24 - 12
            q = dd.pivot(index="day", columns="genotype", values="dark_share")
            wt, ko = t.loc["CD157+/+"], t.loc["CD157-/-"]
            summary.append({
                "copy": copy, "reading": reading,
                "columns_left_out": 0 if reading == "as read" else int((bad.case == case).sum()),
                "acrophase_wt_h": wt.acrophase_h_after_dark_onset, "acrophase_ko_h": ko.acrophase_h_after_dark_onset,
                "acrophase_ko_minus_wt_h": (ko.acrophase_h_after_dark_onset - wt.acrophase_h_after_dark_onset
                                            + 12) % 24 - 12,
                "daily_diff_min_h": per_day.min(), "daily_diff_max_h": per_day.max(),
                "days_ko_earlier": int((per_day < 0).sum()),
                "dark_share_wt": wt.dark_share, "dark_share_ko": ko.dark_share,
                "days_ko_dark_share_higher": int((q["CD157-/-"] > q["CD157+/+"]).sum()),
                "mean_ratio_ko_wt": ko["mean"] / wt["mean"], "amplitude_ratio_ko_wt": ko.amplitude / wt.amplitude,
                "M10_ratio_ko_wt": ko.M10 / wt.M10, "L5_ratio_ko_wt": ko.L5 / wt.L5,
                "RA_wt": wt.RA, "RA_ko": ko.RA, "r2_wt": wt.cosinor_r2, "r2_ko": ko.cosinor_r2})
    pd.DataFrame(rows).to_csv(HERE / "results" / "cd157.csv", index=False, float_format="%.4g")
    dd = pd.DataFrame(days)
    dd.to_csv(HERE / "results" / "cd157_daily.csv", index=False, float_format="%.4g")
    sm = pd.DataFrame(summary)
    sm.to_csv(HERE / "results" / "cd157_summary.csv", index=False, float_format="%.4g")
    print(f"left out as read from the frame or a gridline: {len(bad)} columns "
          + ", ".join(f"{c}: {n}" for c, n in bad.groupby("case").size().items()))
    for _, r in sm.iterrows():
        print(f"{r['copy']}, {r.reading}: acrophase -/- minus +/+ {r.acrophase_ko_minus_wt_h:+.2f} h (days "
              f"{r.daily_diff_min_h:+.1f} to {r.daily_diff_max_h:+.1f} h; earlier on {r.days_ko_earlier}/7); "
              f"dark share {r.dark_share_wt:.3f} vs {r.dark_share_ko:.3f}; mean ratio {r.mean_ratio_ko_wt:.2f}; "
              f"amplitude ratio {r.amplitude_ratio_ko_wt:.2f}")
    print(table(sm))
    first = next(iter(COPIES))
    figure(primary[first], dd[(dd["copy"] == first) & (dd.reading == "primary")])


def table(sm):
    """The summary as the Markdown table in README.md."""
    def signed(v):
        return f"{v:+.1f}".replace("-", "\N{MINUS SIGN}")

    out = ["| Copy | Reading | Columns left out | Acrophase, −/− minus +/+ | Each day | Dark share, +/+ and −/− "
           "| Mean, −/− ÷ +/+ | Amplitude, −/− ÷ +/+ |", "|---|---|---|---|---|---|---|---|"]
    for _, r in sm.iterrows():
        out.append(f"| {r['copy'].replace(' x ', ' × ')} | {r.reading} | {r.columns_left_out} "
                   f"| {signed(r.acrophase_ko_minus_wt_h)} h "
                   f"| {signed(r.daily_diff_min_h)} to {signed(r.daily_diff_max_h)} h | {r.dark_share_wt:.2f}, "
                   f"{r.dark_share_ko:.2f} | {r.mean_ratio_ko_wt:.2f} | {r.amplitude_ratio_ko_wt:.2f} |")
    return "\n".join(out)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def figure(read, days):
    fig = plt.figure(figsize=(7.5, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.6, wspace=0.3)
    for i, (_, name) in enumerate(GENOTYPES):
        ax = fig.add_subplot(gs[i, :])
        x, y = read[name]
        for k in range(7):
            ax.axvspan(24 * k, 24 * k + 12, color=SHADE, lw=0)
        ax.plot(x, y, color=COLOURS[name], lw=1.6)
        ax.set_xlim(0, 168)
        ax.set_ylim(0, 160)
        ax.set_xticks(range(0, 169, 24))
        ax.set_ylabel("activity (cm/s)", fontsize=8, color=MUTED)
        ax.set_title(f"{'AB'[i]}  {name}: hourly values read from Lopatina et al. 2014, Fig 4{'AB'[i]}", loc="left",
                     fontsize=9, color=INK)
        style(ax)
    ax.set_xlabel("hours from the first lights-off (grey: dark)", fontsize=8, color=MUTED)
    ax = fig.add_subplot(gs[2, 0])
    ax.axvspan(0, 12, color=SHADE, lw=0)
    for _, name in GENOTYPES:
        x, y = read[name]
        hourly = np.full(168, np.nan)  # hour k from the first lights-off; a series may start after hour 0
        keep = (x >= 0) & (x < 168)
        hourly[np.round(x[keep]).astype(int)] = y[keep]
        week = hourly.reshape(7, 24)
        prof = np.nanmean(week, axis=0)
        lo, hi = np.nanmin(week, axis=0), np.nanmax(week, axis=0)
        ax.fill_between(np.arange(24), lo, hi, color=COLOURS[name], alpha=0.15, lw=0)
        ax.plot(np.arange(24), prof, color=COLOURS[name], lw=2, label=name)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xlabel("hours from lights-off", fontsize=8, color=MUTED)
    ax.set_ylabel("activity (cm/s)", fontsize=8, color=MUTED)
    ax.set_title("C  Mean day (band: range over 7 days)", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax = fig.add_subplot(gs[2, 1])
    ax.axhspan(0, 12, color=SHADE, lw=0)
    for _, name in GENOTYPES:
        d = days[days.genotype == name].sort_values("day")
        ax.plot(d.day, d.acrophase_h_after_dark_onset, color=COLOURS[name], lw=2, marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 6))
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("day", fontsize=8, color=MUTED)
    ax.set_ylabel("acrophase (h from lights-off)", fontsize=8, color=MUTED)
    ax.set_title("D  24-h cosinor acrophase, day by day", loc="left", fontsize=9, color=INK)
    style(ax)
    fig.savefig(HERE / "results" / "cd157.png", dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
    print(json.dumps({"written": ["results/cd157.csv", "results/cd157_daily.csv", "results/cd157_summary.csv",
                              "results/cd157_misread.csv", "results/cd157.png"]}))
