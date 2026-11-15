"""The tables of Hearn & Lloyd-Owen (revised), worked out from the validation sets' results, as CSV files (the
manuscript's Tables 1-4) and as the rows of S1-S5 Tables.

    python paper/tables.py OUT_DIR --workbench PATH_TO_DIGITISER_WORKBENCH

Table 1 reads the figures of Hearn & Whitmore (2026) again, with the settings of the workbench's walkthroughs, and
compares them with the values each figure shows, with the workbench's own comparison (python/dwb/known.py); it needs
a checkout of digitiser-workbench. The other tables are read from validation/published-figures/results-2.2.0.json and
validation/activity-traces/results/ and validation/dense-traces/results/ (run validation/activity-traces/run.py,
cd157.py and validation/dense-traces/run.py first).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AT = ROOT / "validation" / "activity-traces"
PF = ROOT / "validation" / "published-figures"
DT = ROOT / "validation" / "dense-traces"
sys.path.insert(0, str(AT))

# --------------------------------------------------------------------------- Table 1: our own figures

# what a user of the workbench gives for each figure: the settings of its walkthrough (build/help/1*-walk-*.html,
# tests/browser/examples.js in digitiser-workbench)
OWN = [
    ("figure-1a", "Fig 1a: monthly births, 1955-2014 (blue line)", "lines",
     dict(trace_colours=["#4682b4"], exclude=[(120, 15, 262, 58), (455, 12, 882, 218)]),
     "the line's colour; the legend and the inset left out"),
    ("figure-1a", "Fig 1a: peak-conception month (orange line, right-hand axis labelled with months)", "lines",
     dict(trace_colours=["#ff8c00"], exclude=[(990, 15, 1296, 58)], y_ref=[(485.5, 6), (34.5, 13)]),
     "the line's colour; the legend left out; two ticks of the right-hand axis"),
    ("figure-1a", "Fig 1a: inset, monthly births 1972-1978 (green line)", "lines",
     dict(trace_colours=["#228b22"], plot_area=(543, 20, 876, 159), exclude=[(606, 26, 868, 68)],
          x_ref=[(558, 1972), (865, 1979)], y_ref=[(62, 60000), (118, 50000)]),
     "the inset's plot area; the line's colour; its label left out; two ticks on each axis"),
    ("figure-1b", "Fig 1b: 12 bars with error bars", "bars", dict(), "nothing"),
    ("figure-2", "Fig 2: double-plotted birthogram, 59 years by 24 months, with dots", "rows",
     dict(row_values=list(range(1955, 2014)), bins=24, dots="black", row_spacing_value=1.1),
     "the rows' years, 24 columns, the dots' colour, the rows' spacing"),
    ("figure-3a", "Fig 3a: two stacked spectra", "lines", dict(), "nothing"),
    ("figure-3b", "Fig 3b: polar plot, a dot for each year's peak phase", "polar",
     dict(colours=["#1f77b4"], r_ref=[(5, 1955), (250, 2005)]), "the dots' colour; two rings and their years"),
    ("figure-3c", "Fig 3c: forest plot, dots with horizontal confidence intervals", "points", dict(), "nothing"),
    ("figure-4", "Fig 4: six heat maps sharing one colour bar", "heatmap", dict(rows=21),
     "the number of rows (21)"),
    ("figure-s3", "S3 Fig: two panels of lines with dots", "points", dict(), "nothing"),
]


def own_figures(workbench):
    import trace_digitiser as td

    examples = Path(workbench) / "data" / "examples" / "hearn-whitmore-2026"
    os.environ["DWB_EXAMPLES"] = str(examples)
    # known.py on its own: the workbench's python/ folder also holds a stand-in for pytesseract
    spec = importlib.util.spec_from_file_location("known", Path(workbench) / "python" / "dwb" / "known.py")
    known = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(known)
    reader = {"lines": td.digitise, "points": td.digitise_points, "bars": td.digitise_bars, "rows": td.digitise_rows,
              "heatmap": td.digitise_heatmap, "polar": td.digitise_polar}
    rows = []
    for figure, what, kind, options, given in OWN:
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = reader[kind](str(examples / f"{figure}.png"), write_outputs=False, **options)
        data = r.trace_data if kind == "lines" else r.data
        c = known.compare(figure, data, kind)
        st = c["stats"]
        if figure == "figure-1a":
            label = next(k for k in st if ("blue" in k and "blue" in what) or ("orange" in k and "orange" in what)
                         or ("green" in k and "green" in what))
            s = st[label]
            compared, err, extra = f"{s['n']:,} pixel columns", f"{s['median_abs_pct']:.2f}", ""
        elif figure == "figure-3a":
            ns = [s["n"] for s in st.values()]
            compared = " + ".join(f"{n:,}" for n in ns) + " pixel columns"
            err = ", ".join(f"{s['median_abs_pct']:.2f}" for s in st.values())
            extra = "one value per panel"
        elif figure == "figure-3b":
            found = next(x for x in c["lines"] if "years found" in x)
            compared = found.split(";")[0].replace("years found", "years").strip()
            err = f"{st['angle']['median_abs_pct']:.2f} (angle), {st['radius']['median_abs_pct']:.2f} (radius)"
            extra = f"angle error {st['angle']['median_abs']:.2f} degrees"
        elif figure == "figure-3c":
            compared = f"{st['points']['n']} of 51 points, and their intervals"
            err = (f"{st['points']['median_abs_pct']:.2f} (points), "
                   f"{max(st['ci_low']['median_abs_pct'], st['ci_high']['median_abs_pct']):.2f} (interval ends)")
            extra = ""
        elif figure == "figure-4":
            per = [s["median_abs_pct"] for k, s in st.items() if k != "all"]
            compared = f"{st['all']['n']:,} cells"
            err = f"{st['all']['median_abs_pct']:.2f}"
            extra = f"{min(per):.2f}-{max(per):.2f} by map"
        elif figure == "figure-2":
            s = st["rows"]
            compared = f"{s['n']:,} cells"
            err = f"{s['median_abs_pct']:.2f}"
            dots = next((x for x in c["lines"] if x.startswith("dots:")), "")
            extra = dots.replace("dots: ", "dots: ").replace(" months agree on whether a dot is drawn there",
                                                                   " cells right")
        else:
            s = next(iter(st.values()))
            unit = {"figure-1b": "bars", "figure-s3": "points"}[figure]
            compared = f"{s['n']} {unit}"
            err = f"{s['median_abs_pct']:.2f}"
            extra = ""
        rows.append({"figure": what, "reader": kind, "given": given, "compared": compared,
                     "median_error_pct_of_axis": err, "note": extra, "seconds": round(time.time() - t, 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Table 2: published figures

def _pf(res, key):
    return res[key]


def _fmt_case(c):
    """One case's result, briefly."""
    d, s = c["details"], c["summary"]
    if isinstance(d, dict) and "error" in d:
        return d["error"].replace("not read", "not read")
    return s


PUBLISHED = [
    # (figure, what it shows, cases with nothing given, cases with help, the help)
    ("Leypunskiy et al. 2017, Fig 1C", "12 points with error bars, a fitted line and three other lines",
     ["leypunskiy-1C-nothing"], ["leypunskiy-1C-nothing"], "none needed"),
    ("Leypunskiy et al. 2017, Fig 2B", "three stacked line plots over shaded days and nights",
     ["leypunskiy-2B-nothing"], ["leypunskiy-2B-areas-colour"], "the plot areas, the line's colour"),
    ("Leypunskiy et al. 2017, Fig 2B", "the same plots' 102 square markers",
     [], ["leypunskiy-2B-points"], "the plot areas, the markers' colour"),
    ("Leypunskiy et al. 2017, Fig 2C", "four series of points with error bars and fitted lines",
     ["leypunskiy-2C-nothing"], ["leypunskiy-2C-nothing"], "none needed"),
    ("Leypunskiy et al. 2017, Fig 5A", "a phase-response curve: 9 points with error bars",
     ["leypunskiy-5A-nothing"], ["leypunskiy-5A-ticks"], "the plot area, two ticks on each axis"),
    ("Leypunskiy et al. 2017, Fig 5B", "two series of markers with error bars",
     ["leypunskiy-5B-nothing"], ["leypunskiy-5B-colour-size"], "the colour, the marker size"),
    ("Leypunskiy et al. 2017, Fig 5C", "five series in five colours",
     ["leypunskiy-5C-nothing"], ["leypunskiy-5C-colours"], "the colours"),
    ("Foley et al. 2019, Fig 2C-H", "ten bar charts, 35 bars filled, hatched or outlined, with error bars",
     ["foley-2C-25C-nothing", "foley-2C-30C-nothing", "foley-2D-nothing", "foley-2E-nothing",
      "foley-2F-period-nothing", "foley-2F-rhythmic-nothing", "foley-2G-period-nothing", "foley-2G-rhythmic-nothing",
      "foley-2H-period-nothing", "foley-2H-rhythmic-nothing"], [], "none needed"),
    ("Pearl & Reed 1920, Fig 3 (a scan)", "census populations as dots and a fitted logistic curve, on a printed grid",
     ["pearl-reed-3-dots-nothing", "pearl-reed-3-curve-nothing"],
     ["pearl-reed-3-dots-ticks-colour", "pearl-reed-3-curve-ticks-exclude"],
     "two ticks on each axis; the dots' colour; the printed equation left out"),
    ("Andreani et al. 2022, Fig 1B (held out)", "three series of markers on lines over shading, slanted labels",
     ["andreani-1B-area"], ["andreani-1B-area-ticks-colours"], "the plot area, two ticks on the x axis, the colours"),
    ("Andreani et al. 2022, Fig 1C (held out)", "two lines inside translucent bands of their own colour",
     ["andreani-1C-nothing"], [], "none tried"),
    ("Andreani et al. 2022, Fig 1E (held out)", "24 bars with error bars, a dashed line and an equation",
     ["andreani-1E-area"], ["andreani-1E-area-ticks"], "the plot area, two ticks"),
    ("Andreani et al. 2022, Fig 3A-D (held out)", "four dot plots, 540 dots, overlapping",
     ["andreani-3A-nothing", "andreani-3B-nothing", "andreani-3C-nothing", "andreani-3D-nothing"],
     ["andreani-3A-colours", "andreani-3B-colours", "andreani-3C-colours", "andreani-3D-colours"], "the colours"),
]


def _one(c):
    """One case's result in a few words: how many of the known values were read, and the median error (% of the
    axis); the repository's README for the published figures gives the worst errors too."""
    d = c["details"]
    if isinstance(d, dict) and "error" in d:
        return d["error"]
    if isinstance(d, list):  # one entry per stacked plot
        read = [e for e in d if "error" not in e]
        unread = len(d) - len(read)
        if read and "known" in read[0]:
            m, k = sum(e["matched"] for e in read), sum(e["known"] for e in d if "known" in e)
            k += 34 * unread if unread else 0
            ys = ", ".join(f"{e['y_median']:.2f}%" for e in read)
            out = f"{m}/{k} points; {ys}"
        else:
            out = "; ".join(f"LD {name} {v['median']:.2f}%" for e in read for name, v in e.items())
        return out + (f"; {unread} of {len(d)} plots' y axis not read" if unread else "")
    if c["reader"] == "bars":
        return f"{d['within']}/{d['known']} bars within 1%; {d['median']:.2f}%"
    if "known" in d:  # points or dots
        what = "dots" if "columns" in d else "points"
        extra = f" (+{d['extra']} extra)" if d.get("extra") else ""
        return f"{d['matched']}/{d['known']} {what}{extra}; {d['y_median']:.2f}%"
    # curves
    return "; ".join(f"{'curve' if name == 'logistic' else name} {v['median']:.2f}%" for name, v in d.items())


def _summarise(res, keys):
    """The result of one or more cases of the same figure, in a few words."""
    if not keys:
        return ""
    cs = [res[k] for k in keys]
    if len(cs) == 1:
        return _one(cs[0])
    if all(c["reader"] == "bars" for c in cs):
        within = sum(c["details"]["within"] for c in cs)
        known = sum(c["details"]["known"] for c in cs)
        meds = [c["details"]["median"] for c in cs]
        return f"{within}/{known} bars within 1%; {min(meds):.2f}-{max(meds):.2f}%"
    if all(c["reader"] == "points" and "columns" in c["details"] for c in cs):
        m = sum(c["details"]["matched"] for c in cs)
        k = sum(c["details"]["known"] for c in cs)
        ys = [c["details"]["y_median"] for c in cs]
        return f"{m}/{k} dots; {min(ys):.2f}-{max(ys):.2f}%"
    return "; ".join(dict.fromkeys(_one(c) for c in cs))


def published_figures():
    res = json.loads((PF / "results-2.2.0.json").read_text())["cases"]
    rows = []
    for figure, what, nothing, helped, help_given in PUBLISHED:
        rows.append({"figure": figure, "what": what, "nothing_given": _summarise(res, nothing) or "-",
                     "with_help": _summarise(res, helped) if helped and helped != nothing else "-",
                     "help": help_given})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Table 3: activity traces

def activity_traces():
    comp = pd.read_csv(AT / "results" / "comparisons.csv")
    data = comp[comp["true_cos_p"].notna()].copy()
    calls = pd.read_csv(AT / "results" / "agreement_rhythmic_calls.csv")
    pairs = pd.read_csv(AT / "results" / "agreement_pairs.csv")
    claims = pd.read_csv(AT / "results" / "claims_checked.csv")
    n = len(data)
    rows = []

    def add(what, result, note=""):
        rows.append({"what": what, "result": result, "note": note})

    add("Values: correlation with the authors' series (median r)", f"{data['r'].median():.2f}",
        f"{data['r_aligned'].median():.2f} with each figure's own shift in time allowed for")
    add("Values: mean absolute error (median, % of the series' range)", f"{data['mae_pct'].median():.1f}",
        f"range {data['mae_pct'].min():.1f}-{data['mae_pct'].max():.1f}")
    for _, c in calls.iterrows():
        agree = int(c.both_rhythmic + c.neither)
        what = {"chi-square periodogram (records >= 60 h)":
                "Rhythmic or not: chi-square periodogram (records of 60 h or more)",
                "Lomb-Scargle (FAP < 0.05)": "Rhythmic or not: Lomb-Scargle periodogram (FAP < 0.05)",
                "cosinor at 24 h (zero-amplitude F test)": "Rhythmic or not: 24-h cosinor (F test)",
                "cosinor at 24 h, each whole day": "Rhythmic or not: 24-h cosinor of each whole day"}[c.test]
        unit = "days" if c.unit == "day" else "series"
        note = (f"{int(c.dig_only)} rhythmic only as read, {int(c.orig_only)} only in the data"
                if c.dig_only or c.orig_only else "")
        add(what, f"{agree} of {int(c.n)} {unit} agree", note)
    long = data[data["true_span_h"] >= 60]
    pd_ = (long["dig_ls_period"] - long["true_ls_period"]).abs()
    add("Lomb-Scargle period, records of 60 h or more", f"{int((pd_ <= 0.5).sum())} of {len(long)} within 0.5 h",
        f"median absolute difference {pd_.median():.2f} h")
    dd = data["dark_fraction_diff"].abs()
    add("Share of activity in the dark phase", f"{int((dd <= 0.05).sum())} of {n} within 0.05",
        f"largest difference {dd.max():.2f}")
    ad = data["acrophase_diff_h"].abs()
    add("24-h cosinor acrophase", f"{int((ad <= 1).sum())} of {n} within 1 h", f"largest difference {ad.max():.1f} h")
    ar = data["amplitude_ratio"]
    add("24-h cosinor amplitude", f"{int(((ar >= 0.8) & (ar <= 1.25)).sum())} of {n} within 0.8-1.25 times",
        f"ratios {ar.min():.2f}-{ar.max():.2f}")
    for _, p in pairs.iterrows():
        add(f"Which of two series in a panel has the {p.iloc[0]}", f"{int(p.same)} of {int(p.pairs)} pairs agree")
    t = claims[claims.testable.astype(str) == "True"]
    wd = t[t.group == "data"]
    same = int((wd.supported_by_digitised.astype(str) == wd.supported_by_original.astype(str)).sum())
    both = int(((wd.supported_by_digitised.astype(str) == "True")
                & (wd.supported_by_original.astype(str) == "True")).sum())
    neither = int(((wd.supported_by_digitised.astype(str) == "False")
                   & (wd.supported_by_original.astype(str) == "False")).sum())
    add("Statements in the papers with data: same verdict from the trace as read and the data",
        f"{same} of {len(wd)}", f"{both} supported by both, {neither} by neither")
    nd = t[t.group == "nodata"]
    add("Statements in the papers without data: supported by the trace as read",
        f"{int((nd.supported_by_digitised.astype(str) == 'True').sum())} of {len(nd)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Table 4: CD157

MEASURES = [("mean", "Mean activity (cm/s)", 1), ("amplitude", "24-h cosinor amplitude (cm/s)", 1),
            ("acrophase_h_after_dark_onset", "24-h cosinor acrophase (h after lights-off)", 1),
            ("cosinor_r2", "24-h cosinor R²", 2), ("ls_period_h", "Lomb-Scargle period (h)", 1),
            ("chi_square_period_h", "Chi-square periodogram: period of the rhythm (h)", 0),
            ("dark_mean", "Mean in the dark phase (cm/s)", 1), ("light_mean", "Mean in the light phase (cm/s)", 1),
            ("dark_share", "Share of activity in the dark phase", 2), ("M10", "M10 (cm/s)", 1),
            ("L5", "L5 (cm/s)", 1), ("RA", "Relative amplitude", 2)]


def cd157_table():
    t = pd.read_csv(AT / "results" / "cd157.csv")
    t = t[t.reading == "primary"]
    copies = list(dict.fromkeys(t["copy"]))
    rows = []
    for col, label, nd in MEASURES:
        row = {"measure": label}
        for g in ("CD157+/+", "CD157-/-"):
            vals = [t[(t["copy"] == c) & (t.genotype == g)][col].iloc[0] for c in copies]
            digits = 0 if col == "chi_square_period_h" else nd
            row[g] = f"{vals[0]:.{digits}f} ({vals[1]:.{digits}f})"
        a = [t[(t["copy"] == c) & (t.genotype == "CD157+/+")][col].iloc[0] for c in copies]
        b = [t[(t["copy"] == c) & (t.genotype == "CD157-/-")][col].iloc[0] for c in copies]
        if col == "acrophase_h_after_dark_onset":
            diff = [((y - x + 12) % 24) - 12 for x, y in zip(a, b)]
            row["comparison"] = f"−/− earlier by {-diff[0]:.1f} h ({-diff[1]:.1f} h)"
        elif col == "dark_share":
            row["comparison"] = f"−/− higher by {b[0] - a[0]:.2f} ({b[1] - a[1]:.2f})"
        elif col in ("mean", "amplitude", "dark_mean", "light_mean", "M10", "L5"):
            row["comparison"] = f"−/− ÷ +/+ = {b[0] / a[0]:.2f} ({b[1] / a[1]:.2f})"
        else:
            row["comparison"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def cd157_sensitivity():
    s = pd.read_csv(AT / "results" / "cd157_summary.csv")
    return s


# --------------------------------------------------------------------------- S tables

def s1_papers():
    p = pd.read_csv(AT / "papers.csv")
    comp = pd.read_csv(AT / "results" / "comparisons.csv")
    claims = pd.read_csv(AT / "results" / "claims_checked.csv")
    series = comp.groupby("pmcid").size()
    testable = claims[claims.testable.astype(str) == "True"].groupby("pmcid").size()
    untestable = claims[claims.testable.astype(str) != "True"].groupby("pmcid").size()
    inc = p.decision == "included"
    for col, counts in (("series_analysed", series), ("statements_testable", testable),
                        ("statements_untestable", untestable)):
        p[col] = [str(int(counts.get(k, 0))) if i else "" for k, i in zip(p.pmcid, inc)]
    p["group"] = p["group"].map({"data": "with data", "nodata": "without data"}).fillna(p["group"])
    return p[["pmcid", "cite", "journal", "doi", "figure", "group", "rhythm_analysis", "data_used", "decision", "note",
              "series_analysed", "statements_testable", "statements_untestable"]]


def s2_series():
    comp = pd.read_csv(AT / "results" / "comparisons.csv")
    d = comp[comp["true_cos_p"].notna()].copy()

    def yn(c):
        return d[c].map({True: "yes", False: "no", "True": "yes", "False": "no"}).fillna("-")

    out = pd.DataFrame({
        "pmcid": d.pmcid, "panel": d.panel, "series": d.series, "condition": d.condition, "values": d.n_values,
        "r": d.r.round(2), "mean_abs_error_pct_range": d.mae_pct.round(1), "best_shift_h": d.best_lag_h.round(2),
        "r_at_best_shift": d.r_aligned.round(2),
        "dark_share_figure": d.dig_dark_fraction.round(3), "dark_share_data": d.true_dark_fraction.round(3),
        "acrophase_figure_h": d.dig_cos_acrophase.round(2), "acrophase_data_h": d.true_cos_acrophase.round(2),
        "acrophase_difference_h": d.acrophase_diff_h.round(2), "amplitude_ratio": d.amplitude_ratio.round(2),
        "ls_period_figure_h": d.dig_ls_period.round(2), "ls_period_data_h": d.true_ls_period.round(2),
        "chi_square_rhythmic_figure": yn("dig_rhythmic_chi"), "chi_square_rhythmic_data": yn("true_rhythmic_chi"),
        "ls_rhythmic_figure": yn("dig_rhythmic_ls"), "ls_rhythmic_data": yn("true_rhythmic_ls"),
        "cosinor_rhythmic_figure": yn("dig_rhythmic_cos"), "cosinor_rhythmic_data": yn("true_rhythmic_cos")})
    na = ~d.dig_chi_applicable.astype(bool)
    out.loc[na.values, ["chi_square_rhythmic_figure", "chi_square_rhythmic_data"]] = "record < 60 h"
    return out


def s3_claims():
    c = pd.read_csv(AT / "results" / "claims_checked.csv")
    p = pd.read_csv(AT / "papers.csv").set_index("pmcid")["cite"]
    out = pd.DataFrame({"paper": c.pmcid.map(p), "pmcid": c.pmcid, "figure": c.figure, "where": c["where"],
                        "statement (paraphrased)": c.claim, "kind of test": c.type,
                        "group": c.group.map({"data": "with data", "nodata": "without data"}),
                        "testable": c.testable.astype(str).map({"True": "yes", "False": "no"}),
                        "why not testable": c.why_untestable.fillna(""),
                        "from the figure": c.digitised_found.fillna(""),
                        "supported by the figure": c.supported_by_digitised.astype(str).map(
                            {"True": "yes", "False": "no"}).fillna("-"),
                        "from the data": c.original_found.fillna(""),
                        "supported by the data": c.supported_by_original.astype(str).map(
                            {"True": "yes", "False": "no"}).fillna("-")})
    out.loc[out.testable == "no", ["supported by the figure", "supported by the data"]] = "-"
    out.loc[out.group == "without data", "supported by the data"] = "-"
    return out


def s4_cd157():
    t = pd.read_csv(AT / "results" / "cd157.csv")
    s = pd.read_csv(AT / "results" / "cd157_summary.csv")
    d = pd.read_csv(AT / "results" / "cd157_daily.csv")
    m = pd.read_csv(AT / "results" / "cd157_misread.csv")
    return t, s, d, m


def s5_dense():
    """The dense-trace test: its summary, every reading and every pair."""
    sm = pd.read_csv(DT / "results" / "summary.csv")
    r = pd.read_csv(DT / "results" / "readings.csv")
    pr = pd.read_csv(DT / "results" / "pairs.csv")
    keep = ["record", "size", "colour", "statistic", "columns", "samples_per_column", "hours_read", "r", "level",
            "mean_true", "mean_read", "acrophase_true_h", "acrophase_read_h", "acrophase_error_h",
            "daily_acrophase_max_abs_error_h", "dark_start_h", "dark_share_true", "dark_share_read",
            "dark_share_error", "y_max", "colour_clicked", "image"]
    return sm, r[keep], pr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--workbench", help="a checkout of digitiser-workbench, for Table 1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.workbench:
        t1 = own_figures(a.workbench)
        t1.to_csv(out / "table1_own_figures.csv", index=False)
        print(t1.to_string())
    published_figures().to_csv(out / "table2_published_figures.csv", index=False)
    activity_traces().to_csv(out / "table3_activity_traces.csv", index=False)
    cd157_table().to_csv(out / "table4_cd157.csv", index=False)
    s1_papers().to_csv(out / "s1_table_papers.csv", index=False)
    s2_series().to_csv(out / "s2_table_series.csv", index=False)
    s3_claims().to_csv(out / "s3_table_claims.csv", index=False)
    t, s, d, m = s4_cd157()
    t.to_csv(out / "s4_table_cd157_measures.csv", index=False)
    s.to_csv(out / "s4_table_cd157_summary.csv", index=False)
    d.to_csv(out / "s4_table_cd157_daily.csv", index=False)
    m.to_csv(out / "s4_table_cd157_on_grey.csv", index=False)
    sm, r, pr = s5_dense()
    sm.to_csv(out / "s5_table_dense_summary.csv", index=False)
    r.to_csv(out / "s5_table_dense_readings.csv", index=False)
    pr.to_csv(out / "s5_table_dense_pairs.csv", index=False)
    for f in sorted(out.glob("table*.csv")):
        print(f"\n== {f.name}\n" + pd.read_csv(f).to_string())


if __name__ == "__main__":
    main()
