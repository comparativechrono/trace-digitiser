"""The series behind each plotted trace, from each paper's own data: one function per paper.

    python truth.py SOURCES_DIR OUT_DIR [PMCID ...]

Each writes OUT_DIR/<PMCID>.csv with columns panel, series, x (in the figure's x units), y (as plotted: the
group mean where the figure shows a mean), sem, n.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _long(rows):
    return pd.DataFrame(rows, columns=["panel", "series", "x", "y", "sem", "n"])


def _mean_sem(block):
    v = block.astype(float)
    n = v.notna().sum(axis=1)
    m = v.mean(axis=1, skipna=True)
    s = v.std(axis=1, ddof=1, skipna=True) / np.sqrt(n.where(n > 0))
    return m, s, n


def pmc12370043(d):
    """PLoS One 2025 (Slc43a3), S1 Data: hourly counts of each mouse; the figure plots the mean of each group at
    hours 1-60."""
    f = d / "PMC12370043" / "pone.0323853.s001.xlsx"
    rows = []
    for sheet, label in (("X-activity", "X-activity"), ("Ambulatory", "Ambulatory"), ("Z-activity", "Z-activity")):
        x = pd.read_excel(f, sheet_name=sheet, header=None)
        head = list(x.iloc[2])
        body = x.iloc[3:63]
        hours = body.iloc[:, 0].astype(float).to_numpy()
        for sex in ("Male", "Female"):
            for g in ("WT", "KO"):
                cols = [i for i, h in enumerate(head) if h == f"{sex} {g}"]
                m, s, n = _mean_sem(body.iloc[:, cols])
                for h, a, b, c in zip(hours, m, s, n):
                    rows.append((f"{label} {sex.lower()}", g, h, a, b, int(c)))
    return _long(rows)


def _zt_columns(head, day_row):
    """Continuous Zeitgeber time of each column headed ZTn under Day 1, Day 2, ...: day k adds 24 (k - 1) h."""
    xs, day = {}, 0
    for i, h in enumerate(head):
        if isinstance(day_row[i], str) and day_row[i].startswith("Day"):
            day = int(day_row[i].split()[1])
        if isinstance(h, str) and h.startswith("ZT"):
            xs[i] = int(h[2:]) + 24 * (day - 1)
    return xs


def _grouped(x, first, last, panel):
    """Rows first..last of a sheet whose column 0 names the group on its first animal's row."""
    head, day_row = list(x.iloc[2]), list(x.iloc[1])
    cols = _zt_columns(head, day_row)
    body = x.iloc[first:last + 1].copy()
    body[0] = body[0].ffill()
    rows = []
    for g, b in body.groupby(0, sort=False):
        m, s, n = _mean_sem(b[list(cols)].T)
        for i, (a, e, k) in enumerate(zip(m, s, n)):
            rows.append((panel, str(g), cols[list(cols)[i]], a, e, int(k)))
    return rows


def pmc10082796(d):
    """Nat Commun 2023 (eNAMPT), Source data: hourly locomotor activity (counts/h) of each mouse; the figure plots
    the group mean against Zeitgeber time running on across days (Fig 2b from ZT13 of day 1, Fig 2f from ZT3)."""
    f = d / "PMC10082796" / "Source data (sheets Figure 2b and Figure 2f).xlsx"
    b = pd.read_excel(f, sheet_name="Figure 2b", header=None)
    ff = pd.read_excel(f, sheet_name="Figure 2f", header=None)
    return _long(_grouped(b, 3, 16, "b LMA") + _grouped(ff, 3, 8, "f LMA"))


def pmc6379088(d):
    """eLife 2019 (CIRBP), Figure 6-source data 1: locomotor activity counts of each mouse in each hour of the 96-h
    experiment (rows at 0.5, 1.5, ... h); the figure plots the mean of each genotype."""
    f = d / "PMC6379088" / "elife-43400-fig6-data1.xlsx"
    x = pd.read_excel(f, sheet_name="Fig6-A LMA", header=None)
    head = list(x.iloc[3])
    body = x.iloc[4:100]
    hours = body.iloc[:, 0].astype(float).to_numpy()
    rows = []
    for g in ("WT", "KO"):
        cols = [i for i, h in enumerate(head) if isinstance(h, str) and h.startswith(("_1" if g == "WT" else "_2") + g)]
        m, s, n = _mean_sem(body.iloc[:, cols])
        rows += [("A LMA", g, h, a, b, int(c)) for h, a, b, c in zip(hours, m, s, n)]
    return _long(rows)


def pmc8257250(d):
    """eLife 2021 (TDCA/valine), Figure 4-source data 1: the indirect-calorimetry records (about every 28 min) of
    each mouse, as entered in CalR, with CalR's group settings beside them (Control: Subjects 1-6; Treatment: 7-12;
    lights on 7:00-19:00). Fig 4D is CalR's hourly plot of total beam breaks (xytot): each mouse's mean in each
    clock hour, then the group mean, over 120 h from lights on (07:00) on the second day."""
    import io
    f = d / "PMC8257250" / "elife-62928-fig4-data1.xlsx"
    x = pd.read_excel(f, sheet_name="Tabelle1", header=None)
    lines = [str(v) for v in x.iloc[3:, 2] if isinstance(v, str)]
    r = pd.read_csv(io.StringIO("\n".join(lines)), na_values=["NA"])
    r = r.dropna(subset=["hour", "xytot"])
    start = pd.Timestamp("2016-07-13 07:00:00")
    r["x"] = (pd.to_datetime(r["hour"]) - start).dt.total_seconds() / 3600
    groups = {"Control": [f"Subject {i}" for i in range(1, 7)], "TDCA/valine": [f"Subject {i}" for i in range(7, 13)]}
    # CalR's settings in the file switch outlier removal on; CalR removes values more than 3 SD from the mean of
    # their group in their photoperiod (lights on 7:00-19:00). Applied here to the records, light and dark pooled
    # over the run. (One control mouse's records reach 40000 beam breaks at 108-114 h; the figure does not show them.)
    r["grp"] = np.where(r["subject.id"].isin(groups["Control"]), "Control", "TDCA/valine")
    r["dark"] = (r["x"] % 24) >= 12
    st = r.groupby(["grp", "dark"])["xytot"].agg(["mean", "std"]).reset_index()
    r = r.merge(st, on=["grp", "dark"], how="left")
    r = r[(r["xytot"] - r["mean"]).abs() <= 3 * r["std"]]
    r = r[(r["x"] >= 0) & (r["x"] < 120)]
    per = r.groupby(["subject.id", "x"])["xytot"].mean().unstack(0)
    rows = []
    for g, subs in groups.items():
        m, s, n = _mean_sem(per[subs])
        rows += [("D LMA", g, float(h), a, b, int(c)) for h, a, b, c in zip(per.index, m, s, n)]
    return _long(rows)


def pmc6290015(d):
    """Commun Biol 2018 (social isolation), Supplementary Data 1, sheet Fig.4b: activity (cm/min) of each of 12 mice
    per group in each clock hour of days 2-4 (each day's block runs 8:00 to 7:00); the figure plots the group mean."""
    f = d / "PMC6290015" / "42003_2018_213_MOESM2_ESM.xlsx"
    x = pd.read_excel(f, sheet_name="Fig.4b", header=None)
    rows = []
    for g, first in (("GH", 5), ("SI", 81)):
        body = x.iloc[first:first + 72]
        hours = 8 + np.arange(72)
        assert list(body.iloc[:, 1].astype(int)) == [h % 24 for h in hours]
        m, s, n = _mean_sem(body.iloc[:, 2:14])
        rows += [("b activity", g, float(h), a, b, int(c)) for h, a, b, c in zip(hours, m, s, n)]
    return _long(rows)


def pmc11741586(d):
    """PLoS One 2025 (oxycodone withdrawal, rats), S1 Data Fig3_Data.xlsx, sheet 3B - Activity: activity counts
    (thousands) of 11 rats in 3-h bins on baseline days B4-B6 and withdrawal days W2-W8. The figure lays the ten days
    end to end; x here is hours along that record, at the middle of each bin."""
    f = next((d / "PMC11741586").rglob("Fig3_Data.xlsx"))
    x = pd.read_excel(f, sheet_name="3B - Activity", header=None)
    head = list(x.iloc[1])
    body = x.iloc[3:14]
    days = ["B4", "B5", "B6", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
    rows = []
    for k, day in enumerate(days):
        for j in range(8):
            col = head.index(f"{day}_ZT{3 * j}-{3 * j + 3}_(Counts)")
            v = body.iloc[:, col].astype(float)
            n = int(v.notna().sum())
            rows.append(("B actigraphy", "baseline" if day.startswith("B") else "withdrawal", 24 * k + 3 * j + 1.5,
                         float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)), n))
    return _long(rows)


def pmc9465459(d):
    """Front Mol Neurosci 2022 (methamphetamine), Mendeley Data 10.17632/7zbggkpmdy.1 (CC BY 4.0), Table 2: the
    Promethion 5-min records of each cage as exported for CalR (Control: cages 1, 2, 3, 5; METH: 4, 6, 8, per the
    sheet name). Fig 2A is CalR's hourly plot of X+Y beam breaks: each mouse's mean 5-min record in each hour, then
    the group mean; x = hours from midnight on 25 Dec 2021."""
    f = d / "PMC9465459" / "Table 2-Metabolic cage data-1235 Control group---468 METH group.xlsx"
    x = pd.read_excel(f, sheet_name=0)
    t = pd.to_datetime(x["Date_Time_1"], format="%m/%d/%Y %H:%M:%S")
    hour = ((t - pd.Timestamp("2021-12-25")).dt.total_seconds() // 3600).astype(int)
    rows = []
    for g, cages in (("Control", (1, 2, 3, 5)), ("METH", (4, 6, 8))):
        per = pd.DataFrame({c: x[f"xbreak_{c}"] + x[f"ybreak_{c}"] for c in cages})
        per = per.groupby(hour.to_numpy()).mean()
        m, s, n = _mean_sem(per)
        rows += [("A locomotor", g, float(h), a, b, int(c)) for h, a, b, c in zip(per.index, m, s, n)]
    return _long(rows)


def pmc13720268(d):
    """Adv Sci 2026 (Ube3a), source data workbook, sheet Fig 2, block 'Fig 2n': locomotor counts of each mouse in
    each of 49 hours (9 con, 5 NKO); the figure plots the group mean at ZT0-48, one point per hour."""
    f = next((d / "PMC13720268").rglob("UBE3A Source data_excel raw data.xlsx"))
    x = pd.read_excel(f, sheet_name="Fig 2", header=None)
    start = int(x.index[x.iloc[:, 0].astype(str).str.startswith("Fig 2n")][0])
    body = x.iloc[start + 2:start + 2 + 49]
    rows = []
    for g, cols in (("Ube3a con", range(1, 10)), ("Ube3a NKO", range(10, 15))):
        m, s, n = _mean_sem(body.iloc[:, list(cols)])
        rows += [("n locomotor", g, float(h), a, b, int(c)) for h, a, b, c in zip(range(49), m, s, n)]
    return _long(rows)


FUNCS = {"PMC12370043": pmc12370043, "PMC10082796": pmc10082796, "PMC6379088": pmc6379088,
         "PMC8257250": pmc8257250, "PMC6290015": pmc6290015, "PMC11741586": pmc11741586,
         "PMC9465459": pmc9465459, "PMC13720268": pmc13720268}


def main(ddir, odir, ids):
    ddir, odir = Path(ddir), Path(odir)
    odir.mkdir(parents=True, exist_ok=True)
    for pmcid in ids or FUNCS:
        t = FUNCS[pmcid](ddir)
        t.to_csv(odir / f"{pmcid}.csv", index=False)
        print(pmcid, t.groupby(["panel", "series"]).size().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
