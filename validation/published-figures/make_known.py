"""Build known.json - the values each test figure shows - from the papers' own data in sources/.

    python validation/published-figures/make_known.py      (needs openpyxl, to read the .xlsx files)

known.json is kept in the repository, so that run.py needs nothing beyond trace-digitiser itself; run this only to
check it, or after changing how the values are taken from the sources. It holds, for each panel:

  points  {series: [[x, y, error bar or null], ...]}       error bar: its half-length (the paper's SD or SEM)
  curves  {series: {"x": [...], "y": [...]}}
  bars    [[label, value, error bar or null], ...]         left to right
  dots    {column: [value, ...]}                           the columns of a dot plot, left to right
"""

import csv
import json
import math
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"


def _csv(name):
    text = (SRC / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode()
    rows = list(csv.reader(text.strip().split("\n")))
    return rows[0], [[float(v) if v not in ("", "NaN") else math.nan for v in r] for r in rows[1:]]


def _sheet(name, sheet=None):
    wb = openpyxl.load_workbook(SRC / name, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    return [list(r) for r in ws.iter_rows(values_only=True)]


def points(pairs):
    return [[float(x), float(y), None if e is None else float(e)] for x, y, e in pairs]


# ------------------------------------------------------------------ Leypunskiy et al. (2017), eLife 6:e23539
def leypunskiy():
    out = {}
    _, rows = _csv("elife-23539-fig1-data2-v2.csv")  # day length, peak time, standard deviation
    out["leypunskiy-1C"] = {"in vivo": points((r[0], r[1], r[2]) for r in rows)}
    _, rows = _csv("elife-23539-fig2-data1-v2.csv")  # reaction set, day length, time (h), % P-KaiC
    out["leypunskiy-2B"] = {
        name: points((r[2], r[3], None) for r in rows if r[1] == day)
        for name, day in (("18:6", 18), ("12:12", 12), ("6:18", 6))
    }
    rows = _sheet("elife-23539-fig2-data2-v2.xlsx")[1:]  # trial, day length, peak time, error
    two_c = {}
    for trial, day, peak, err in rows:
        if trial is not None:
            two_c.setdefault(str(trial).strip(), []).append((day, peak, err))
    out["leypunskiy-2C"] = {k: points(v) for k, v in two_c.items()}
    prc = _sheet("elife-23539-fig5-data1-v2.xlsx", "PRC")[1:]
    # the figure's axis of pulse times runs 24 h behind the data's: 36 in the figure is 60 in the sheet
    out["leypunskiy-5A"] = {"12 h pulses": points((r[1] - 24, r[2], r[4]) for r in prc if r[0] is not None)}
    wedge = {}
    for r in _sheet("elife-23539-fig5-data1-v2.xlsx", "wedge")[1:]:
        if r[0] is not None:
            wedge.setdefault(f"pulse at {int(r[1]) - 24} h", []).append((r[0], r[2], r[4]))
    out["leypunskiy-5B"] = {k: points(v) for k, v in wedge.items()}
    ent = {}
    for r in _sheet("elife-23539-fig5-data1-v2.xlsx", "entrainment")[1:]:
        if r[0] is not None:
            ent.setdefault(f"T = {int(r[1])} h", []).append((r[0], r[2], r[4]))
    out["leypunskiy-5C"] = {k: points(v) for k, v in ent.items()}
    return out


# ------------------------------------------------------------------ Foley et al. (2019), eLife 8:e50063
FOLEY_PANELS = {
    # panel: (bar heights, genotypes left to right)
    "2C-25C": ("period", ["PsiRNAiGD/+ (25°C)", "PsiRNAiKK/+ (25°C)", "w1118 (25°C)", "TD2/+ (25°C)",
                          "TD2/PsiRNAiGD (25°C)", "TD2/PsiRNAiKK (25°C)"]),
    "2C-30C": ("period", ["TD2/+ (30°C)", "PsiRNAiGD/+ (30°C)", "TD2/PsiRNAiGD (30°C)", "PsiRNAiKK/+ (30°C)",
                          "TD2/PsiRNAiKK (30°C)"]),
    "2D": ("period", ["PD2/+", "PD2/PsiRNAiGD", "PD2/PsiRNAiKK"]),
    "2E": ("period", ["TD2/+; PdfGAL80/+", "TD2/PsiRNAiGD; PdfGAL80/+", "TD2/PsiRNAiKK; PdfGAL80/+"]),
    "2F-period": ("period", ["UAS-Psi/+", "TG4/+", "TG4/+; UAS-Psi/+"]),
    "2F-rhythmic": ("percent", ["UAS-Psi/+", "TG4/+", "TG4/+; UAS-Psi/+"]),
    "2G-period": ("period", ["UAS-Psi/+", "PG4/+", "PG4/+; UAS-Psi/+"]),
    "2G-rhythmic": ("percent", ["UAS-Psi/+", "PG4/+", "PG4/+; UAS-Psi/+"]),
    "2H-period": ("period", ["UAS-Psi/+", "TG4/+; PdfGAL80/+", "TG4/+; PdfGAL80/UAS-Psi"]),
    "2H-rhythmic": ("percent", ["UAS-Psi/+", "TG4/+; PdfGAL80/+", "TG4/+; PdfGAL80/UAS-Psi"]),
}
# The per-fly sheet gives this genotype a mean of 23.95 h (SEM 1.0) for its 13 rhythmic flies, while the paper's
# summary sheet (fig2-data2), and the bar in the figure, have 24.88 h (SEM 0.1895): one fly's period in the per-fly
# sheet is presumably mistyped. The summary sheet's values are used for that bar.
FOLEY_OVERRIDE = {("2H-period", "TG4/+; PdfGAL80/UAS-Psi"): (24.88, 0.1895)}


def foley():
    rows = _sheet("elife-50063-fig2-data1-v2.xlsx")[2:]  # genotype, R/A, period, power, width
    per = {}
    for g, rhythmic, period, *_ in rows:
        if g is None:
            continue
        d = per.setdefault(str(g).strip(), {"n": 0, "periods": []})
        d["n"] += 1
        if str(rhythmic).strip() == "R" and period is not None:
            d["periods"].append(float(period))
    stats = {}
    for g, d in per.items():
        p = d["periods"]
        if len(p) > 1:
            m = sum(p) / len(p)
            sem = (sum((v - m) ** 2 for v in p) / (len(p) - 1)) ** 0.5 / len(p) ** 0.5
            stats[g] = {"mean": m, "sem": sem, "percent": 100.0 * len(p) / d["n"]}
    out = {}
    for panel, (kind, genotypes) in FOLEY_PANELS.items():
        bars = []
        for g in genotypes:
            if (panel, g) in FOLEY_OVERRIDE:
                v, e = FOLEY_OVERRIDE[(panel, g)]
            elif kind == "period":
                v, e = stats[g]["mean"], stats[g]["sem"]
            else:
                v, e = stats[g]["percent"], None
            bars.append([g, v, e])
        out[f"foley-{panel}"] = bars
    return out


# ------------------------------------------------------------------ Pearl and Reed (1920), PNAS 6:275-288
def logistic(year):
    """Their equation (xviii): y = 2930.3009 / (exp(-0.0313395 x) + 0.014854), y in thousands, x in years from 1780.
    It gives their table 3's calculated populations to within the table's rounding (1,000)."""
    return 1000.0 * 2930.3009 / (math.exp(-0.0313395 * (year - 1780)) + 0.014854)


def pearl_reed():
    with open(SRC / "pearl-reed-1920-tables.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert abs(logistic(int(r["year"])) - int(r["calculated_population_table_3"])) <= 1000
    xs = [1790 + 0.5 * i for i in range(241)]
    return {
        "pearl-reed-3-dots": {"census": [[int(r["year"]), int(r["observed_population_table_1"]), None] for r in rows]},
        "pearl-reed-3-curve": {"logistic": {"x": xs, "y": [logistic(x) for x in xs]}},
    }


# ------------------------------------------------------------------ held out: Andreani et al. (2022), eLife 11:e74327
def andreani():
    rows = _sheet("elife-74327-fig1-data1-v2.xlsx")
    out = {}
    one_b = {"baseline": [], "during deprivation": [], "during rebound": []}
    for r in rows[12:36]:
        if r[0] is None:
            continue
        for k, col in (("baseline", 1), ("during deprivation", 2), ("during rebound", 3)):
            if len(r) > col and r[col] not in (None, ""):
                one_b[k].append((float(r[0]), float(r[col]), None))
    out["andreani-1B"] = {k: points(v) for k, v in one_b.items()}
    ctl = [(float(r[0]), float(r[1])) for r in rows[41:89] if r[0] is not None]
    fd = [(float(r[0]), float(r[3])) for r in rows[41:89] if r[0] is not None]
    out["andreani-1C"] = {"control": {"x": [a for a, _ in ctl], "y": [b for _, b in ctl]},
                          "FD": {"x": [a for a, _ in fd], "y": [b for _, b in fd]}}
    # The rows are labelled the wrong way round: the row headed "standard deviation" holds the standard errors the
    # figure draws (and the row headed "Standard error" the standard deviations).
    means = [float(v) for v in rows[130][1:25]]
    sems = [float(v) for v in rows[131][1:25]]
    out["andreani-1E"] = [[f"bar {i + 1}", m, e] for i, (m, e) in enumerate(zip(means, sems))]
    rows3 = _sheet("elife-74327-fig3-data1-v2.xlsx")
    starts = {"A": 4, "B": 52, "C": 81, "D": 128}
    cats = ("sleep lost", "baseline sleep", "sleep gain")
    for panel, i in starts.items():
        cols = {}
        while i < len(rows3) and not str(rows3[i][0] or "").startswith("Averag"):
            r = rows3[i]
            for k, cat in enumerate(cats):
                for t, col in (("first time", 1 + k), ("second time", 8 + k)):
                    if len(r) > col and isinstance(r[col], (int, float)):
                        cols.setdefault(f"{cat}, {t}", []).append(float(r[col]))
            i += 1
        out[f"andreani-3{panel}"] = {f"{c}, {t}": cols.get(f"{c}, {t}", [])
                                     for c in cats for t in ("first time", "second time")}
    return out


if __name__ == "__main__":
    known = {}
    for part in (leypunskiy(), foley(), pearl_reed(), andreani()):
        known.update(part)
    (HERE / "known.json").write_text(json.dumps(known, indent=1, ensure_ascii=False) + "\n")
    for k, v in known.items():
        parts = v.values() if isinstance(v, dict) else [v]
        print(f"{k:24s}", " + ".join(str(len(s["x"]) if isinstance(s, dict) else len(s)) for s in parts))
