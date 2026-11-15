"""Real activity records in 5- and 1-min bins, from the source data of two CC BY papers, for run.py.

    python validation/dense-traces/records.py [CACHE_DIR]

Downloads the two source-data files from PubMed Central's open-access bucket (once, into CACHE_DIR, by default
~/.cache/trace-digitiser-dense) and writes records.csv beside this script: record, t_h (hours from the start of the
window used), value. The windows are days of light-dark entrainment:

* Hirano et al. 2026 (PLoS Biol 24:e3003475, CC BY 4.0), S1 Data: sheet Fig1A, the locomotor activity of three mice
  in 5-min bins (the actograms of Fig 1A), days 2-8 of the record (24-192 h; the record starts part-way through
  day 1); and sheet FigS2A, the wheel-running activity of three mice in 5-min bins (S2 Fig A), days 8-14 (168-336 h;
  days 5-7 are missing from the file). The group mean of each three is added as a record of its own.
* Hoekstra et al. 2019 (eLife 8:e43400, CC BY 4.0), Figure 2-source data 1: sheet Fig2-A, the locomotor activity
  (LMA) of one mouse per minute over the two baseline days, BL1 and BL2 (ZT0-48).
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
SOURCES = {"hirano2026": f"{S3}/PMC13099087.1/pbio.3003475.s004.xlsx",
           "hoekstra2019": f"{S3}/PMC6379088.1/elife-43400-fig2-data1.xlsx"}


def fetch(url, cache):
    path = cache / url.rsplit("/", 1)[1]
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def hirano(path, sheet, kind, lo, hi):
    df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=4).iloc[:, :6]
    df.columns = ["day", "hr", "min", "m1", "m2", "m3"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["day", "hr", "min"])
    t = ((df.day - 1) * 24 + df.hr + df["min"] / 60).to_numpy()
    w = (t >= lo) & (t < hi)
    mice = {f"{kind} mouse {k}": df[f"m{k}"].to_numpy(float)[w] for k in (1, 2, 3)}
    out = dict(mice)
    out[f"{kind}, mean of 3"] = np.nanmean(np.vstack(list(mice.values())), axis=0)
    return {name: (t[w] - lo, v) for name, v in out.items()}


def hoekstra(path):
    df = pd.read_excel(path, sheet_name="Fig2-A", header=None, skiprows=5)
    bl1 = pd.to_numeric(df[2], errors="coerce").dropna().to_numpy(float)[:1440]
    bl2 = pd.to_numeric(df[8], errors="coerce").dropna().to_numpy(float)[:1440]
    return {"LMA per minute": (np.arange(2880) / 60.0, np.r_[bl1, bl2])}


def main(cache=None):
    cache = Path(cache) if cache else Path.home() / ".cache" / "trace-digitiser-dense"
    h = fetch(SOURCES["hirano2026"], cache)
    recs = {}
    recs.update(hirano(h, "Fig1A", "locomotor", 24, 192))
    recs.update(hirano(h, "FigS2A", "wheel", 168, 336))
    recs.update(hoekstra(fetch(SOURCES["hoekstra2019"], cache)))
    out = pd.concat([pd.DataFrame({"record": k, "t_h": t, "value": v}) for k, (t, v) in recs.items()])
    out.to_csv(HERE / "records.csv", index=False, float_format="%.6g")
    print(out.groupby("record", sort=False).agg(values=("value", "size"), hours=("t_h", "max")).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:2])
