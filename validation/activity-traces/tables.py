"""Print the README's results tables (markdown) from results/.

    python validation/activity-traces/tables.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def yn(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "yes" if bool(v) else "no"


def tick(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "✓" if bool(v) else "✗"


def main():
    papers = {r["pmcid"]: r for r in csv.DictReader(open(HERE / "papers.csv"))}
    cases = json.loads((HERE / "cases.json").read_text())
    comp = pd.read_csv(RES / "comparisons.csv")
    claims = pd.read_csv(RES / "claims_checked.csv")

    print("### Papers\n")
    print("| Paper | Figure | Data | Rhythm analysis in the paper | Series | Claims checked |")
    print("|---|---|---|---|---|---|")
    for c in cases:
        p = papers[c["pmcid"]]
        n = int((comp.case == c["id"]).sum())
        k = claims[claims.case == c["id"]]
        link = f"[{c['pmcid']}](https://doi.org/{p['doi']})"
        print(f"| {p['cite']}, *{p['journal']}* ({link}) "
              f"| {p['figure']} | {'yes' if c['group'] == 'data' else 'no'} | {p['rhythm_analysis']} | {n} "
              f"| {int(k.testable.sum())} (+{int((~k.testable).sum())} untestable) |")

    d = comp[comp["true_cos_p"].notna()]
    print("\n### Each series of the papers with data\n")
    print("| Figure | Series | r | Mean abs. error (% of range) | Best shift (h) | r at best shift "
          "| Dark share (figure / data) | Acrophase difference (h) | Amplitude ratio | Rhythmic: χ² / LS / cosinor "
          "(figure; data) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        calls = "/".join(yn(r[f"dig_rhythmic_{k}"]) if k != "chi" or r["dig_chi_applicable"] else "–"
                         for k in ("chi", "ls", "cos"))
        calls_t = "/".join(yn(r[f"true_rhythmic_{k}"]) if k != "chi" or r["true_chi_applicable"] else "–"
                           for k in ("chi", "ls", "cos"))
        p = papers[r.pmcid]
        print(f"| {p['cite']}, {r.panel} | {r.series} | {r.r:.2f} | {r.mae_pct:.1f} "
              f"| {r.best_lag_h:+.2f} "
              f"| {r.r_aligned:.2f} | {r.dig_dark_fraction:.2f} / {r.true_dark_fraction:.2f} "
              f"| {r.acrophase_diff_h:+.2f} | {r.amplitude_ratio:.2f} | {calls}; {calls_t} |")

    print("\n### Claims\n")
    print("| Claim | Paper, figure | Kind | From the figure | ✓ | From the data | ✓ |")
    print("|---|---|---|---|---|---|---|")
    for _, r in claims.iterrows():
        p = papers[r.pmcid]
        if not r.testable:
            print(f"| {r.claim} | {p['cite']}, {r.figure} | not testable: {r.why_untestable} "
                  "| | – | | – |")
            continue
        orig = r.get("original_found")
        orig = "" if not isinstance(orig, str) else orig
        print(f"| {r.claim} | {p['cite']}, {r.figure} | {r.type} | {r.digitised_found} "
              f"| {tick(r.supported_by_digitised)} | {orig} | {tick(r.get('supported_by_original'))} |")


if __name__ == "__main__":
    main()
