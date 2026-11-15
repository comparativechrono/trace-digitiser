"""Tables of agreement from compare.py's output.

    python summarise.py RESULTS_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def crosstab(a, b):
    a, b = pd.Series(a, dtype="boolean"), pd.Series(b, dtype="boolean")
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(bool), b[ok].astype(bool)
    return {"n": int(ok.sum()), "both_rhythmic": int((a & b).sum()), "dig_only": int((a & ~b).sum()),
            "orig_only": int((~a & b).sum()), "neither": int((~a & ~b).sum()),
            "agreement": float((a == b).mean()) if ok.any() else np.nan}


def main(res):
    res = Path(res)
    c = pd.read_csv(res / "comparisons.csv")
    c = c[c["true_cos_p"].notna()] if "true_cos_p" in c else c.iloc[0:0]
    out = []
    for k, label in (("chi", "chi-square periodogram (records >= 60 h)"), ("ls", "Lomb-Scargle (FAP < 0.05)"),
                     ("cos", "cosinor at 24 h (zero-amplitude F test)")):
        out.append({"unit": "series", "test": label, **crosstab(c[f"dig_rhythmic_{k}"], c[f"true_rhythmic_{k}"])})
    d = pd.read_csv(res / "daily.csv")
    d = d[d["case"].isin(c["case"])]
    w = d.pivot_table(index=["case", "panel", "series", "day"], columns="source", values="rhythmic_cos",
                      aggfunc="first")
    if {"digitised", "original"} <= set(w.columns):
        out.append({"unit": "day", "test": "cosinor at 24 h, each whole day",
                    **crosstab(w["digitised"], w["original"])})
    t = pd.DataFrame(out)
    print(t.to_string(index=False))
    print()
    cont = {
        "r (digitised vs original values)": c["r"],
        "r after best time shift": c["r_aligned"],
        "best time shift (h)": c["best_lag_h"],
        "mean abs error (% of range)": c["mae_pct"],
        "acrophase difference (h)": c["acrophase_diff_h"],
        "cosinor amplitude ratio": c["amplitude_ratio"],
        "dark-phase share difference": c["dark_fraction_diff"],
        "Lomb-Scargle period difference (h)": c["period_diff_h"],
    }
    rows = [{"measure": k, "median": v.median(), "q1": v.quantile(0.25), "q3": v.quantile(0.75), "min": v.min(),
             "max": v.max()} for k, v in cont.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pr = pd.read_csv(res / "pairs.csv")
    pr = pr[pr["original_mean_1_minus_2"].notna()] if "original_mean_1_minus_2" in pr else pr.iloc[0:0]
    print()
    rows = []
    for k, label in (("dark_mean", "higher dark-phase mean"), ("light_mean", "higher light-phase mean"),
                     ("mean", "higher overall mean"), ("cos_amplitude", "larger cosinor amplitude"),
                     ("dark_fraction", "larger dark-phase share"), ("acrophase", "later acrophase")):
        a, b = pr[f"digitised_{k}_1_minus_2"], pr[f"original_{k}_1_minus_2"]
        rows.append({"which of two series has the": label, "pairs": len(pr),
                     "same": int((np.sign(a) == np.sign(b)).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    t.to_csv(res / "agreement_rhythmic_calls.csv", index=False)
    pd.DataFrame(rows).to_csv(res / "agreement_pairs.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
