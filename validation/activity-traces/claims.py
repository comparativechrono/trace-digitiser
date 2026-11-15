"""Check each claim the papers make about their traces against the digitised series (and, where the paper gives
its data, against the original series too).

    python claims.py CLAIMS.json CASES.json RESULTS_DIR

Reads RESULTS_DIR/sampled.csv and series_metrics.csv (from compare.py); writes RESULTS_DIR/claims_checked.csv.

A claim is one of:
  rhythmic   {panel, series, expect}: the primary test (chi-square periodogram for records of 60 h or more,
             otherwise Lomb-Scargle) finds a circadian rhythm, or does not
  nocturnal  {panel, series, expect}: more active in the dark phase than in the light phase
  period     {panel, series, value, tol_h}: the Lomb-Scargle period is within tol_h of value
  compare    {metric, a, b, op}: a's metric is greater (op ">") or smaller ("<") than b's
  order      {metric, panel, series: [highest, ..., lowest]}
  ratio      {metric, a, b, value, tol}: a / b lies within value +- tol
  value      {metric, a, value, scale, tol_rel}: a's metric times scale is within tol_rel of value
  rel_drop   {metric, a: [reference, treated], b: [reference, treated], op}: (ref - treated) / ref of a vs b
  untestable {reason}: a claim of no significant difference, which group means alone cannot test
a and b are {panel, series, windows (optional list of [from, to] in hours)}. Metrics: mean, dark_mean, light_mean,
night_minus_day (dark mean minus light mean), dark_share (share of activity in the dark phase), amplitude (24-h
cosinor), M10 (most active 10 h of the average day).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhythm  # noqa: E402


def time_of(cases, case, panel):
    c = cases[case]
    return {**c["time"], **c.get("panel_time", {}).get(panel, {})}


def series(sampled, case, panel, name, source, windows=None):
    s = sampled[(sampled.case == case) & (sampled.panel == panel) & (sampled.series == name)].sort_values("x")
    if s.empty:
        raise KeyError(f"no series {case} | {panel} | {name}")
    x, y = s["x"].to_numpy(float), s[source].to_numpy(float)
    if windows:
        keep = np.zeros(len(x), bool)
        for lo, hi in windows:
            keep |= (x >= lo) & (x < hi)
        x, y = x[keep], y[keep]
    ok = np.isfinite(y)
    return x[ok], y[ok]


def metric(cases, sampled, case, spec, name, source):
    tm = time_of(cases, case, spec["panel"])
    x, y = series(sampled, case, spec["panel"], spec["series"], source, spec.get("windows"))
    if len(y) == 0:
        return np.nan
    if name == "mean":
        return float(y.mean())
    if name in ("dark_mean", "light_mean", "night_minus_day", "dark_share"):
        dl = rhythm.dark_light(x, y, tm["dark_start_h"])
        if name == "night_minus_day":
            return dl["dark_mean"] - dl["light_mean"]
        if name == "dark_share":
            ds = tm["dark_start_h"]
            dark = np.zeros(len(x), bool)
            if isinstance(ds, (list, tuple)):
                for a, b in ds:
                    dark |= (x >= a) & (x < b)
            else:
                dark = ((x - ds) % 24) < 12
            return float(y[dark].sum() / y.sum()) if y.sum() > 0 else np.nan
        return dl[name]
    if name == "amplitude":
        return rhythm.cosinor(x, y, 24.0)["amplitude"]
    if name == "M10":
        _, _, dt = rhythm.regular(x, y)
        return rhythm.m10_l5(x, y, dt)["M10"]
    raise ValueError(name)


def rhythm_call(metrics, case, panel, name, source):
    m = metrics[(metrics.case == case) & (metrics.panel == panel) & (metrics.series == name)
                & (metrics.source == source)]
    if m.empty:
        return None, {}
    r = m.iloc[0]
    if bool(r.get("chi_applicable")):
        call, test = bool(r["rhythmic_chi"]), "chi-square periodogram"
    else:
        call, test = bool(r["rhythmic_ls"]), "Lomb-Scargle"
    info = {"test": test, "chi": r.get("rhythmic_chi"), "ls": r.get("rhythmic_ls"), "cos": r.get("rhythmic_cos"),
            "ls_period": r.get("ls_period"), "ls_fap": r.get("ls_fap"), "chi_excess": r.get("chi_excess")}
    return call, info


def check(cl, cases, sampled, metrics, source):
    """(holds: True/False/None, what was found: str)."""
    t = cl["type"]
    case = cl["case"]
    if t == "untestable":
        return None, ""
    if t == "rhythmic":
        call, info = rhythm_call(metrics, case, cl["panel"], cl["series"], source)
        if call is None:
            return None, "no series"
        found = (f"{info['test']}: {'rhythmic' if call else 'not rhythmic'} (LS period "
                 f"{info['ls_period']:.1f} h, FAP {info['ls_fap']:.3g})")
        return call == cl["expect"], found
    if t == "period":
        m = metrics[(metrics.case == case) & (metrics.panel == cl["panel"]) & (metrics.series == cl["series"])
                    & (metrics.source == source)]
        p = float(m.iloc[0]["ls_period"])
        return abs(p - cl["value"]) <= cl["tol_h"], f"Lomb-Scargle period {p:.2f} h"
    if t == "nocturnal":
        spec = {"panel": cl["panel"], "series": cl["series"], "windows": cl.get("windows")}
        d = metric(cases, sampled, case, spec, "dark_mean", source)
        li = metric(cases, sampled, case, spec, "light_mean", source)
        sh = metric(cases, sampled, case, spec, "dark_share", source)
        return (d > li) == cl["expect"], f"dark mean {d:.4g}, light mean {li:.4g} (dark share {sh:.2f})"
    if t == "compare":
        a = metric(cases, sampled, case, cl["a"], cl["metric"], source)
        b = metric(cases, sampled, case, cl["b"], cl["metric"], source)
        holds = a > b if cl["op"] == ">" else a < b
        return bool(holds), f"{cl['metric']}: {a:.4g} vs {b:.4g}"
    if t == "order":
        vals = [metric(cases, sampled, case, {"panel": cl["panel"], "series": s, "windows": cl.get("windows")},
                       cl["metric"], source) for s in cl["series"]]
        return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), \
            f"{cl['metric']}: " + " > ".join(f"{v:.4g}" for v in vals)
    if t == "ratio":
        a = metric(cases, sampled, case, cl["a"], cl["metric"], source)
        b = metric(cases, sampled, case, cl["b"], cl["metric"], source)
        r = a / b
        return abs(r - cl["value"]) <= cl["tol"], f"{cl['metric']} ratio {r:.2f}"
    if t == "value":
        a = metric(cases, sampled, case, cl["a"], cl["metric"], source) * cl.get("scale", 1)
        return abs(a - cl["value"]) <= cl["tol_rel"] * abs(cl["value"]), f"{a:.4g} (reported {cl['value']:.4g})"
    if t == "rel_drop":
        def drop(pair):
            ref = metric(cases, sampled, case, pair[0], cl["metric"], source)
            tr = metric(cases, sampled, case, pair[1], cl["metric"], source)
            return (ref - tr) / ref
        da, db = drop(cl["a"]), drop(cl["b"])
        holds = da > db if cl["op"] == ">" else da < db
        return bool(holds), f"relative drop {da:.2f} vs {db:.2f}"
    raise ValueError(t)


def main(claims_path, cases_path, res):
    claims = json.loads(Path(claims_path).read_text())
    cases = {c["id"]: c for c in json.loads(Path(cases_path).read_text())}
    res = Path(res)
    sampled = pd.read_csv(res / "sampled.csv")
    metrics = pd.read_csv(res / "series_metrics.csv")
    rows = []
    for cl in claims:
        c = cases[cl["case"]]
        row = {k: cl.get(k) for k in ("id", "pmcid", "case", "figure", "where", "claim", "type")}
        row["group"] = c["group"]
        row["tools"] = c["tools"]
        row["testable"] = cl["type"] != "untestable"
        row["why_untestable"] = cl.get("reason", "")
        row["supported_by_digitised"], row["digitised_found"] = check(cl, cases, sampled, metrics, "digitised")
        if c.get("truth"):
            row["supported_by_original"], row["original_found"] = check(cl, cases, sampled, metrics, "original")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(res / "claims_checked.csv", index=False)
    t = out[out.testable]
    print(f"{len(out)} claims, {len(t)} testable; supported by the digitised traces: "
          f"{int(t.supported_by_digitised.sum())}/{len(t)}")
    if "supported_by_original" in t:
        a = t[t.supported_by_original.notna()]
        same = (a.supported_by_original.astype(bool) == a.supported_by_digitised.astype(bool)).sum()
        print(f"with data: {len(a)} claims; supported by the original series: "
              f"{int(a.supported_by_original.astype(bool).sum())}; digitised and original agree on {int(same)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
