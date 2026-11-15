"""Read every activity trace in this set, rebuild the series behind each from the papers' own data, run the same
rhythm analysis on both, and check what the papers say about their traces.

    python validation/activity-traces/run.py                  # everything (reading the figures takes a few minutes)
    python validation/activity-traces/run.py --keep-readings   # reuse digitised/ and redo only the analysis
    python validation/activity-traces/run.py --overlays DIR    # also save each figure with what was read drawn on it

Steps, each a script in this folder that can also be run on its own:
  digitise_case.py  each figure in configs/ -> digitised/<case>.csv
  truth.py          the papers' data in sources/ -> truth/<PMCID>.csv, the series each figure was drawn from
  compare.py        rhythm analysis of every series, digitised and original -> results/
  claims.py         each statement in claims.json checked against the digitised (and original) series
  summarise.py      agreement tables; summary_figure.py the summary figure
"""
import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import claims  # noqa: E402
import compare  # noqa: E402
import digitise_case  # noqa: E402
import summarise  # noqa: E402
import summary_figure  # noqa: E402
import truth  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep-readings", action="store_true", help="use the readings already in digitised/")
    ap.add_argument("--overlays", help="folder for the package's overlays and per-figure outputs")
    a = ap.parse_args()
    cases = json.loads((HERE / "cases.json").read_text())
    if not a.keep_readings:
        for c in cases:
            print(c["id"], file=sys.stderr)
            digitise_case.main(str(HERE / c["config"]), str(HERE / c["digitised"]), a.overlays)
        (HERE / "digitised" / "VERSION").write_text(f"trace-digitiser {version('trace-digitiser')}\n")
    truth.main(HERE / "sources", HERE / "truth", sorted({c["pmcid"] for c in cases if c.get("truth")}))
    compare.main(HERE / "cases.json", HERE, HERE / "results")
    claims.main(HERE / "claims.json", HERE / "cases.json", HERE / "results")
    summarise.main(HERE / "results")
    summary_figure.main(HERE / "results", HERE / "results" / "summary.png")


if __name__ == "__main__":
    main()
