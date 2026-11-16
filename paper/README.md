# The figures and tables of the paper

Hearn TJ, Lloyd-Owen B. *Recovering circadian activity data from published figures: a validated browser-based
digitiser and its application to CD157 knockout mice* (revised for PLOS ONE). This folder draws the paper's figures
and works out its tables from the validation sets in [`../validation`](../validation), so that every number in the
paper can be traced to a file and a line of code.

| File | What it makes |
|---|---|
| `figures.py` | Fig 1 (reading a figure), Fig 2 (the activity-trace validation), Fig 3 (the CD157 re-analysis) and S1–S6 Fig: each validation's figures read against the values they show (S1, Validation 1; S2 and S3, Validation 2; S4, Validation 3; S5, Validation 4) and the columns of the CD157 figure with the result of every analysis of it (S6), as TIFF files at 300 dpi and PNG files to look at |
| `tables.py` | Tables 1–4 and the rows of S1–S5 Tables, as CSV files; Table 1 reads the figures of Hearn & Whitmore (2026) again and compares them with the values they show, with the workbench's own comparison |
| `si_tables.py` | S1–S5 Tables as Excel workbooks, from the CSV files `tables.py` writes |
| `workbench_shots.js` | the screenshot of the workbench, `workbench-marked.png` (Fig 1A); how to take it is at the top of the file |

## Running it

From the repository, with trace-digitiser 2.3.0 installed and pandas, openpyxl, SciPy and matplotlib:

```bash
python validation/published-figures/run.py --digitised      # validation 2, keeping what each case read
python validation/activity-traces/run.py --keep-readings    # the validation set's results
python validation/activity-traces/lowess_window.py          # the LOWESS window of Sayler & Hammond's lines
python validation/activity-traces/cd157.py                  # the CD157 re-analysis
python validation/dense-traces/run.py                       # the dense-trace test (validation 4)
python paper/figures.py paper/out --workbench PATH/TO/digitiser-workbench
python paper/tables.py paper/out --workbench PATH/TO/digitiser-workbench
python paper/si_tables.py paper/out paper/out
```

`figures.py --workbench` and `tables.py --workbench` need a checkout of
[digitiser-workbench](https://github.com/comparativechrono/digitiser-workbench), whose
`data/examples/hearn-whitmore-2026` holds the figures of validation 1 and the values they show, and whose
`python/dwb` has the comparison (`known.py`) and the reading of narrow peaks at their tips (`lines.py`, the
workbench's *Peaks* option, used for Fig 3a); without it, S1 Fig and Table 1 are left out. The figures use Liberation Sans (metrically the same as Arial) where it is installed.

The figure of Lopatina et al. (2014) that Fig 1 and S6 Fig show is reproduced under its CC BY licence, as are all
the figures and records in `validation`; see each folder's README for the sources.
