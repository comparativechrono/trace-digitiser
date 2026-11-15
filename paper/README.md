# The figures and tables of the paper

Hearn TJ, Lloyd-Owen B. *Recovering circadian activity data from published figures: a validated digitiser that
runs in the browser, applied to CD157 knockout mice* (revised for PLOS ONE). This folder draws the paper's figures and
works out its tables from the validation sets in [`../validation`](../validation), so that every number in the paper
can be traced to a file and a line of code.

| File | What it makes |
|---|---|
| `figures.py` | Fig 1 (reading a figure), Fig 2 (the activity-trace validation), Fig 3 (the CD157 re-analysis) and S1 Fig (the pale trace and the gridlines), as TIFF files at 300 dpi and PNG files to look at |
| `tables.py` | Tables 1–4 and the rows of S1–S4 Tables, as CSV files; Table 1 reads the figures of Hearn & Whitmore (2026) again and compares them with the values they show, with the workbench's own comparison |
| `si_tables.py` | S1–S4 Tables as Excel workbooks, from the CSV files `tables.py` writes |
| `workbench_shots.js` | the two screenshots of the workbench, `workbench-marked.png` (Fig 1A) and `workbench-both-colours.png` (S1 Fig B); how to take them is at the top of the file |

## Running it

From the repository, with trace-digitiser 2.2.0 installed and pandas, openpyxl, SciPy and matplotlib:

```bash
python validation/activity-traces/run.py --keep-readings   # the validation set's results
python validation/activity-traces/cd157.py                 # the CD157 re-analysis
python paper/figures.py paper/out
python paper/tables.py paper/out --workbench PATH/TO/digitiser-workbench
python paper/si_tables.py paper/out paper/out
```

`tables.py --workbench` needs a checkout of
[digitiser-workbench](https://github.com/comparativechrono/digitiser-workbench), whose
`data/examples/hearn-whitmore-2026` holds the figures of validation 1 and the values they show; without it, Table 1
is left out. The figures use Liberation Sans (metrically the same as Arial) where it is installed.

The figure of Lopatina et al. (2014) that Fig 1, Fig 3 and S1 Fig show is reproduced under its CC BY licence, as
are all the figures in `validation`; see each folder's README for the sources.
