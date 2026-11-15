"""S1-S5 Tables of Hearn & Lloyd-Owen (revised) as Excel workbooks, from the CSV files tables.py writes.

    python paper/si_tables.py TABLES_DIR OUT_DIR

Each workbook has the table (or, for S4 and S5, its parts) and a sheet of notes: the caption, what each column holds and
where the values come from in the repository.
"""
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FONT = "Arial"
HEAD = PatternFill("solid", fgColor="E7E6E6")

S = {
    "S1_Table": ("S1 Table. The papers of validation 3.",
                 "The 26 papers shortlisted for the activity-trace validation set, with each paper's figure, whether "
                 "its data were used, any rhythm analysis it reported, whether it was included and, if not, why not, "
                 "and the numbers of series analysed and of statements tested (testable and not).",
                 [("Papers", "s1_table_papers.csv")],
                 "validation/activity-traces/papers.csv, results/comparisons.csv and results/claims_checked.csv"),
    "S2_Table": ("S2 Table. Digitised and original series.",
                 "For each of the 29 series with deposited data: agreement of the values read from the figure with "
                 "the authors' series (r at the authors' times; mean absolute error as % of the series' range; the "
                 "shift in time, within 2 h, at which they match best and r there), rhythm measures from the figure "
                 "and from the data (dark-phase share; 24-h cosinor acrophase in h from t = 0 of the record, and its "
                 "difference; amplitude ratio, figure over data; Lomb-Scargle period), and the verdicts of the three "
                 "tests of rhythmicity (chi-square periodogram only for records of 60 h or more).",
                 [("Series", "s2_table_series.csv")],
                 "validation/activity-traces/results/comparisons.csv"),
    "S3_Table": ("S3 Table. Statements in the papers and their tests.",
                 "The 60 statements about the traces in the 18 papers, paraphrased, with where in the paper each is "
                 "made, the kind of test that decides it, what was found in the digitised trace and in the authors' "
                 "data, and whether each supports the statement. Statements of no significant difference cannot be "
                 "tested from a mean trace. Kinds: compare (one series higher or lower than another over the windows "
                 "named), order, ratio or value (a reported number, within the tolerance in claims.json), rhythmic "
                 "(chi-square periodogram, or Lomb-Scargle for records under 60 h), nocturnal (dark-phase mean above "
                 "light-phase mean), period, rel_drop (relative decrease).",
                 [("Statements", "s3_table_claims.csv")],
                 "validation/activity-traces/claims.json and results/claims_checked.csv"),
    "S4_Table": ("S4 Table. The CD157 figure in every reading.",
                 "Lopatina et al. (2014), Fig 4, read from PubMed Central's copy (630 x 597 px) and the journal's "
                 "copy (397 x 376 px), each seven ways. primary: the gridlines, the sides of the frame and the strip "
                 "of axis labels of panel B left out; each column's midpoint (the median of its trace pixels); "
                 "hourly values the mean of the columns within each hour from lights-off. as read: as primary, with "
                 "nothing left out (as in validation 3). bins on the hour: as primary, with each hour's bin centred on "
                 "the hour. nearest the hour: as primary, with the value at each whole hour taken as the median of "
                 "the columns within 0.23 h (the rule of validation 3 for figures without data, used in the first "
                 "revision). top, mean of pixels, bottom: as primary, with the top, the mean or the bottom of each "
                 "column's trace pixels in place of its midpoint. Measures: rhythm measures of each genotype for each "
                 "reading (activity in cm/s; acrophase in h after lights-off; ZT = acrophase + 12, mod 24). Summary: "
                 "the knockout against the wild type for each reading. Daily: the 24-h cosinor and dark-phase share "
                 "of each day. Values on grey: the columns whose midpoint, as read, lies on a grey or white pixel "
                 "rather than on the trace.",
                 [("Measures", "s4_table_cd157_measures.csv"), ("Summary", "s4_table_cd157_summary.csv"),
                  ("Daily", "s4_table_cd157_daily.csv"), ("Values on grey", "s4_table_cd157_on_grey.csv")],
                 "validation/activity-traces/cd157.py and results/cd157*.csv"),
    "S5_Table": ("S5 Table. Real records drawn as dense traces and read back.",
                 "Nine activity records (three mice's home-cage activity and three mice's wheel running in 5-min bins, "
                 "and the mean of each three, from Hirano et al. (2026); one mouse's activity per minute over two "
                 "days, from Hoekstra et al. (2019)), each drawn as Lopatina et al. (2014), Fig 4, is drawn, in plot "
                 "areas the size of its two copies and in two colours, and read back. Each pixel column gives four "
                 "values (top, midpoint, mean of pixels, bottom), averaged within each hour and compared with the "
                 "record's hourly means. Summary: medians, ranges and 90th percentiles over the 36 readings and the "
                 "112 pairs of 7-day records drawn alike. Readings: one row per record, size, colour and statistic "
                 "(level = mean read / true mean; acrophase of the 24-h cosinor in h from the start of the record; "
                 "dark share with lights-off taken as the start of the 12 h of the day holding the largest share of "
                 "the record's activity). Pairs: for two records drawn alike, the errors in the difference of their "
                 "acrophases, the ratio of their means (as a factor) and the difference of their dark shares.",
                 [("Summary", "s5_table_dense_summary.csv"), ("Readings", "s5_table_dense_readings.csv"),
                  ("Pairs", "s5_table_dense_pairs.csv")],
                 "validation/dense-traces/run.py and results/"),
}


def sheet(ws, df):
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
    for c in ws[1]:
        c.font = Font(name=FONT, bold=True, size=10)
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, col in enumerate(df.columns, start=1):
        longest = max([len(str(col))] + [len(str(v)) for v in df[col].tolist()])
        ws.column_dimensions[get_column_letter(i)].width = max(8, min(60, longest * 0.9 + 2))
    ws.freeze_panes = "A2"


def main(tables, out):
    tables, out = Path(tables), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (title, caption, parts, source) in S.items():
        wb = Workbook()
        wb.remove(wb.active)
        for sheet_name, csv in parts:
            df = pd.read_csv(tables / csv)
            sheet(wb.create_sheet(sheet_name), df)
        notes = wb.create_sheet("Notes")
        for r in ([title], [caption], [f"Source: github.com/comparativechrono/trace-digitiser, {source} "
                                      "(made by paper/tables.py and paper/si_tables.py)."],
                  ["Hearn TJ, Lloyd-Owen B. Recovering circadian activity data from published figures: a validated "
                   "digitiser that runs in the browser, applied to CD157 knockout mice."]):
            notes.append(r)
        notes.column_dimensions["A"].width = 120
        for row in notes.iter_rows():
            for c in row:
                c.font = Font(name=FONT, size=10, bold=(c.row == 1))
                c.alignment = Alignment(wrap_text=True, vertical="top")
        wb.save(out / f"{name}.xlsx")
        print(out / f"{name}.xlsx", [p[0] for p in parts])


if __name__ == "__main__":
    main(*sys.argv[1:])
