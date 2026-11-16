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
    "S1_Table": ("S1 Table. Papers considered for Validation 3.",
                 "The 26 papers shortlisted for the activity-trace validation, with the figure used from each paper, "
                 "whether the authors' data were used, any rhythm analysis reported, whether the paper was included "
                 "(and if not, why not), and the numbers of series analysed and of statements tested and not "
                 "testable.",
                 [("Papers", "s1_table_papers.csv")],
                 "validation/activity-traces/papers.csv, results/comparisons.csv and results/claims_checked.csv"),
    "S2_Table": ("S2 Table. Digitised and original series.",
                 "For each of the 29 series with deposited data: agreement between the digitised values and the "
                 "authors' series (Pearson r at the authors' time points; mean absolute error as a percentage of the "
                 "range of the series; the time shift, within 2 h, giving the best match, and r at that shift); "
                 "rhythm measures from the digitised trace and from the data (dark-phase share; 24-h cosinor "
                 "acrophase in hours from the start of the record, and the difference between the two; amplitude "
                 "ratio, digitised to original; Lomb-Scargle period); and the results of the three tests of "
                 "rhythmicity (the chi-square periodogram only for records of 60 h or more). The lines of Sayler and "
                 "Hammond (2025) are LOWESS fits drawn through markers at the hourly means: for these 12 series the "
                 "authors' series is the same fit to their hourly means (7 means in each window, the window with "
                 "which the figure's own markers best reproduce its lines), and both series are compared half way "
                 "between the hourly time points, where only the line is drawn, so that each record spans 59 h.",
                 [("Series", "s2_table_series.csv")],
                 "validation/activity-traces/results/comparisons.csv"),
    "S3_Table": ("S3 Table. Statements in the papers and their tests.",
                 "The 60 statements about the traces in the 18 papers, paraphrased, with their location in each "
                 "paper, the type of test used, the result for the digitised trace and for the authors' data, and "
                 "whether each supports the statement. Statements of no significant difference cannot be tested on "
                 "a mean trace. Types of test: compare (one series higher or lower than another over the stated "
                 "intervals), order, ratio or value (a reported number, within the tolerance given in claims.json), "
                 "rhythmic (chi-square periodogram, or Lomb-Scargle periodogram for records shorter than 60 h), "
                 "nocturnal (dark-phase mean higher than light-phase mean), period, and rel_drop (relative "
                 "decrease). For Sayler and Hammond (2025), whose lines are LOWESS fits, the results from the data "
                 "are from the same fit to the authors' hourly means (S2 Table).",
                 [("Statements", "s3_table_claims.csv")],
                 "validation/activity-traces/claims.json and results/claims_checked.csv"),
    "S4_Table": ("S4 Table. Analyses of the CD157 figure.",
                 "Fig 4 of Lopatina et al. (2014), digitised from the PubMed Central copy (630 x 597 px) and the "
                 "journal copy (397 x 376 px), each analysed in seven ways. Primary: the gridlines, the sides of the "
                 "frame and the band of axis labels excluded from panel B; the midpoint (median) of the trace pixels "
                 "in each pixel column; hourly values calculated as the mean of the columns within each hour from "
                 "lights-off. Without exclusions: as primary, but with no regions excluded (as in Validation 3). "
                 "Hourly bins centred on the hour: as primary, with each bin centred on a whole hour. Columns nearest "
                 "each hour: as primary, with the value at each whole hour taken as the median of the columns within "
                 "0.23 h (the rule used in Validation 3 for figures without data). Upper envelope, column mean and "
                 "lower envelope: as primary, with the highest trace pixel, the mean of the trace pixels or the "
                 "lowest trace pixel of each column in place of the midpoint. Measures: rhythm measures for each "
                 "genotype in each analysis (activity in cm/s; acrophase in hours after lights-off; ZT = acrophase "
                 "+ 12, modulo 24). Summary: knockout compared with wild type in each analysis. Daily: 24-h cosinor "
                 "and dark-phase share for each day. Values on grey: columns whose midpoint lies on a grey or white "
                 "pixel rather than on the trace when no regions are excluded.",
                 [("Measures", "s4_table_cd157_measures.csv"), ("Summary", "s4_table_cd157_summary.csv"),
                  ("Daily", "s4_table_cd157_daily.csv"), ("Values on grey", "s4_table_cd157_on_grey.csv")],
                 "validation/activity-traces/cd157.py and results/cd157*.csv"),
    "S5_Table": ("S5 Table. Validation 4: real records plotted as dense traces and digitised.",
                 "Nine activity records (home-cage activity and wheel running of three mice each in 5-min bins, and "
                 "the mean of each group of three, from Hirano et al. (2026); and the activity of one mouse in 1-min "
                 "bins over two days, from Hoekstra et al. (2019)), each plotted in the same way as Fig 4 of Lopatina "
                 "et al. (2014), at the sizes of its two copies and in two colours, and then digitised. Four values "
                 "were taken from each pixel column (upper envelope, midpoint, column mean and lower envelope), "
                 "averaged within each hour and compared with the hourly means of the record. Summary: medians, "
                 "ranges and 90th percentiles over the 36 images and over the 112 pairs of 7-day records plotted in "
                 "the same way. Readings: one row per record, plot size, colour and column value (level = mean of the "
                 "digitised values divided by the true mean; 24-h cosinor acrophase in hours from the start of the "
                 "record; dark-phase share, with lights-off taken as the start of the 12 h of the day containing the "
                 "largest share of the record's activity). Pairs: for two records plotted in the same way, the errors "
                 "in the difference between their acrophases, in the ratio of their means (as a factor) and in the "
                 "difference between their dark-phase shares.",
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
                   "browser-based digitiser and its application to CD157 knockout mice."]):
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
