# Dense traces: real records drawn densely and read back

A week of activity recorded every minute or every few minutes, drawn as a line in a plot a few hundred pixels wide,
puts several samples in each pixel column. The line through them then makes each column a vertical stroke from the
lowest to the highest sample within it: the "bars" of a dense trace. Lopatina et al. (2014), Fig 4, the figure of the
paper's CD157 re-analysis ([`../activity-traces/cd157.py`](../activity-traces/cd157.py)), is drawn so: 18 min of
record in each pixel column of PubMed Central's copy, 29 min in the journal's. None of the traces in the
[activity-trace set](../activity-traces/README.md) whose authors deposited their data is dense (each of their points
spans several pixel columns), so that set cannot say how well a dense trace is read.

The line reader gives four values for each column: the median of the trace's pixels in it (the **midpoint** of the
stroke), their **mean**, and the **top** and **bottom** of the stroke. None of them is the mean of the samples, which
the drawing does not record; the mean lies somewhere within the stroke. The midpoint is never more than half a stroke
from it, the top and the bottom up to a whole stroke; and for activity, which comes in bursts, the mean lies nearer
the bottom. This folder tests the four on real records drawn as Lopatina et al.'s figure is drawn.

```bash
python validation/dense-traces/records.py    # downloads the two source files, writes records.csv (already here)
python validation/dense-traces/run.py        # draws, reads and scores: figures/, results/
```

## The records

| Record | Source | Bins | Window |
|---|---|---|---|
| locomotor mouse 1–3, and their mean | Hirano A et al. (2026) Circadian rhythms remain temperature compensated during a Q neuron-induced hibernation-like state in mice. *PLoS Biol* 24:e3003475. [10.1371/journal.pbio.3003475](https://doi.org/10.1371/journal.pbio.3003475), S1 Data, sheet Fig1A (home-cage activity, infrared sensors) | 5 min | days 2–8 (light–dark) |
| wheel mouse 1–3, and their mean | the same, sheet FigS2A (wheel running) | 5 min | days 8–14 (light–dark) |
| LMA per minute | Hoekstra MMB et al. (2019) Cold-inducible RNA-binding protein (CIRBP) adjusts clock-gene expression and REM-sleep recovery following sleep deprivation. *eLife* 8:e43400. [10.7554/eLife.43400](https://doi.org/10.7554/eLife.43400), Figure 2—source data 1, sheet Fig2-A (one mouse, locomotor activity) | 1 min | the two baseline days, ZT0–48 |

Both papers and their data are published under CC BY 4.0; `records.csv` holds the windows used. Each 7-day record has
2,016 values, the 2-day record 2,880.

## The test

Each record is drawn as Lopatina et al.'s figure is: a line through every sample, in a plot area the size of each copy
of that figure (7 days across 551 × 217 or 347 × 137 pixels; the 2-day record in proportion), with grey gridlines at
sixths of the axis, drawn at four times the size and reduced, and saved as a JPEG; once in blue and once in red
(Excel's first two series colours), the red read with the gridlines left out, as the CD157 knockout's pale trace is
read. That puts 3.6–5.8 samples in each column of the 5-min records and 18–29 in each column of the 1-min record.
The reader is given the plot area and the axis limits, so that only the reading of the columns is tested. Hourly values
are the mean of each statistic over the columns within each hour, leaving out columns the reader filled in from their
neighbours (except on the trace's floor), as `cd157.py` makes them, and are compared with the record's hourly means:
their correlation; the level (the mean of the hourly values read, divided by the record's); the 24-h cosinor
acrophase over the whole record; and the share of activity in the dark phase, with lights-off taken as the start of
the 12 h of the day holding the largest share of the record's activity (the clock times of the light–dark cycles are
not all given; the records' dark-phase shares are 0.68–0.97). For the CD157 comparison of two traces, the same is
done for the 112 pairs of 7-day records drawn alike (same size and colour): the error in the difference of their
acrophases, in the ratio of their means, and in the difference of their dark-phase shares.

## Results

36 readings (9 records, 2 sizes, 2 colours) for each statistic; medians, with the range or the 90th percentile.

| Column statistic | r | Level, read ÷ true | Acrophase error | Dark share, read − true | Two records: acrophase difference error | Two records: ratio of means error | Two records: dark-share difference error |
|---|---|---|---|---|---|---|---|
| top | 0.82 | 1.96 (1.46–7.34) | 0.33 h (90%: 1.2 h) | −0.099 (−0.21 to −0.02) | 0.42 h (90%: 1.3 h) | 25% (90%: 54%) | 0.033 (90%: 0.098) |
| midpoint | 0.86 | 1.19 (1.00–3.62) | 0.36 h (90%: 1.1 h) | −0.085 (−0.21 to −0.00) | 0.42 h (90%: 1.4 h) | 12% (90%: 33%) | 0.043 (90%: 0.085) |
| mean of pixels | 0.85 | 1.18 (1.00–3.66) | 0.37 h (90%: 1.4 h) | −0.087 (−0.21 to −0.00) | 0.41 h (90%: 1.4 h) | 12% (90%: 33%) | 0.043 (90%: 0.086) |
| bottom | 0.74 | 0.41 (0.22–0.67) | 0.58 h (90%: 2.7 h) | −0.064 (−0.50 to +0.09) | 0.92 h (90%: 2.7 h) | 22% (90%: 51%) | 0.096 (90%: 0.178) |

By kind of record (medians over the 4 readings of each record):

| Record | Samples per column | Column statistic | Level, read ÷ true | Dark share, read − true | Acrophase error |
|---|---|---|---|---|---|
| locomotor (one mouse) | 4.7 | top | 2.11 | −0.121 | 0.39 h |
| locomotor (one mouse) | 4.7 | midpoint | 1.22 | −0.085 | 0.44 h |
| locomotor (one mouse) | 4.7 | mean of pixels | 1.23 | −0.091 | 0.37 h |
| locomotor (one mouse) | 4.7 | bottom | 0.40 | +0.011 | 0.58 h |
| locomotor, mean of 3 | 4.7 | top | 1.62 | −0.076 | 0.20 h |
| locomotor, mean of 3 | 4.7 | midpoint | 1.10 | −0.039 | 0.24 h |
| locomotor, mean of 3 | 4.7 | mean of pixels | 1.10 | −0.040 | 0.24 h |
| locomotor, mean of 3 | 4.7 | bottom | 0.57 | +0.062 | 0.27 h |
| wheel (one mouse) | 4.7 | top | 1.99 | −0.094 | 0.51 h |
| wheel (one mouse) | 4.7 | midpoint | 1.17 | −0.086 | 0.45 h |
| wheel (one mouse) | 4.7 | mean of pixels | 1.16 | −0.087 | 0.44 h |
| wheel (one mouse) | 4.7 | bottom | 0.40 | −0.110 | 0.77 h |
| wheel, mean of 3 | 4.7 | top | 1.51 | −0.052 | 0.30 h |
| wheel, mean of 3 | 4.7 | midpoint | 1.05 | −0.032 | 0.36 h |
| wheel, mean of 3 | 4.7 | mean of pixels | 1.05 | −0.033 | 0.37 h |
| wheel, mean of 3 | 4.7 | bottom | 0.59 | +0.009 | 0.57 h |
| LMA per minute | 23.7 | top | 6.64 | −0.190 | 0.39 h |
| LMA per minute | 23.7 | midpoint | 3.23 | −0.202 | 0.40 h |
| LMA per minute | 23.7 | mean of pixels | 3.29 | −0.202 | 0.49 h |
| LMA per minute | 23.7 | bottom | 0.27 | −0.424 | 9.27 h |

* **Timing survives.** The top, the midpoint and the mean of the pixels all put the acrophase within 0.33–0.37 h of
  the record's (medians; 90% within 1.1–1.4 h), and the difference between two records' acrophases as closely. The bottom does
  worse: where activity falls to zero between bursts, the bottom of every column is zero.
* **Levels and ratios need the midpoint.** The midpoint reads the level 1.19 times too high (1.05–1.10 for the means
  of three mice), the top about twice too high, and the ratio of two records' means is twice as far out from the top
  as from the midpoint (25% against 12%). The mean of the pixels behaves as the midpoint does.
* **The day–night contrast is compressed.** The top, the midpoint and the mean of the pixels all put too little of the
  activity in the dark phase: the midpoint by 0.03–0.04 for the means of three mice, about 0.09 for single mice and
  about 0.20 for the single mouse drawn per minute, whose columns hold 18–29 samples each, nearly all of the record's
  range. In a dense trace a quiet hour with
  one burst draws nearly as tall a stroke as a busy hour.
* **The top is fragile.** It is set by a single pixel, so anything of a similar colour above the trace (a gridline,
  a label) becomes the top; the midpoint needs half the column's pixels to move. In the CD157 figure the grey of the
  gridlines and of the axis labels inside the plot lies within the reader's colour tolerance of the knockout's pale
  pink, and until they were left out they set the top of many of its columns while hardly moving the midpoint.

The midpoint is therefore the value `cd157.py` analyses, with the other three as a sensitivity analysis.

## Files

| File | What it holds |
|---|---|
| `records.py` → `records.csv` | the records (record, t_h from the start of the window, value) |
| `run.py` | draws, reads and scores |
| `figures/` | the 36 images read |
| `results/readings.csv` | one row per record, size, colour and statistic: r, level, acrophase and dark share, read and true |
| `results/pairs.csv` | the 112 pairs of 7-day records drawn alike, for each size, colour and statistic |
| `results/hourly.csv` | the hourly values, true and read |
| `results/summary.csv` | the first table above |

## What this does not show

The records are rodent activity from two laboratories, three of them group means of only three mice; a figure's own
data may be smoother (a mean of more animals) or burstier, and how much a dense trace compresses the day–night contrast
depends on that. The drawing follows the CD157 figure (a line through every sample, as a spreadsheet draws one); a
figure drawn as bars from the axis, or with the samples averaged before drawing, is read differently. The reader was
given the axis limits, so axis errors are not included (the activity-trace set measures those).
