# trace-digitiser

Get the numbers back out of a figure when the figure is all that survives — on your own computer.

`trace-digitiser` takes an image of a figure (a scan of an old paper, a figure saved from a PDF),
finds the plot area, follows each trace, and writes the values to a CSV file. It reads line plots,
and also [points, bars, actograms, heat maps and polar plots](#other-kinds-of-figure). It is
ordinary image processing, written to be read and checked.

**Nothing is uploaded and no AI model is involved.** The only other program it runs is
[Tesseract](https://github.com/tesseract-ocr/tesseract), also on your machine and optional, to read
the numbers printed beside the axes. A figure that you are not free to pass to an online service
can therefore still be digitised. (Whether the values in a published figure may be *reused* is a
matter for the licence of the source and the law where you work; no tool can settle that for you.)

A worked example with pictures is in
[`examples/walkthrough.ipynb`](https://github.com/comparativechrono/trace-digitiser/blob/main/examples/walkthrough.ipynb).

## Install

```bash
pip install trace-digitiser
```

Python 3.10 or later. To have axis values read from the figure, also install Tesseract
(`sudo apt-get install tesseract-ocr`, `brew install tesseract`, or the
[Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)). Without it everything else
works, and you type in the axis limits yourself — which is the better practice anyway.

## Use

From the command line:

```bash
trace-digitiser figure.png --y-min 0 --y-max 300 --x-min 0 --x-max 7 --overlay -o results/
```

or from Python:

```python
from trace_digitiser import digitise

result = digitise("figure.png", y_min=0, y_max=300, x_min=0, x_max=7,
                  output_dir="results", save_overlay=True)

result.trace_data      # one row per pixel column of each trace
result.warnings        # anything you should check
result.overlay_path    # the digitised traces drawn over the figure
```

`y_min` and `y_max` are the values at the bottom and top axis lines, `x_min` and `x_max` those at
the left and right. Nothing else is needed for most figures: the plot areas (any number of panels)
and the traces (coloured, black or grey) are found automatically, and a scan that lies a few
degrees askew is straightened first.

Leave the limits out and the scale is read from the tick labels instead:

```bash
trace-digitiser figure.png --overlay -o results/
```

```
  Panels: 2, Traces: 2
    panel 1: y 0.31 to 299, read from tick labels (medium confidence); x not calibrated
  CHECK: Panel 1: the y-axis scale was read from 7 of 7 tick labels (300, 250, 200, 150, 100, 50, 0);
         the print is only 6 pixels high. Confidence in it is medium: check those values against
         the figure, or give y_min and y_max.
```

A scale read from labels always comes with a statement of how far to trust it and, unless that is
"high", of the labels it rests on. If too few labels can be read to agree on a scale, the axis is
left uncalibrated (values from 0 to 1) and you are told so: the aim is never to pass off a wrong
scale as a good one ([how well that is met](#how-accurate-is-it)).

## Other kinds of figure

Points, bars, stacked rows, heat maps and polar plots each have a reader of their own. They find
the plot areas and read the axes exactly as `digitise` does, take the same settings for them
(`y_min`, `y_ref`, `plot_area`, `exclude` …), and give a table with a row for each thing found:

| Figure | Python | Command line | One row for each |
|---|---|---|---|
| Scatter or dot plot, line with markers | `digitise_points` | `trace-digitiser points` | point, with its error bars |
| Bar chart | `digitise_bars` | `trace-digitiser bars` | bar: where it ends, and its error bar |
| Actogram, double-plotted record, any stack of rows | `digitise_rows` | `trace-digitiser rows` | bin (or column) of each row |
| Heat map with a colour bar | `digitise_heatmap` | `trace-digitiser heatmap` | cell, from its colour on the bar |
| Polar plot | `digitise_polar` | `trace-digitiser polar` | point: its angle and distance |

```python
from trace_digitiser import digitise_points, digitise_bars, digitise_rows, digitise_heatmap

digitise_points("forest.png", save_overlay=True)        # markers, joined by a line or not; error bars too
digitise_bars("bars.png", y_min=-25, y_max=0)           # the legend's patch is left out
digitise_rows("birthogram.png", row_values=range(1955, 2014), row_spacing_value=1.1,
              bins=24, dots="black")                    # each row's fill, month by month, and its dot
digitise_heatmap("ccf.png", rows=21)                    # the colour bar's labels are read too
```

Each returns a `FigureResult`: `result.data` (the table), `result.warnings`, `result.overlay()`
and the files written (`<image>_points.csv`, `<image>_points_provenance.json` …).

For a program with a graphical interface — the
[digitiser workbench](https://github.com/comparativechrono/digitiser-workbench), which runs all of
this in a web browser, where you can mark plot areas, ticks and colours with the mouse —
`describe_figure` reports what the automatic steps find (plot areas, the tick marks on each axis,
each axis's tick labels and what was read from them, the colours drawn, a colour bar, polar axes)
so that they can be shown and corrected, and `pixels_to_values` turns points picked by hand into
the figure's units. `trace-digitiser describe figure.png` prints the same as JSON.

## Check the result

1. **Look at the overlay** (`--overlay`). It shows the plot area that was found, the centre and
   envelope of each digitised trace drawn over the original, and the positions of the axis values.
   If the lines sit on the traces, the digitisation is right; if they do not, you can see where.
2. **Read the messages.** Everything the program is unsure of is printed as a `CHECK:` line
   (in Python: `result.warnings`, also raised as `DigitiserWarning`). A clean run prints none.
3. **Keep the provenance file.** `<name>_automated_provenance.json` records the version, the
   SHA-256 of the image, every setting, what was detected and all messages, so that a digitisation
   can be reported and repeated exactly.

## Output

| File | Contents |
|------|----------|
| `<name>_automated_digitised_trace_by_column.csv` | One row per pixel column per trace (below). |
| `<name>_automated_panel_metadata.csv` | Where each plot area is, its axis calibrations and how they were obtained. |
| `<name>_automated_digitised_summary_by_label.csv` | Means per labelled interval, when the x axis carries interval names (`1N`, `1D`, …). |
| `<name>_automated_provenance.json` | Everything needed to report and repeat the run. |
| `<name>_automated_overlay.png` | With `--overlay`: the result drawn over the figure. |
| `<name>_automated_straightened.png` | Only if the figure was tilted: the image turned upright, which is what pixel positions in the other files refer to. |

Columns of the trace file:

| Column | Meaning |
|--------|---------|
| `panel_id`, `trace_id` | Panels are numbered row by row from the top left; traces within a panel from 1. |
| `x_pixel`, `x_norm_0_to_1` | The pixel column, and its position between the left (0) and right (1) axis lines. |
| `x_value` | The x value, if the x axis is calibrated. |
| `center_estimate` | Middle of the trace's ink in this column (median of its pixels). |
| `mean_estimate` | Mean of the trace's pixels in this column. |
| `upper_envelope`, `lower_envelope` | Top and bottom of the trace's ink in this column. |
| `n_pixels`, `interpolated` | How many pixels the trace has in this column; `True` if it has none and the values were interpolated from its neighbours. |
| `scale_type` | Where the y values come from: `user_supplied`, `ocr_value`, `normalised_0_to_1`, … |
| `trace_median_rgb`, `trace_pixel_count` | The trace's colour and size, for identifying it. |

`result.resample(...)` and `--sampling-interval` put the trace on an even grid in x.

## What the values are — and are not

A digitiser recovers what was **drawn**, not what was measured.

* For an ordinary curve — a thin line with at most a few samples per pixel column —
  `center_estimate` is the plotted value, to within a fraction of a pixel.
* For a **dense record**, where many samples share each pixel column and the trace is drawn as a
  solid band (an activity record of several days in a small figure, for instance), the drawing
  shows only the *range* of the samples in each column. `upper_envelope` and `lower_envelope`
  recover that range; `center_estimate` is its middle. **None of them is the mean of the samples**,
  and for spiky data the middle of the range can be far from the mean. The timing of the rhythm
  survives; the level does not.

How much this matters for a particular kind of data and size of figure can be measured rather than
guessed: see [Test it on data you know](#test-it-on-data-you-know).

Positions cannot be better than the image. In a plot area 350 pixels wide showing seven days, one
pixel is half an hour.

## When something is not found

Each automatic step can be replaced by telling the program the answer. Pixel positions count from
the top-left corner of the image, starting at 0; any image viewer shows them.

| Problem | Option | Python |
|---------|--------|--------|
| The figure is tilted and was not straightened, or was turned when it should not have been | `--straighten DEGREES` (its anticlockwise tilt), `--straighten never` | `straighten=1.3`, `straighten=False` |
| The plot area is wrong or not found | `--plot-area LEFT,TOP,RIGHT,BOTTOM` (repeat per panel) | `plot_area=(l, t, r, b)` |
| The wrong number of panels | `--layout`, `--rows`, `--cols`, `--panels` | `layout_mode`, `expected_rows`, … |
| The axis continues beyond its last tick, or has no line at the top or right | `--y-ref ROW:VALUE` twice or more (also `--x-ref`) | `y_ref=[(row, value), …]` |
| A logarithmic axis, with limits given | `--y-scale log` | `y_scale="log"` |
| A line drawn against a second y axis, on the right | `--y-axis right` (with `--trace-colour`) | `y_axis="right"` |
| A trace is missed, or two are merged | `--trace-colour RRGGBB` (repeat per trace) | `trace_colours=["#578dd2", …]` |
| A legend or annotation is taken for a trace | `--exclude LEFT,TOP,RIGHT,BOTTOM` | `exclude=(l, t, r, b)` |
| A black trace is invented or ignored | `--achromatic never` / `always` | `achromatic="never"` |
| No Tesseract, or it should not be used | `--no-ocr` | `ocr=False` |

## How accurate is it?

The package carries its own tests against known answers, and you can rerun them.

**Traces.** `trace-digitiser --validate results/` draws 44 figures (57 traces) from known data —
different axis styles, twelve colours including black and grey, 60 to 300 dpi, JPEG compression,
noise and blur, line widths, multi-panel layouts, several traces per panel, a logarithmic axis,
jagged and dense records, tilted scans — digitises them, and compares. Errors are in percent of the
height of the plot. A trace counts as *recovered* if its panel and curve were found with no
spurious extra trace, x positions are within 1 % and the RMS error in y is within 2 %.

| | found | recovered | median error | time per figure¹ |
|---|---|---|---|---|
| **2.0, nothing supplied** | 57 of 57 | 55 | 0.13 % | 0.5 s |
| **2.0, axis limits supplied** | 57 of 57 | 55 | 0.09 % | 0.3 s |
| 1.2.0, panel layout supplied | 48 of 57 | 20 | 0.49 % | 20 s |
| 1.2.0, panel layout and axis limits supplied | 48 of 57 | 13 | 1.39 % | 0.3 s |

¹ On a two-core machine.

The two traces that 2.0 does not recover are dense records in low-resolution figures (errors of
2.3 % and 4.3 %), for the reason given [above](#what-the-values-are--and-are-not). Of the traces
that 1.2.0 found but did not recover, most came with a spurious second trace.

**Axis scales.** Reading a scale from tick labels is the one step that depends on OCR, and OCR of
small print is unreliable, so it is tested separately:
`python -m trace_digitiser.validation results/ --axes` reads 750 axes (whole numbers, decimals,
negative values, thousands, clock and angle steps, logarithmic; print from 5 to 24 pixels high;
three typefaces; JPEG compression and blur).

| Height of the labels | axes | read correctly | declined | read wrongly, with a warning | read wrongly, no warning |
|---|---|---|---|---|---|
| 8 pixels or more | 477 | 464 | 13 | 0 | 0 |
| under 8 pixels | 273 | 260 (all with a warning to check them) | 12 | 1 | 0 |

*Declined* means that no scale was offered. All of those at 8 pixels or more are logarithmic axes
whose raised exponents are too small to read, though the rest of the label is not. The outcome for small
print depends on how the text was rasterised: with the test figures drawn by matplotlib 3.7, which
renders small print more softly than current versions, 236 of 292 axes under 8 pixels were read
correctly, 46 declined and 10 read wrongly with a warning — and still none without.

Labels under 8 pixels high are never given high confidence, however well they agree, because print
that small is misread in consistent ways — the decimal point dropped from every label, say. If
your figure has such labels, enlarging the image does not help; typing in two numbers does.

**Published figures.** The tests above draw their own figures.
[`validation/published-figures`](https://github.com/comparativechrono/trace-digitiser/tree/main/validation/published-figures)
holds seven figures from four papers whose authors published the numbers behind them — markers with error bars, a
line over shaded bands, ten bar charts, dot plots, and a scanned page from 1920 — and reads them with nothing given
and with the help a user can give. With nothing given, 2.3.0 reads all 35 bars of the bar charts to within 0.3 % of
the axis, the 29 points of a scatter plot of four series to within 0.5 %, the 9 points of a phase-response curve whose
tick labels are set close together, and each of three line plots stacked with no gap between them to within 0.6 %;
with two ticks clicked on each axis and the dots' colour, it reads the census dots of the 1920 figure to within
0.9 %. In four dot plots of translucent dots that overlap, it finds 386 of the 497 dots compared once their colours
are clicked (2.2.0: 143). The figures of one of the papers were held out until 2.2.0 was made; 2.3.0 was made while
looking at them too, so they are a held-out test of 2.2.0 only. Tick labels that are slanted or hand-lettered, and
series of one colour told apart only by the shape of their markers, still need help; that folder's README has the
details, case by case.

**Activity traces.**
[`validation/activity-traces`](https://github.com/comparativechrono/trace-digitiser/tree/main/validation/activity-traces)
tests the use the package was written for: rhythm analysis of activity traces digitised from papers. It holds 57
traces of mouse and rat activity over days from 18 open-access papers found by a literature search. For the 29
series whose authors published the numbers behind them, the same rhythm analysis is run on the trace as read and on
the authors' series: the Lomb–Scargle periodogram, the 24-h cosinor and (on the 10 records long enough for it) the
chi-square periodogram reach the same verdict on every series, the share of activity in the dark phase agrees to
within 0.05 for 27, and the acrophase to within 1 h for 28. For all 18
papers, what the authors say about their traces is checked against them: 47 of 53 statements are borne out, and
where the authors' numbers can decide, the trace and the numbers agree on all 22.

**Dense traces.**
[`validation/dense-traces`](https://github.com/comparativechrono/trace-digitiser/tree/main/validation/dense-traces)
draws real activity records, kept every minute or every 5 minutes, as dense traces (a week across a few hundred
pixels, so that each pixel column is a stroke from the lowest to the highest sample within it) and reads them back.
Averaged within each hour, the midpoint of each column (`center_estimate`) puts the acrophase within 0.36 h of the
record's (median) and reads the level 1.19 times too high; the top of each column (`upper_envelope`) times the rhythm
as well but reads the level about twice too high. Every reading of a dense trace understates how much of the
activity falls in the dark phase, by 0.03–0.04 for the means of three mice and more for single animals.

### Test it on data you know

The best test of a digitisation is one on data like yours. `round_trip` draws a known series as a
figure of whatever size and style you choose, digitises the drawing, and reports the errors:

```python
from trace_digitiser.validation import round_trip

check = round_trip(t, y, "check/", xlim=(0, 7), ylim=(0, 300),
                   panel_px=(347, 137), dpi=72, linewidth=0.8, fmt="jpg")
check["metrics"]     # errors of centre and envelopes, correlation with the mean per column, …
check["overlay"]     # the picture to look at
```

[`examples/validate_with_known_data.py`](https://github.com/comparativechrono/trace-digitiser/blob/main/examples/validate_with_known_data.py)
does this for a week of minute-by-minute activity drawn at the size of a small journal figure.

## Limits

* The plot must have at least a left and a bottom axis line, drawn in black or grey. A figure
  that is askew is straightened automatically if the tilt is under five degrees; beyond that,
  or for a photograph taken at an angle (where the axes are no longer parallel), correct the
  image first.
* One trace per colour per panel. Two traces of the same colour and dashed or dotted lines are not
  separated or joined up. Markers are read by `digitise_points`.
* Where traces cross or overlap, the one drawn underneath is hidden and its values there are
  interpolated (`interpolated` is `True`). Markers that overlap heavily (the inner turns of a
  spiral on a polar plot, say) are separated by their size, and some may be missed or doubled:
  check the overlay.
* Axis scales are read only from numeric labels on linear or logarithmic axes (on the left, below,
  or for a colour bar or a second y axis, on the right); dates, month names and broken axes are
  not understood: give `x_ref` / `y_ref` for those. Limits typed in are assumed to be at the axis
  lines unless `y_ref` / `x_ref` say otherwise.
* Shaded bands are not digitised. A fill drawn with an outline of its own colour (matplotlib's
  `fill_between`) is measured to the middle of the outline; bars are measured to their edges, to
  the nearest pixel where the bars' edges are not antialiased.

## Development

```bash
git clone https://github.com/comparativechrono/trace-digitiser.git
cd trace-digitiser
pip install -e ".[dev]"
pytest                      # the tests that need Tesseract are skipped without it
ruff check src tests
python validation/published-figures/run.py   # figures from papers against their authors' data (minutes)
python validation/activity-traces/run.py      # rhythm analysis of digitised activity traces (a minute)
python validation/dense-traces/run.py         # real records drawn as dense traces and read back (seconds)
```

`src/trace_digitiser/` in the order the work is done:

| Module | Does |
|--------|------|
| `straighten.py` | Measures the tilt of a scan and turns it upright. |
| `frames.py`, `panel_detection.py` | Finds axis lines and assembles them into plot areas. |
| `axis_labels.py`, `calibration.py`, `x_calibration.py` | Locates and reads tick labels, fits the scale, rates its confidence; or takes the limits given. |
| `trace_detection.py` | Separates the traces from gridlines, axes and text. |
| `digitise.py` | Converts each trace to values, column by column. |
| `ocr.py`, `summarise.py` | Interval labels along the x axis, and means per interval. |
| `diagnostics.py`, `provenance.py`, `io.py` | Overlay, record of the run, files. |
| `validation.py`, `synthetic.py` | Figures drawn from known data, and the comparisons. |
| `_figure.py` | What every reader does first: the image, straightening, plot areas and axes. |
| `markers.py`, `points.py` | Markers and their error bars. |
| `bars.py`, `rows.py`, `heatmap.py`, `polar.py` | Bar charts, stacks of rows, heat maps and polar plots. |
| `explore.py` | `describe_figure` and `pixels_to_values`, for graphical tools. |
| `_results.py` | What the readers of points, bars, rows, heat maps and polar plots give back. |
| `cli.py` | The command line. |

Changes are listed in [CHANGELOG.md](https://github.com/comparativechrono/trace-digitiser/blob/main/CHANGELOG.md);
making a release is described in
[RELEASING.md](https://github.com/comparativechrono/trace-digitiser/blob/main/RELEASING.md).

## Licence

MIT; see `LICENSE`. The example figure has its own licence and attribution, given in
[`examples/README.md`](https://github.com/comparativechrono/trace-digitiser/blob/main/examples/README.md).
