# Changelog

All notable changes to trace-digitiser are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

## [2.3.0] - 2026-11-15

Read again with 2.2.0, some of the figures in the validation sets still came
out wrong, or not at all: dot plots of translucent dots that overlap, legends
and labels drawn inside plots, plots stacked with no gap between them, tick
labels set close together, a line inside a pale band of its own colour. This
release reads most of them: of the values in the figures of published papers
(`validation/published-figures`), 1,184 of 1,601 are read (2.2.0: 614). The
changes were made while looking at all of those figures, including the ones
of Andreani et al. (2022) that were held out until 2.2.0 was made, so their
results are no longer those of figures held out. Activity traces read as well
as before or better (median correlation with the authors' series 0.95; 2.2.0:
0.94), and dense traces give the same results as with 2.2.0.

### Added

- **Legends inside a plot are found and left out**
  (`trace_digitiser.legend`; `digitise(..., legend="auto")`, the default, or
  `legend="keep"`): two or more entries, each a symbol with its words to the
  right, one above another or side by side in a row. Their symbols and words
  are left out as user-given `exclude` boxes are, and a warning says where.
- **Words written inside a plot are taken out of its traces**: a panel's
  label in the trace's colour (such as "18:6") is found as a row of glyphs
  standing on a baseline and left out of the trace.
- **Stacked plots with no gap between them are separated**: lines as dark and
  as thick as the frame's sides that run right across it are noted, and where
  the frame's tick labels give no one scale and the plots between such lines
  each read as an axis of their own, they are taken as plots of their own.
  Leypunskiy et al. (2017) 2B with nothing given: all three plots, 0.2-0.6 %
  (2.2.0: no axis read).
- `validation/published-figures/run.py --digitised` keeps what each case read
  (`validation/published-figures/digitised`), and `paper/figures.py` draws
  the paper's supporting figures (S1-S6 Fig) from it and from the other
  validation sets.
- **A validation set of activity traces**
  ([`validation/activity-traces`](https://github.com/comparativechrono/trace-digitiser/tree/main/validation/activity-traces)):
  57 traces of mouse and rat activity over days, from the figures of 18
  open-access papers found by a search of Europe PMC, read with 2.2.0 as a
  workbench user would read them. For the 8 papers that published the numbers
  behind their figures, the same rhythm analysis is run on the trace as read
  and on the authors' series: the Lomb-Scargle periodogram and the 24-h
  cosinor reach the same verdict for all 29 series, the share of activity in
  the dark phase is within 0.05 for 27 of 29. For all 18, what the papers say
  about their traces is checked against them. Not part of the package.
- **A test of dense traces**
  ([`validation/dense-traces`](https://github.com/comparativechrono/trace-digitiser/tree/main/validation/dense-traces)):
  nine real activity records, kept every minute or every 5 minutes, drawn as
  a week across a few hundred pixels (so that each pixel column is a stroke
  from the lowest to the highest sample within it) and read back. Averaged
  within each hour, `center_estimate` (the midpoint of each column) puts the
  acrophase within 0.36 h of the record's (median) and reads the level 1.19
  times too high; `upper_envelope` times the rhythm as well but reads the
  level about twice too high. Every column statistic but the lower envelope
  understates the share of activity in the dark phase. Not part of the
  package.

### Fixed

- **Overlapping markers** are found by the edges of each that can be seen: a
  Hough transform for circles of one size, in which each pixel on an edge -
  of the blob, or, where the markers are translucent, between one layer and
  two - votes for a centre a marker's radius inwards. Markers mostly hidden
  are added where the blob is otherwise not covered; a marker partly hidden
  under dark ink (a mean's bar, an error bar) is taken to go on under it; the
  size of one marker is taken from the round blobs where most blobs are two
  or three markers. Dot plots of Andreani et al. (2022) 3A-D: 386 of the 497
  dots compared, with their colours given (2.2.0: 143), 348 with nothing
  given (117); the polar plot of Hearn & Whitmore (2026): 55 of 60 years (50).
- **Tick labels set close together, or run into an axis title**, are each
  looked for under (or beside) their own tick, where the ticks are at a
  regular step: Leypunskiy et al. 5A with nothing given, 9 of 9 points
  (2.2.0: x axis not read); Andreani et al. 1E with its plot area, 24 of 24
  bars (y axis not read).
- **Ticks are found on plots shaded with pale bands**, by their contrast with
  what they are drawn on.
- **A line drawn inside a pale band of its own colour** (a mean within its
  error, shaded) is followed by its strong colour, not by the band's middle:
  Andreani et al. 1C, FD 1.0 % (4.2 %).
- **Error bars** are judged against what lies round them, so that a pale band
  is not taken for a bar; a line through a marker that runs on, uncapped, to
  the edge of the plot, or aslant past where a bar would end, or on beyond a
  gap (a broken gridline), is not a bar.
- A label read as "04" counts as evidence that an axis's decimal points are
  being lost in small print, or in larger print only if every reading of it
  agrees.
- A lump of more than fifty markers' area is a region of colour, not markers
  (a figure with shaded bands took four minutes).

## [2.2.0] - 2026-11-15

Tested on the figures of published papers whose underlying numbers are
published too (Leypunskiy et al. 2017, eLife 6:e23539: markers with error
bars, a line over shaded bands; Foley et al. 2019, eLife 8:e50063: bar charts;
Pearl and Reed 1920, PNAS 6:275, a microfilm scan), 2.1.0 read what it found
accurately but often found nothing. This release finds much more of it, there
and on the figures of a paper held out until the changes were made (Andreani
et al. 2022, eLife 11:e74327). Line plots digitise as before: the package's
validation suite and the Hearn and Whitmore (2026) examples give the same
results.

### Added

- **A test set of published figures**
  ([`validation/published-figures`](https://github.com/comparativechrono/trace-digitiser/tree/main/validation/published-figures)):
  seven figures from those four papers, the numbers behind them from their
  authors' data, and a script that reads them with nothing given and with the
  help a user can give, and compares. Its README has the results of 2.1.0 and
  2.2.0, case by case, including the few that 2.2.0 reads worse. It is run
  before each release and is not part of the package.

### Fixed

- **Bar charts in the style of GraphPad Prism**: black, grey, hatched and
  outlined bars are found, on an axis that does not start at 0 (2.1.0 found 3
  of the 35 bars of Foley et al.'s figure 2, and read those upside down; all
  35 are now read, within 0.3 % of the axis). Bars are the columns filled from
  the baseline, so an error bar or a significance bracket joined to a bar no
  longer spoils it; a border of another colour (a black outline round a grey
  bar) is part of the bar; a bar cut by the edge of the plot area is kept.
- **Error bars drawn in the colour of their bar or marker** (the default in
  Prism, MATLAB and matplotlib) are followed beyond the bar's end, or from the
  marker, to their caps; one running across a fitted line goes on beyond it.
  Where only the half beyond a bar's end shows, a note says so.
- **Markers sitting on a line of their own colour** (a fitted line, their
  error bars) are found: other sizes are tried where the lines' ridges
  outnumber the markers, and such a colour counts as a colour of markers if
  its markers are joined to its lines and much thicker than them. Round
  letters of words written in the plot, and a legend's markers beside their
  words, are left out; a marker found in two series of close colours is kept
  in the nearer one.
- **A plot area given by hand** (a box drawn in the workbench, or numbers read
  off a picture) moves to the middle of the axis lines within a few pixels of
  it, and their thickness is measured: with a thick axis line the tick labels
  were looked for inside it, and the axis title was read instead.
- Long tick marks of a small plot in a large picture are found, so that its
  labels are anchored to them.
- A colour that fills areas (a shaded band behind a plot) is not taken for a
  trace of a line plot.
- Tick labels that Tesseract misreads as a line of text (75 as 15, in large
  Helvetica) are read once more as a single word when an axis is not settled.
- Looking at a whole page of a scanned paper (`describe_figure`, the
  workbench's first look at a page of a PDF) took a minute and a half, nearly
  all of it in the search for polar axes; the circles are now proposed on a
  smaller copy of the page (16 seconds in all).

## [2.1.0] - 2026-11-14

Everything in 2.0.0 works as before. What is new is readers for other kinds of
figure, and helpers for graphical tools such as the
[digitiser workbench](https://github.com/comparativechrono/digitiser-workbench).

### Added

- **Points**: `digitise_points` (`trace-digitiser points`) reads the markers of
  a scatter or dot plot, or of a line drawn with markers, whatever their
  shape, separating markers that overlap; and the error bars on them,
  vertical or horizontal, to the middle of their caps.
- **Bars**: `digitise_bars` (`trace-digitiser bars`) reads where each bar
  ends, upright or lying down, rising or hanging from the baseline, and its
  error bar; a legend's patch is left out.
- **Rows**: `digitise_rows` (`trace-digitiser rows`) reads figures drawn as a
  stack of rows - actograms, double-plotted records, birthograms - bin by bin,
  and the dots that mark a peak in them.
- **Heat maps**: `digitise_heatmap` (`trace-digitiser heatmap`) reads each
  cell's value from its colour on the colour bar, whose labels are read like
  an axis's (or from a matplotlib colour map).
- **Polar plots**: `digitise_polar` (`trace-digitiser polar`) finds the polar
  axes and their rings, and reads each point's angle and distance.
- All of them return a `FigureResult` and write a CSV file, a provenance file
  and, if asked, an overlay, as `digitise` does.
- `describe_figure` (`trace-digitiser describe`): what the automatic steps
  find in a figure - plot areas, each axis's tick labels and readings, the
  colours drawn, a colour bar, polar axes - as plain values.
- `pixels_to_values`: points picked by hand, in the figure's units.
- Tick labels are read on the right-hand side of a plot too (a colour bar, a
  second y axis): `calibrate_axis(..., "y2")`.
- `y_axis="right"` (`--y-axis right`) reads a line's values against a second
  y axis on the right of the plot (a twin axis); with `trace_colours`, the
  lines drawn against each axis are digitised in a run of their own.
- `describe_figure` gives the tick marks found on each plot's axes, for a
  program that lets you click on a tick.
- Colours can be given by name (`"black"`, `"steelblue"`, `"tab:blue"`).
- Plots stacked on a shared x axis, of which only the lowest has labels, take
  its x scale (with a message saying so).
- A plot area hidden among others in a grid (a heat map whose dark colours
  hide its axis lines) is found by its frame.

### Changed

- An axis scale is held to medium confidence when the tick labels support a
  different scale just as well. Tesseract compiled to WebAssembly (as it runs
  in a browser) read the superscript minus signs of a log axis's exponents
  differently from the Tesseract program, and two scales then fitted equally;
  between log scales that tie, the one that steps by a decade is chosen.
- The sides of filled areas (bars) no longer make a plot area of their own
  inside the real one, which bar charts with a legend could cause.
- A long axis (ten or more labels) whose labels nearly all agree is not read
  again in the slower ways; fitting the labels of a long axis (the 59 years of
  a birthogram) takes a second or two rather than half a minute, with the same
  result.
- A heat map's colours are looked up on the colour bar once each, rather than
  once for each pixel: Figure 4 of Hearn and Whitmore (2026) takes 4 seconds
  rather than 45, with the same values.
- pytesseract is imported only when it is first used.

## [2.0.0] - 2026-11-14

Every call and command-line option of 1.2.0 still works, with the same
meaning. The version number changes because the **results** change: figures
are analysed in a new way, and the values that come out differ from those of
1.2.0. By the package's own tests against known data they are much closer to
what was drawn (55 of 57 test traces recovered to within 2 % with nothing
supplied, against 20; see the README). Anything digitised with 1.x that
matters should be digitised again.

### Added

- **Plot areas are found without being told the layout.** Axis lines are
  located to a fraction of a pixel and assembled into plot areas: any number
  of panels, boxed or L-shaped axes, with or without gridlines and tick
  marks. `layout_mode`, `expected_rows` and the rest remain as hints and
  checks.
- **Tilted scans are straightened.** The tilt of the figure is measured and
  undone first (`straighten="auto"`, `--straighten`), so a page that lay
  askew on the scanner is digitised as accurately as one that did not.
- **Each automatic step can be replaced by the answer.** `plot_area` for the
  position of the axes, `y_ref` / `x_ref` for reference points on an axis,
  `y_scale` / `x_scale` for logarithmic axes, `trace_colours` for the
  colours to follow, `exclude` for legends and annotations, `achromatic` for
  black and grey traces, `ocr=False` to do without Tesseract.
- **The x axis can be calibrated** from `x_min` / `x_max`, from reference
  points, or from numeric tick labels (`x_value` in the output).
- **Axis scales read from tick labels come with a confidence rating**, and a
  scale that too few labels agree on is declined rather than guessed. The
  rating, and the labels the scale rests on, are in the panel metadata.
- **Messages about anything that needs checking**: `result.warnings`,
  `DigitiserWarning`, and `CHECK:` lines on the command line.
- **An overlay image** of the result drawn over the figure (`save_overlay`,
  `--overlay`, `result.overlay()`): the quickest way to see whether a
  digitisation is right.
- **A provenance file** for every run: version, SHA-256 of the image, all
  settings, what was detected, and all messages.
- **Validation tools** (`trace_digitiser.validation`): `round_trip` tests the
  digitiser on your own data drawn at the size of your figure;
  `run_validation` (`--validate`) and `run_axis_validation` test it on 44
  figures and 750 axes drawn from known data.
- New columns in the trace file: `n_pixels`, `interpolated`, `x_value`. New
  columns at the end of the panel metadata: the sub-pixel plot area and how
  each axis was calibrated. The 1.2.0 columns keep their names and order.
- Image arrays are accepted as well as file names; transparent images are
  placed on white, and photographs are turned according to their EXIF data.
- `--version`; exit status 1 if any image could not be processed.
- Examples: two scripts, a notebook, and a published figure (CC BY) that is
  also used in the tests.
- The repository itself: continuous testing on Python 3.10 to 3.14, and a
  release workflow that publishes to PyPI through trusted publishing (see
  `RELEASING.md`).

### Changed

- **Values differ from 1.2.0**, chiefly because:
  - the plot area is now bounded by the centres of the axis lines; 1.2.0 put
    its edges up to 13 pixels outside them when tick marks pointed outwards,
    which shifted and rescaled every value read with `y_min` / `y_max`;
  - traces are followed down onto the axis lines when their colour allows,
    and thin, pale, grey and black traces are no longer missed;
  - gridlines, text and compression fringes are no longer reported as
    extra traces (1.2.0 did so in about half of single-trace test figures).
- Axis labels are read in a new way (located geometrically, read by
  Tesseract in several ways, and fitted on the principle that tick values
  step by exactly the same amount). On the test axes it never gave a wrong
  scale with high confidence; 1.2.0 misread decimal and logarithmic axes by
  large factors without warning. It takes about half a second a figure
  rather than twenty.
- Tesseract is now optional: without it, give the axis limits.
- Labels under 8 pixels high are read but never trusted outright; the
  result carries a message asking for the values to be checked.
- `mean_estimate` and `center_estimate` are documented for what they are
  (statistics of the trace's pixels in a column), with the consequence for
  dense records spelt out in the README.

### Fixed

- A one-pixel shift in the trace mask (from a 2×2 morphological opening)
  that erased one-pixel lines and produced all-white "traces".
- `tests/conftest.py` was missing from the 1.2.0 source distribution, so its
  tests could not be run from it.
- `use_lab=True` needed scikit-learn, and one helper needed SciPy, without
  either being declared. scikit-learn is now the optional extra
  `trace-digitiser[lab]`; SciPy is no longer used.
- `eval` of text read back from a data frame replaced by `ast.literal_eval`.
- The trace-accuracy test that failed in 1.2.0 (RMSE 102 against a limit of
  50) passes, and a new one holds all eight example figures to 7.5.

### Removed

- The slow "exhaustive" reading of y labels is no longer tried by default
  (`calibrate_y(..., exhaustive_fallback=True)` still does it). On axes that
  the new method declined, it was right in none of twenty trials.

## [1.2.0] - 2026-06-07

First version tracked in this repository; identical to the 1.2.0 files on PyPI.
Releases 0.1.0 to 1.0.0 (6-7 June 2026) were uploaded to PyPI directly and are
not recorded here.

[Unreleased]: https://github.com/comparativechrono/trace-digitiser/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/comparativechrono/trace-digitiser/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/comparativechrono/trace-digitiser/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/comparativechrono/trace-digitiser/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/comparativechrono/trace-digitiser/compare/1d34f75...v2.0.0
[1.2.0]: https://pypi.org/project/trace-digitiser/1.2.0/
