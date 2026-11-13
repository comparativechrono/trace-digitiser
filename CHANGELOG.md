# Changelog

All notable changes to trace-digitiser are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/).

## [2.0.0] - not yet released

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

[2.0.0]: https://github.com/comparativechrono/trace-digitiser/compare/1d34f75...HEAD
[1.2.0]: https://pypi.org/project/trace-digitiser/1.2.0/
