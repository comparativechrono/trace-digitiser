"""Command-line interface for trace_digitiser.

Usage::

    # Let the tool find the plot, read the axes and extract the traces
    trace-digitiser figure.jpg -o out/ --overlay

    # Supply the axis limits (recommended when you know them)
    trace-digitiser figure.jpg --y-min 0 --y-max 300 --total-days 7 -o out/ --overlay

    # Set the plot area and axis reference points by hand
    trace-digitiser scan.png --plot-area 41,23,388,159 \\
        --y-ref 159:0 --y-ref 23:300 --x-ref 41:0 --x-ref 388:7 -o out/

    # Batch mode
    trace-digitiser figures/*.jpg -o results/ --plot

    # Check the tool against figures drawn from known data
    trace-digitiser --validate validation/

    # Other kinds of figure (each takes the axis options above, and options of its own: --help)
    trace-digitiser points scatter.png --overlay        # markers, and their error bars
    trace-digitiser bars bars.png --overlay             # the end of each bar, and its error bar
    trace-digitiser rows actogram.png --bins 48         # a stack of rows (actograms, double plots)
    trace-digitiser heatmap heat.png --rows 21          # cells, read off the colour bar
    trace-digitiser polar phase.png --r-ref 37.5:1955 --r-ref 98.2:1965
    trace-digitiser describe figure.png                 # what is found in a figure, as JSON

Everything runs on this computer; no image or data is sent anywhere.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np


def _box(text: str) -> tuple[float, float, float, float]:
    """Parse ``X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM``."""
    try:
        values = tuple(float(v) for v in text.replace(";", ",").split(","))
    except ValueError:
        values = ()
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"expected X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM (four numbers), got {text!r}")
    return values  # type: ignore[return-value]


def _point(text: str) -> tuple[float, float]:
    """Parse ``PIXEL:VALUE``."""
    pixel, sep, value = text.partition(":")
    try:
        if not sep:
            raise ValueError
        return float(pixel), float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected PIXEL:VALUE (for example 159:0), got {text!r}") from None


def _straighten(text: str):
    """Parse ``auto``, ``never`` or a number of degrees."""
    if text in ("auto", "never"):
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected auto, never or a number of degrees, got {text!r}") from None


def _build_parser() -> argparse.ArgumentParser:
    from .provenance import package_version

    p = argparse.ArgumentParser(
        prog="trace-digitiser",
        description=(
            "Digitise line plots from figure images. Runs entirely on this computer: "
            "nothing is uploaded and no AI model is involved."
        ),
        epilog="Pixel coordinates count from the top-left corner of the image, starting at 0.",
    )
    p.add_argument("--version", action="version", version=f"trace-digitiser {package_version()}")

    p.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Input image file(s) to digitise.",
    )

    layout = p.add_argument_group("finding the plot area")
    layout.add_argument(
        "--straighten", type=_straighten, default="auto", metavar="auto|never|DEGREES",
        help="A tilted figure (a scan) is turned upright first. 'auto' (default) measures the tilt; a number "
        "gives the figure's anticlockwise tilt in degrees; 'never' leaves the image alone.",
    )
    layout.add_argument(
        "--plot-area", dest="plot_area", type=_box, action="append", metavar="X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM",
        help="Pixel positions of the axis lines, instead of automatic detection. Repeat for several panels.",
    )
    layout.add_argument(
        "--layout",
        dest="layout_mode",
        default="auto",
        choices=["auto", "single", "stacked", "horizontal", "grid"],
        help="Panel layout hint (default: auto). Rarely needed.",
    )
    layout.add_argument("--rows", dest="expected_rows", type=int, default=None, help="Expected number of panel rows.")
    layout.add_argument(
        "--cols", dest="expected_cols", type=int, default=None, help="Expected number of panel columns."
    )
    layout.add_argument("--panels", dest="expected_panels", type=int, default=None, help="Expected total panel count.")

    axes = p.add_argument_group("axis values (given values always take precedence over OCR)")
    axes.add_argument(
        "--y-min", dest="y_min", type=float, default=None, help="y value at the bottom edge of the plot area."
    )
    axes.add_argument(
        "--y-max", dest="y_max", type=float, default=None, help="y value at the top edge of the plot area."
    )
    axes.add_argument(
        "--x-min", dest="x_min", type=float, default=None, help="x value at the left edge of the plot area."
    )
    axes.add_argument(
        "--x-max", dest="x_max", type=float, default=None, help="x value at the right edge of the plot area."
    )
    axes.add_argument(
        "--total-days", dest="total_days", type=float, default=None,
        help="Shorthand for --x-min 0 --x-max TOTAL_DAYS.",
    )
    axes.add_argument(
        "--y-ref",
        dest="y_ref",
        type=_point,
        action="append",
        metavar="ROW:VALUE",
        help="A y-axis reference point: the pixel row of a tick and its value. Give at least two. "
        "Single-panel figures.",
    )
    axes.add_argument(
        "--x-ref",
        dest="x_ref",
        type=_point,
        action="append",
        metavar="COLUMN:VALUE",
        help="An x-axis reference point: the pixel column of a tick and its value. Give at least two. "
        "Single-panel figures.",
    )
    axes.add_argument(
        "--y-scale", dest="y_scale", default="auto", choices=["auto", "linear", "log"], help="Kind of y axis."
    )
    axes.add_argument(
        "--x-scale", dest="x_scale", default="auto", choices=["auto", "linear", "log"], help="Kind of x axis."
    )
    axes.add_argument(
        "--no-ocr", dest="ocr", action="store_false", help="Never run Tesseract; use only the values given."
    )

    traces = p.add_argument_group("finding the traces")
    traces.add_argument(
        "--trace-colour", "--trace-color", dest="trace_colours", action="append", metavar="RRGGBB",
        help="Colour of a trace to extract, as hex (e.g. 578dd2). Repeat for several traces. Default: automatic.",
    )
    traces.add_argument(
        "--exclude", dest="exclude", type=_box, action="append", metavar="X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM",
        help="A region to ignore when looking for traces, e.g. a legend. Repeatable.",
    )
    traces.add_argument(
        "--achromatic", default="auto", choices=["auto", "always", "never"],
        help="Whether to look for a black or grey trace (default: auto).",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory for CSV files, plots, and diagnostics (default: current directory).",
    )
    out.add_argument(
        "--overlay", action="store_true",
        help="Save the digitised traces drawn on the original figure, for checking by eye. Recommended.",
    )
    out.add_argument(
        "--plot",
        action="store_true",
        help="Save a plot of the digitised traces.",
    )
    out.add_argument(
        "--plot-envelope",
        action="store_true",
        help="Include upper/lower envelope shading in the plot (implies --plot).",
    )
    out.add_argument(
        "--x-units", dest="x_units", default="",
        help="Label for x-axis units in resampled output (e.g. 'hours', 'days').",
    )
    out.add_argument(
        "--sampling-interval", dest="sampling_interval", default=None,
        help="Resampling interval in x-axis units, or 'auto' to detect from trace steps. "
             "If omitted, per-pixel output is kept.",
    )
    out.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Write intermediate images (panel detection, trace masks) as well as the overlay.",
    )

    tools = p.add_argument_group("other tools")
    tools.add_argument(
        "--validate", dest="validate", type=Path, default=None, metavar="DIR",
        help="Digitise a suite of figures drawn from known data, write the results to DIR and print the errors.",
    )
    tools.add_argument(
        "--generate-test-figures",
        action="store_true",
        help="Generate synthetic test plots instead of digitising.",
    )
    tools.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for synthetic test generation (default: 123).",
    )

    return p


def _describe(result) -> list[str]:
    """One line per panel: where the axis values came from."""
    from . import _tidy

    lines = []
    for panel in result.panels:
        cal = panel.calibration
        if cal is None:
            continue
        e = panel.edges
        lo, hi = float(cal.y_to_value(e["bottom"])), float(cal.y_to_value(e["top"]))
        if not cal.is_log:  # tidy rounding noise such as -2.8e-13
            step = 10.0 ** (np.floor(np.log10(abs(hi - lo) or 1.0)) - 4)
            lo, hi = round(lo / step) * step + 0.0, round(hi / step) * step + 0.0
        source = {
            "tick_labels": f"read from tick labels ({cal.confidence} confidence)",
            "tick_labels_exhaustive": f"read from tick labels ({cal.confidence} confidence)",
            "axis_limits": "from the limits given",
            "reference_points": "from the reference points given",
            "propagated": "copied from a neighbouring panel",
            "none": "NOT CALIBRATED (0-1)",
        }.get(cal.method, cal.scale_type)
        xcal = panel.x_calibration
        x_source = {
            None: "not calibrated",
            "tick_labels": f"read from tick labels ({(xcal or {}).get('confidence', '')} confidence)",
            "axis_limits": "from the limits given",
            "reference_points": "from the reference points given",
        }.get((xcal or {}).get("method") if xcal else None, "calibrated")
        if cal.method.startswith("tick_labels"):
            span = f"{_tidy(lo)} to {_tidy(hi)}"  # a fitted scale: three figures say all there is to say
        else:
            span = f"{lo:g} to {hi:g}"
        lines.append(
            f"    panel {panel.panel_id}: y {span}{' (log)' if cal.is_log else ''}, {source}; x {x_source}"
        )
    return lines


SUBCOMMANDS = ("lines", "points", "bars", "rows", "heatmap", "polar", "describe")


def _numbers(text: str) -> list[float]:
    """``1,2,3`` or ``1955:2013`` (every whole number from one to the other) or ``0:1:0.25`` (with a step)."""
    text = text.strip()
    if ":" in text and "," not in text:
        parts = [float(v) for v in text.split(":")]
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) > 2 else 1.0
        n = int(round((stop - start) / step)) + 1
        return [start + i * step for i in range(max(0, n))]
    try:
        return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected numbers separated by commas, or FIRST:LAST, got {text!r}") from None


def _pair(text: str) -> tuple[float, float]:
    values = _numbers(text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"expected two numbers, A,B, got {text!r}")
    return values[0], values[1]


def _figure_options(p: argparse.ArgumentParser, axes: bool = True) -> None:
    p.add_argument("images", nargs="+", type=Path, help="Input image file(s).")
    g = p.add_argument_group("the figure and its axes")
    g.add_argument("--straighten", type=_straighten, default="auto", metavar="auto|never|DEGREES",
                   help="Turn a tilted figure upright first (default: auto).")
    if axes:
        g.add_argument("--plot-area", dest="plot_area", type=_box, action="append",
                       metavar="X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM", help="The plot area, by hand. Repeatable.")
        g.add_argument("--y-min", dest="y_min", type=float, help="y value at the bottom edge of the plot area.")
        g.add_argument("--y-max", dest="y_max", type=float, help="y value at the top edge of the plot area.")
        g.add_argument("--x-min", dest="x_min", type=float, help="x value at the left edge of the plot area.")
        g.add_argument("--x-max", dest="x_max", type=float, help="x value at the right edge of the plot area.")
        g.add_argument("--y-ref", dest="y_ref", type=_point, action="append", metavar="ROW:VALUE",
                       help="A y-axis reference point (give two or more).")
        g.add_argument("--x-ref", dest="x_ref", type=_point, action="append", metavar="COLUMN:VALUE",
                       help="An x-axis reference point (give two or more).")
        g.add_argument("--y-scale", dest="y_scale", default="auto", choices=["auto", "linear", "log"])
        g.add_argument("--x-scale", dest="x_scale", default="auto", choices=["auto", "linear", "log"])
        g.add_argument("--exclude", type=_box, action="append", metavar="X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM",
                       help="A region to ignore (a legend, a label). Repeatable.")
        g.add_argument("--no-ocr", dest="ocr", action="store_false", help="Never run Tesseract.")
    o = p.add_argument_group("output")
    o.add_argument("--output-dir", "-o", dest="output_dir", type=Path, default=None,
                   help="Where to write the files (default: the current directory).")
    o.add_argument("--overlay", action="store_true", help="Also save what was found drawn over the figure.")


def _figure_kwargs(args) -> dict:
    out = {"straighten": args.straighten}
    names = ("plot_area", "y_min", "y_max", "x_min", "x_max", "y_ref", "x_ref", "y_scale", "x_scale", "exclude", "ocr")
    for k in names:
        if hasattr(args, k):
            out[k] = getattr(args, k)
    return out


def _subcommand(name: str, argv: list[str]) -> None:
    if name == "lines":
        return main(argv)
    from .provenance import package_version

    p = argparse.ArgumentParser(prog=f"trace-digitiser {name}", epilog="Pixel coordinates count from the top-left "
                                "corner of the image, starting at 0. Everything runs on this computer.")
    p.add_argument("--version", action="version", version=f"trace-digitiser {package_version()}")
    if name == "describe":
        p.description = "Say what is found in a figure - its plot areas, axes, tick labels and colours - as JSON."
        p.add_argument("images", nargs="+", type=Path, help="Input image file(s).")
        p.add_argument("--straighten", type=_straighten, default="auto", metavar="auto|never|DEGREES")
        p.add_argument("--plot-area", dest="plot_area", type=_box, action="append",
                       metavar="X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM")
        p.add_argument("--no-ocr", dest="ocr", action="store_false")
        p.add_argument("--output", "-o", dest="output", type=Path, default=None, help="Write the JSON to a file.")
        args = p.parse_args(argv)
        import json

        from .explore import describe_figure

        out = {}
        for image in args.images:
            out[str(image)] = describe_figure(image, straighten=args.straighten, plot_area=args.plot_area,
                                              ocr=args.ocr)
        text = json.dumps(out if len(out) > 1 else next(iter(out.values())), indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(text)
        return None
    if name == "points":
        p.description = "Read the points (markers) of a figure, and their error bars."
        _figure_options(p)
        g = p.add_argument_group("the points")
        g.add_argument("--colour", "--color", dest="colours", action="append", metavar="COLOUR",
                       help="The colour of the points (#rrggbb, or a name such as black). Repeatable.")
        g.add_argument("--colour-tolerance", dest="colour_tolerance", type=float, default=60.0)
        g.add_argument("--marker-size", dest="marker_size", type=float, help="The markers' diameter in pixels.")
        g.add_argument("--error-bars", dest="error_bars", default="auto",
                       choices=["auto", "vertical", "horizontal", "both", "none"])
        args = p.parse_args(argv)
        from .points import digitise_points as reader

        extra = {"colours": args.colours, "colour_tolerance": args.colour_tolerance,
                 "marker_size": args.marker_size, "error_bars": args.error_bars}
    elif name == "bars":
        p.description = "Read the bars of a bar chart: where each ends, and its error bar."
        _figure_options(p)
        g = p.add_argument_group("the bars")
        g.add_argument("--colour", "--color", dest="colours", action="append", metavar="COLOUR")
        g.add_argument("--colour-tolerance", dest="colour_tolerance", type=float, default=40.0)
        g.add_argument("--baseline", type=float, help="The value the bars start from (default 0, if on the axis).")
        g.add_argument("--horizontal", action="store_true", help="Bars that run across from the y axis.")
        g.add_argument("--no-error-bars", dest="error_bars", action="store_false")
        args = p.parse_args(argv)
        from .bars import digitise_bars as reader

        extra = {"colours": args.colours, "colour_tolerance": args.colour_tolerance, "baseline": args.baseline,
                 "orientation": "horizontal" if args.horizontal else "vertical", "error_bars": args.error_bars}
    elif name == "rows":
        p.description = "Read a figure drawn as a stack of rows: an actogram, a double-plotted record, a birthogram."
        _figure_options(p)
        g = p.add_argument_group("the rows")
        g.add_argument("--row-values", dest="row_values", type=_numbers, metavar="V1,V2,... or FIRST:LAST",
                       help="The y-axis value of each row's baseline (1955:2013).")
        g.add_argument("--row-positions", dest="row_positions", type=_numbers, metavar="ROW1,ROW2,...",
                       help="The pixel row of each baseline.")
        g.add_argument("--rows", dest="rows", type=int, help="How many evenly spaced rows (with --row-range).")
        g.add_argument("--row-range", dest="row_range", type=_pair, metavar="FIRST,LAST",
                       help="The pixel rows of the first and last baselines.")
        g.add_argument("--row-spacing-value", dest="row_spacing_value", type=float,
                       help="What the distance between two baselines stands for (1.1 if values 0-1 are drawn 1.1 "
                       "apart).")
        g.add_argument("--direction", default="auto", choices=["auto", "up", "down"])
        g.add_argument("--bins", type=int, help="Average each row over this many bins (24, 48 ...).")
        g.add_argument("--colour", "--color", dest="colour", metavar="COLOUR", help="The colour of the fill.")
        g.add_argument("--dots", metavar="COLOUR", help="Also find dots of this colour, and say which bin each is in.")
        args = p.parse_args(argv)
        from .rows import digitise_rows as reader

        extra = {k: getattr(args, k) for k in ("row_values", "row_positions", "rows", "row_range",
                                               "row_spacing_value", "direction", "bins", "colour", "dots")}
    elif name == "heatmap":
        p.description = "Read a heat map: the value of each cell, from its colour on the colour bar."
        _figure_options(p)
        g = p.add_argument_group("the heat map")
        g.add_argument("--colourbar", "--colorbar", dest="colourbar", type=_box,
                       metavar="X_LEFT,Y_TOP,X_RIGHT,Y_BOTTOM", help="Where the colour bar is (found if not given).")
        g.add_argument("--colourbar-values", "--colorbar-values", dest="colourbar_values", type=_pair,
                       metavar="BOTTOM,TOP", help="The values at the ends of the colour bar (read if not given).")
        g.add_argument("--colormap", "--colourmap", dest="colormap", metavar="NAME",
                       help="Instead of a colour bar: matplotlib's colour map (viridis ...), with --vmin and --vmax.")
        g.add_argument("--vmin", type=float)
        g.add_argument("--vmax", type=float)
        g.add_argument("--rows", type=int, help="How many cells down the map (found if not given).")
        g.add_argument("--cols", type=int, help="How many cells across.")
        g.add_argument("--max-colour-error", dest="max_colour_error", type=float, default=12.0)
        args = p.parse_args(argv)
        from .heatmap import digitise_heatmap as reader

        extra = {k: getattr(args, k) for k in ("colourbar", "colourbar_values", "colormap", "vmin", "vmax", "rows",
                                               "cols", "max_colour_error")}
    elif name == "polar":
        p.description = "Read the points of a polar plot: the angle of each and its distance from the centre."
        _figure_options(p, axes=False)
        g = p.add_argument_group("the polar axes")
        g.add_argument("--centre", "--center", dest="centre", type=_pair, metavar="X,Y")
        g.add_argument("--r-ref", dest="r_ref", type=_point, action="append", metavar="RADIUS:VALUE",
                       help="A distance from the centre (pixels) and what it stands for. Give two or more.")
        g.add_argument("--outer-value", dest="outer_value", type=float, help="The value at the outer circle.")
        g.add_argument("--inner-value", dest="inner_value", type=float, default=0.0)
        g.add_argument("--zero", default="east", choices=["east", "north", "west", "south"])
        g.add_argument("--clockwise", action="store_true")
        g.add_argument("--colour", "--color", dest="colours", action="append", metavar="COLOUR")
        g.add_argument("--marker-size", dest="marker_size", type=float)
        args = p.parse_args(argv)
        from .polar import digitise_polar as reader

        extra = {k: getattr(args, k) for k in ("centre", "r_ref", "outer_value", "inner_value", "zero", "clockwise",
                                               "colours", "marker_size")}
    failures = 0
    for image in args.images:
        if not image.exists():
            print(f"WARNING: {image} not found, skipping.", file=sys.stderr)
            failures += 1
            continue
        print(f"\n{'=' * 60}")
        print(f"Processing: {image}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kwargs = dict(extra, output_dir=args.output_dir, save_overlay=args.overlay)
                if name == "polar":
                    kwargs["straighten"] = False if args.straighten in ("auto", "never") else args.straighten
                else:
                    kwargs.update(_figure_kwargs(args))
                result = reader(image, **kwargs)
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"  Found: {len(result.data)} {name if len(result.data) != 1 else name.rstrip('s')}"
              + (f" in {len(result.panels)} panel(s)" if result.panels else ""))
        if len(result.data):
            import pandas as pd

            with pd.option_context("display.width", 160, "display.max_columns", 12):
                print("  " + result.data.head(5).to_string(index=False).replace("\n", "\n  "))
        print(f"  CSV:          {result.csv_path}")
        print(f"  Provenance:   {result.provenance_path}")
        if result.straightened_path:
            print(f"  Straightened: {result.straightened_path}")
        if result.overlay_path:
            print(f"  Overlay:      {result.overlay_path}")
        for message in result.warnings:
            print(f"  CHECK: {message}")
        if not result.overlay_path:
            print("  Tip: add --overlay to save a picture of what was found, drawn on the figure.")
    if failures:
        sys.exit(1)
    return None


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS and not Path(argv[0]).exists():
        return _subcommand(argv[0], argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Validation suite ------------------------------------------------
    if args.validate is not None:
        import pandas as pd

        from .validation import run_validation, summarise

        results = run_validation(args.validate, progress=True)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print()
            print(summarise(results).to_string(index=False))
        print(f"\nErrors are percent of the y-axis range. Per-trace results: {args.validate / 'results.csv'}")
        return

    # --- Generate test figures -------------------------------------------
    if args.generate_test_figures:
        from .synthetic import generate_noisy_example_plots

        out = args.output_dir or Path("test_figures")
        paths, truth = generate_noisy_example_plots(output_dir=out, seed=args.seed, show=False)
        print("Generated test figures:")
        for p in paths:
            print(f"  {p}")
        print(f"Ground truth: {truth}")
        return

    # --- Digitise images -------------------------------------------------
    if not args.images:
        parser.print_help()
        sys.exit(1)
    for name, points in (("--y-ref", args.y_ref), ("--x-ref", args.x_ref)):
        if points is not None and len(points) < 2:
            parser.error(f"{name} must be given at least twice (two reference points define an axis)")
    for lo, hi, name in ((args.y_min, args.y_max, "y"), (args.x_min, args.x_max, "x")):
        if (lo is None) != (hi is None):
            parser.error(f"--{name}-min and --{name}-max must be given together")

    from . import digitise

    do_plot = args.plot or args.plot_envelope
    failures = 0

    for image_path in args.images:
        if not image_path.exists():
            print(f"WARNING: {image_path} not found, skipping.", file=sys.stderr)
            failures += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {image_path}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # shown below, from result.warnings
                result = digitise(
                    image_path,
                    layout_mode=args.layout_mode,
                    expected_rows=args.expected_rows,
                    expected_cols=args.expected_cols,
                    expected_panels=args.expected_panels,
                    y_min=args.y_min,
                    y_max=args.y_max,
                    x_min=args.x_min,
                    x_max=args.x_max,
                    total_days=args.total_days,
                    output_dir=args.output_dir,
                    show_debug=False,
                    save_diagnostics=args.save_diagnostics,
                    plot_area=args.plot_area,
                    y_ref=args.y_ref,
                    x_ref=args.x_ref,
                    y_scale=args.y_scale,
                    x_scale=args.x_scale,
                    trace_colours=args.trace_colours,
                    exclude=args.exclude,
                    achromatic=args.achromatic,
                    ocr=args.ocr,
                    save_overlay=args.overlay,
                    straighten=args.straighten,
                )
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(f"  Panels: {len(result.panels)}, Traces: {result.n_traces}")
        for line in _describe(result):
            print(line)
        print(f"  Trace CSV:    {result.trace_csv_path}")
        print(f"  Metadata CSV: {result.metadata_csv_path}")
        if result.summary_csv_path:
            print(f"  Summary CSV:  {result.summary_csv_path}")
        print(f"  Provenance:   {result.provenance_path}")
        if result.straightened_path:
            print(f"  Straightened: {result.straightened_path}")
        if result.overlay_path:
            print(f"  Overlay:      {result.overlay_path}")

        # --- Resampling ---
        if args.sampling_interval is not None:
            ivl = None if args.sampling_interval == "auto" else float(args.sampling_interval)
            resampled = result.resample(interval=ivl, x_units=args.x_units)
            out_dir = args.output_dir or Path(".")
            resample_path = out_dir / f"{image_path.stem}_resampled.csv"
            resampled.to_csv(resample_path)
            print(f"  Resampled:    {resample_path}  ({len(resampled)} rows)")

        if do_plot:
            out_dir = args.output_dir or Path(".")
            plot_path = out_dir / f"{image_path.stem}_trace_plot.png"
            result.plot(
                output_path=plot_path,
                show=False,
                show_envelope=args.plot_envelope,
            )
            print(f"  Plot:         {plot_path}")

        for message in result.warnings:
            print(f"  CHECK: {message}")
        if not result.overlay_path:
            print("  Tip: add --overlay to save a picture of the result drawn on the figure.")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
