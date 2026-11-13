"""Command-line interface for trace_digitiser.

Usage::

    trace-digitiser figure.jpg --layout stacked --rows 2 --cols 1

    # With known axis ranges and a plot
    trace-digitiser figure.jpg --layout stacked --rows 2 --cols 1 \
        --y-min 0 --y-max 300 --total-days 7 --plot

    # Batch mode
    trace-digitiser figures/*.jpg --layout auto --output-dir results/ --plot

    # Generate synthetic test figures
    trace-digitiser --generate-test-figures --output-dir test_figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trace-digitiser",
        description="Template-free digitisation of raster scientific line plots.",
    )

    p.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Input image file(s) to digitise.",
    )
    p.add_argument(
        "--layout",
        dest="layout_mode",
        default="auto",
        choices=["auto", "single", "stacked", "horizontal", "grid"],
        help="Panel layout hint (default: auto).",
    )
    p.add_argument("--rows", dest="expected_rows", type=int, default=None, help="Expected number of panel rows.")
    p.add_argument("--cols", dest="expected_cols", type=int, default=None, help="Expected number of panel columns.")
    p.add_argument("--panels", dest="expected_panels", type=int, default=None, help="Expected total panel count.")
    p.add_argument("--y-min", dest="y_min", type=float, default=None, help="Known y-axis minimum (bypasses OCR).")
    p.add_argument("--y-max", dest="y_max", type=float, default=None, help="Known y-axis maximum (bypasses OCR).")
    p.add_argument("--x-min", dest="x_min", type=float, default=None, help="Known x-axis minimum (bypasses OCR).")
    p.add_argument("--x-max", dest="x_max", type=float, default=None, help="Known x-axis maximum (bypasses OCR).")
    p.add_argument(
        "--total-days", dest="total_days", type=float, default=None,
        help="Convenience shorthand for --x-min 0 --x-max TOTAL_DAYS.",
    )
    p.add_argument(
        "--x-units", dest="x_units", default="",
        help="Label for x-axis units in resampled output (e.g. 'hours', 'days').",
    )
    p.add_argument(
        "--sampling-interval", dest="sampling_interval", default=None,
        help="Resampling interval in x-axis units, or 'auto' to detect from trace steps. "
             "If omitted, per-pixel output is kept.",
    )
    p.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        type=Path,
        default=None,
        help="Output directory for CSV files, plots, and diagnostics.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a trace plot recreating the original figure.",
    )
    p.add_argument(
        "--plot-envelope",
        action="store_true",
        help="Include upper/lower envelope shading in the plot (implies --plot).",
    )
    p.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Write diagnostic overlay images.",
    )
    p.add_argument(
        "--generate-test-figures",
        action="store_true",
        help="Generate synthetic test plots instead of digitising.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for synthetic test generation (default: 123).",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

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

    from . import digitise

    do_plot = args.plot or args.plot_envelope

    for image_path in args.images:
        if not image_path.exists():
            print(f"WARNING: {image_path} not found, skipping.", file=sys.stderr)
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {image_path}")

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
        )

        n_panels = len(result.panels)
        n_traces = result.trace_data.groupby(["panel_id", "trace_id"]).ngroups if not result.trace_data.empty else 0
        print(f"  Panels: {n_panels}, Traces: {n_traces}")
        print(f"  Trace CSV:    {result.trace_csv_path}")
        print(f"  Metadata CSV: {result.metadata_csv_path}")
        if result.summary_csv_path:
            print(f"  Summary CSV:  {result.summary_csv_path}")

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


if __name__ == "__main__":
    main()
