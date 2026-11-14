"""Digitise a figure from a published paper.

The figure is Figure 4 of Lopatina et al. (2014): two stacked panels, each
with one dense coloured record of home-cage activity over seven days.  See
README.md in this directory for the source and its licence.

    python digitise_published_figure.py [output_directory]

Everything runs on this computer; nothing is uploaded.
"""

import sys
from pathlib import Path

from trace_digitiser import digitise

here = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "output"

result = digitise(
    here / "lopatina2014_fig4.jpg",
    # Two numbers read off each axis by eye.  The plot areas and the traces
    # are found automatically; the y scale would be too (leave these out and
    # see), but the labels in this image are six pixels high, and two
    # numbers typed in are worth more than any amount of OCR.
    y_min=0, y_max=300,  # the y axis runs from 0 at the bottom line to 300 at the top
    x_min=0, x_max=7,  # seven days, each labelled as a night (N) and a day (D)
    output_dir=out,
    save_overlay=True,
)

print(f"{len(result.panels)} panels, {result.n_traces} traces")
for message in result.warnings:
    print("CHECK:", message)

# One row per pixel column of each trace.
traces = result.trace_data
print(traces[["panel_id", "x_value", "center_estimate", "lower_envelope", "upper_envelope"]].head())

# Mean of each night and day, from the interval labels printed in the figure.
summary = result.summary_by_label
print(summary[["panel_id", "interval_label", "center_mean", "upper_envelope_max"]].head(4))

print("\nLook at this first - the digitised traces drawn over the figure:")
print(" ", result.overlay_path)
print("Data:      ", result.trace_csv_path)
print("Provenance:", result.provenance_path)
