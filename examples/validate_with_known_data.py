"""How accurately can a figure like yours be digitised?  Test it with data you know.

Take a recording of the same kind as the published trace - here, a made-up
week of locomotor activity sampled every minute - draw it the way the
published figure was drawn, digitise the drawing, and compare what comes
back with what went in.

    python validate_with_known_data.py [output_directory]

Replace `t` and `activity` with your own recording, and set the size,
resolution and line width to resemble the figure you want to digitise.
"""

import sys
from pathlib import Path

import numpy as np

from trace_digitiser.validation import round_trip, synthetic_activity

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "output"

# A week of activity, one value a minute, between 0 and 1.
t, activity = synthetic_activity(days=7, seed=1)

for width, height in ((347, 137), (1400, 550)):  # the size of a small web figure, and four times that
    check = round_trip(
        t, 300 * activity, out, name=f"activity_{width}px",
        xlim=(0, 7), ylim=(0, 300), panel_px=(width, height), dpi=72,
        linewidth=0.8, style="paper", fmt="jpg", jpeg_quality=85, fontsize=6.5,
    )
    m = check["metrics"].iloc[0]
    columns = check["columns"]
    samples_per_column = len(t) / width
    print(f"\nPlot area {width} x {height} pixels: {samples_per_column:.0f} samples share each pixel column")
    print(f"  the middle of the drawn band is recovered to {m['rmse_pct']:.1f} % of the axis (RMS)")
    print(f"  its upper and lower edges to {m['rmse_upper_pct']:.1f} % and {m['rmse_lower_pct']:.1f} %")
    # What most analyses want is the mean of the samples in each column, and
    # a drawing does not keep it: many samples drawn on top of each other
    # show only their range.
    print(f"  mean of the digitised centre: {columns['center_estimate'].mean():.1f}")
    print(f"  mean of the recording:        {np.mean(300 * activity):.1f}")
    print(f"  correlation of the two, column by column: {m['corr_bin_mean']:.3f}")
    print(f"  overlay: {check['overlay']}")
