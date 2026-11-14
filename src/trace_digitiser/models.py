"""Domain dataclasses for the trace-digitiser pipeline.

These replace the raw dicts that were passed between functions in the
original notebook.  Every field that the downstream code reads is an
explicit attribute so that typos become AttributeErrors, not silent
KeyError / None bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Y-axis calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Mapping from a pixel row to a y value.

    Linear axes use ``y_value = a * y_pixel + b``; for a standard upward
    y-axis rendered in an image whose y-pixel coordinate increases
    downward, *a* is negative.  Logarithmic axes (any ``scale_type``
    beginning with ``"log"``) use ``y_value = 10 ** (a * y_pixel + b)``.

    ``scale_type`` records where the calibration came from:

    ``"ocr_value"`` / ``"log_ocr_value"``
        tick labels read by OCR (see ``confidence``);
    ``"user_supplied"`` / ``"log_user_supplied"``
        axis limits given by the user, mapped to the plot-area edges;
    ``"user_reference"`` / ``"log_user_reference"``
        pixel/value reference points given by the user;
    ``"propagated_from_panel_N"``
        copied from the labelled panel in the same row;
    ``"normalised_0_to_1"``
        no calibration: 0 at the bottom of the plot area, 1 at the top.
    """

    a: float
    b: float
    scale_type: str
    orientation_flipped: bool = False
    ocr_candidates: dict[int, list[float]] = field(default_factory=dict)
    inliers: list[tuple[float, float]] = field(default_factory=list)
    # How far to trust an OCR calibration: "high", "medium" or "low" ("" if not from OCR).
    confidence: str = ""
    # How it was obtained: "tick_labels", "tick_labels_exhaustive", "axis_limits",
    # "reference_points", "propagated" or "none".
    method: str = ""
    n_labels: int = 0  # tick labels found beside the axis (``inliers`` holds those the scale rests on)
    rms_px: Optional[float] = None  # scatter of the agreeing labels about the fit, in pixels
    label_px: Optional[float] = None  # height of the labels' print in pixels

    # Convenience -----------------------------------------------------------

    @property
    def is_log(self) -> bool:
        return self.scale_type.startswith("log")

    @property
    def is_calibrated(self) -> bool:
        """False when the values are only normalised 0-1."""
        return self.scale_type != "normalised_0_to_1"

    def y_to_value(self, y) -> np.ndarray:
        """Apply the calibration to one or many y-pixel values."""
        y_arr = np.asarray(y, dtype=float)
        if self.is_log:
            return np.power(10.0, self.a * y_arr + self.b)
        return self.a * y_arr + self.b

    def value_to_y(self, value) -> np.ndarray:
        """Inverse of :meth:`y_to_value`: the pixel row at which a value is drawn."""
        v = np.asarray(value, dtype=float)
        if self.is_log:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log10(v)
        return (v - self.b) / self.a

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "scale_type": self.scale_type,
            "orientation_flipped": self.orientation_flipped,
            "ocr_candidates": self.ocr_candidates,
            "inliers": self.inliers,
            "confidence": self.confidence,
            "method": self.method,
            "n_labels": self.n_labels,
            "rms_px": self.rms_px,
            "label_px": self.label_px,
        }


# ---------------------------------------------------------------------------
# X-axis label
# ---------------------------------------------------------------------------

@dataclass
class XLabel:
    """A single OCR-detected x-axis label."""

    text: str
    conf: float
    x: float
    y: float
    left: float
    top: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# Detected panel
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    """One rectangular plot panel inside a figure image."""

    panel_id: int
    x0: int
    x1: int
    y_top: int
    y_bottom: int
    gridline_y: list[int] = field(default_factory=list)
    source: str = ""
    score: float = 0.0
    layout_mode: str = "auto"
    calibration: Optional[Calibration] = None
    x_labels: list[XLabel] = field(default_factory=list)
    # Sub-pixel positions of the plot-area edges: left, right, top, bottom.
    edges: dict[str, float] = field(default_factory=dict)
    # "box" (closed frame), "L" (left and bottom axes only) or "partial".
    kind: str = "box"
    # x-axis calibration, if any: keys a, b, scale_type, inliers, confidence.
    x_calibration: Optional[dict] = None

    # Convenience -----------------------------------------------------------

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y_bottom - self.y_top

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


# ---------------------------------------------------------------------------
# Detected trace (colour mask)
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """A colour-segmented trace inside one panel."""

    panel_id: int
    trace_id: int
    mask: np.ndarray  # bool array, shape = panel ROI
    pixel_count: int
    median_rgb: tuple[int, int, int]
    hue_group: tuple[float, float] = (0.0, 1.0)


# ---------------------------------------------------------------------------
# End-to-end result
# ---------------------------------------------------------------------------

@dataclass
class DigitiserResult:
    """Everything returned by :func:`digitise`."""

    image_path: Optional[Path]  # None if an image array was passed in
    rgb: np.ndarray
    panels: list[Panel]
    trace_data: pd.DataFrame
    summary_by_label: pd.DataFrame
    panel_metadata: pd.DataFrame
    trace_csv_path: Optional[Path] = None
    summary_csv_path: Optional[Path] = None
    metadata_csv_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    provenance_path: Optional[Path] = None
    straightened_path: Optional[Path] = None  # the image turned upright, if it was tilted
    # Things the user should check: weak calibrations, missing panels, etc.
    warnings: list[str] = field(default_factory=list)
    # Full record of inputs, settings, versions and results (see provenance.py).
    provenance: dict = field(default_factory=dict)

    @property
    def n_traces(self) -> int:
        """Number of traces digitised, over all panels."""
        if self.trace_data.empty:
            return 0
        return int(self.trace_data.groupby(["panel_id", "trace_id"]).ngroups)

    def overlay(self, output_path: Optional[str | Path] = None, show: bool = False):
        """Draw the digitised traces on top of the original figure.

        This is the quickest check of a digitisation: the drawn line should
        sit on the trace, the box on the axes, and the labelled tick values
        on the ticks.  Returns the overlay as an RGB array.
        """
        from .diagnostics import draw_result_overlay

        return draw_result_overlay(
            self.rgb, self.panels, self.trace_data,
            output_path=Path(output_path) if output_path else None, show=show,
        )

    def plot(
        self,
        output_path: Optional[str | Path] = None,
        show: bool = True,
        show_envelope: bool = False,
        title: Optional[str] = None,
    ):
        """Plot the digitised traces, recreating the original figure layout.

        Parameters
        ----------
        output_path : str or Path, optional
            Save the plot to this file.
        show : bool
            Display the plot interactively (default True).
        show_envelope : bool
            Shade the area between upper and lower envelopes.
        title : str, optional
            Overall figure title.
        """
        from .diagnostics import plot_traces

        return plot_traces(
            self.trace_data,
            self.panels,
            output_path=Path(output_path) if output_path else None,
            show=show,
            show_envelope=show_envelope,
            title=title,
        )

    def resample(
        self,
        interval: Optional[float] = None,
        x_units: str = "",
    ) -> ResampledResult:
        """Resample per-pixel trace data to a regular sampling interval.

        Parameters
        ----------
        interval : float, optional
            Sampling interval in x-axis units (the same units used for
            *x_min*/*x_max*).  If None, the function auto-detects the
            interval from the trace's step structure.  If the trace is
            smoothed or too dense for detection, per-pixel data is
            returned unchanged.
        x_units : str
            Label for the x-axis units (e.g. ``"hours"``, ``"days"``,
            ``"minutes"``).  Included as a column in the output.

        Returns
        -------
        ResampledResult
            Wrapper with ``.data`` (the DataFrame), ``.plot()``, and
            ``.to_csv()`` convenience methods.

        Examples
        --------
        ::

            resampled = result.resample(interval=1/24, x_units="days")
            resampled.plot()                        # visualise
            resampled.plot(output_path="out.png")   # save
            resampled.to_csv("resampled.csv")       # export
            df = resampled.data                     # raw DataFrame
        """
        from .resampling import resample_trace

        # Convert user interval in x-units to pixel interval.
        interval_pixels = None
        if interval is not None and "x_value" in self.trace_data.columns:
            # Compute pixels-per-x-unit from the first trace.
            sub = self.trace_data[self.trace_data["panel_id"] == self.trace_data["panel_id"].iloc[0]]
            x_range = sub["x_value"].max() - sub["x_value"].min()
            n_pix = sub["x_pixel"].nunique()
            if x_range > 0 and n_pix > 1:
                pix_per_unit = (n_pix - 1) / x_range
                interval_pixels = interval * pix_per_unit

        df = resample_trace(
            self.trace_data,
            interval_pixels=interval_pixels,
            x_units=x_units,
        )

        return ResampledResult(data=df, panels=self.panels, x_units=x_units)


@dataclass
class ResampledResult:
    """Resampled trace data with plotting support.

    Attributes
    ----------
    data : pd.DataFrame
        The resampled DataFrame.
    panels : list[Panel]
        Panel metadata (for x-label positioning in plots).
    x_units : str
        Label for the x-axis units.
    """

    data: pd.DataFrame
    panels: list[Panel]
    x_units: str = ""

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        n_rows = len(self.data)
        n_panels = self.data["panel_id"].nunique() if not self.data.empty else 0
        n_traces = self.data.groupby(["panel_id", "trace_id"]).ngroups if not self.data.empty else 0
        return f"ResampledResult({n_rows} rows, {n_panels} panels, {n_traces} traces)"

    def to_csv(self, path: str | Path, **kwargs) -> None:
        """Save the resampled data to CSV."""
        self.data.to_csv(str(path), index=False, **kwargs)

    def plot(
        self,
        output_path: Optional[str | Path] = None,
        show: bool = True,
        show_envelope: bool = True,
        title: Optional[str] = None,
    ):
        """Plot the resampled traces.

        Parameters
        ----------
        output_path : str or Path, optional
            Save the plot to this file.
        show : bool
            Display the plot interactively (default True).
        show_envelope : bool
            Shade the upper/lower envelope (default True for resampled).
        title : str, optional
            Overall figure title.
        """
        from .diagnostics import plot_traces

        return plot_traces(
            self.data,
            self.panels,
            output_path=Path(output_path) if output_path else None,
            show=show,
            show_envelope=show_envelope,
            title=title,
            x_label=f"Time ({self.x_units})" if self.x_units else None,
        )
