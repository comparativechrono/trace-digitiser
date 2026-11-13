"""Regression tests on a real published figure.

Figure 4 of Lopatina et al. (2014), Front. Behav. Neurosci. 8:133
(CC BY; see examples/README.md): two stacked panels, 397 x 376 pixels,
each a week of home-cage activity drawn as a dense trace on a 0-300 axis
with interval labels 1N, 1D ... 7N, 7D along the top.
"""

import warnings

import numpy as np
import pytest

from trace_digitiser import digitise

from .conftest import needs_published_figure, needs_tesseract

pytestmark = needs_published_figure


@pytest.fixture(scope="module")
def with_limits(published_figure):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return digitise(published_figure, y_min=0, y_max=300, x_min=0, x_max=7, ocr=False, write_outputs=False)


def test_two_panels_one_trace_each(with_limits):
    assert len(with_limits.panels) == 2
    assert with_limits.n_traces == 2
    assert with_limits.warnings == []
    blue, pink = (
        np.array(eval(with_limits.trace_data[with_limits.trace_data.panel_id == i]["trace_median_rgb"].iloc[0]))
        for i in (1, 2)
    )
    assert blue[2] > blue[0] + 50  # the upper trace is blue ...
    assert pink[0] > pink[2] + 30  # ... the lower one pink


def test_every_column_is_seen(with_limits):
    data = with_limits.trace_data
    assert data.groupby("panel_id").size().tolist() == [348, 348]
    assert data["interpolated"].mean() < 0.03


def test_levels(with_limits):
    """Mean of the centre estimate: about 87 (upper panel) and 30 (lower), as read off the figure."""
    means = with_limits.trace_data.groupby("panel_id")["center_estimate"].mean()
    assert 84 <= means[1] <= 91
    assert 27 <= means[2] <= 33
    assert with_limits.trace_data["upper_envelope"].max() <= 300
    assert with_limits.trace_data["lower_envelope"].min() >= 0


def test_daily_rhythm_in_lower_panel(with_limits):
    """In the lower panel each interval labelled N is visibly busier than the D after it.

    (The upper panel is deliberately not tested this way: its busy periods
    straddle the N/D boundaries, so the two halves of the day differ little.)
    """
    data = with_limits.trace_data
    lower = data[data["panel_id"] == 2]
    first_half = (lower["x_value"] % 1.0) < 0.5
    assert lower[first_half]["center_estimate"].mean() > 1.5 * lower[~first_half]["center_estimate"].mean()


def test_matches_reference(with_limits, published_reference):
    """Guard against unintended changes: compare with the series stored in tests/data."""
    merged = with_limits.trace_data.merge(
        published_reference, on=["panel_id", "trace_id", "x_pixel"], suffixes=("", "_ref")
    )
    assert len(merged) == len(published_reference)
    for column in ("center_estimate", "upper_envelope", "lower_envelope"):
        diff = merged[column] - merged[f"{column}_ref"]
        assert np.sqrt((diff ** 2).mean()) < 1.0, column  # of a 0-300 axis
        assert np.corrcoef(merged[column], merged[f"{column}_ref"])[0, 1] > 0.999, column


@needs_tesseract
def test_fully_automatic(published_figure, with_limits):
    """Nothing supplied: the plot areas, the 0-300 scale and the interval labels are all found."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = digitise(published_figure, write_outputs=False)
    assert len(auto.panels) == 2 and auto.n_traces == 2
    for panel in auto.panels:
        cal = panel.calibration
        assert cal.scale_type == "ocr_value"
        # The labels are 5 to 6 pixels high.  Nearly all are read, and the scale is right, but print
        # that small is never trusted outright: the user is told what was read and asked to check it.
        assert cal.label_px < 8 and cal.confidence == "medium"
        assert len(cal.inliers) >= cal.n_labels - 1 == 6
        assert any(f"Panel {panel.panel_id}: the y-axis scale was read from" in w for w in auto.warnings)
        assert float(cal.y_to_value(panel.edges["top"])) == pytest.approx(300, abs=3)
        assert float(cal.y_to_value(panel.edges["bottom"])) == pytest.approx(0, abs=3)
        assert [lab.text for lab in panel.x_labels] == [f"{d}{p}" for d in range(1, 8) for p in "ND"]
        assert panel.x_calibration is None  # interval names are not a numeric axis
    diff = auto.trace_data["center_estimate"] - with_limits.trace_data["center_estimate"]
    assert diff.abs().max() < 3.0
    assert any("x axis" in w for w in auto.warnings)
    assert auto.summary_by_label["interval_label"].nunique() == 14
