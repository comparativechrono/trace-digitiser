"""Tests for axis calibration without OCR (trace_digitiser.calibration, x_calibration)."""

import numpy as np
import pytest

from trace_digitiser.calibration import (
    calibration_from_points,
    calibration_from_range,
    normalised_calibration,
    propagate_y_calibration_across_rows,
)
from trace_digitiser.models import Calibration
from trace_digitiser.x_calibration import apply_x, x_calibration_from_points, x_calibration_from_range

PANEL = {
    "panel_id": 1, "x0": 40, "x1": 390, "y_top": 20, "y_bottom": 160,
    "edges": {"left": 40.5, "right": 389.5, "top": 20.25, "bottom": 160.25},
}


class TestYCalibration:
    def test_from_range_uses_subpixel_edges(self):
        cal = calibration_from_range(PANEL, 0, 300)
        assert cal.scale_type == "user_supplied" and cal.method == "axis_limits"
        assert float(cal.y_to_value(160.25)) == pytest.approx(0)
        assert float(cal.y_to_value(20.25)) == pytest.approx(300)
        assert cal.a < 0

    def test_from_points(self):
        cal = calibration_from_points([(150, 10), (50, 60)])
        assert cal.scale_type == "user_reference"
        assert float(cal.y_to_value(100)) == pytest.approx(35)

    def test_from_points_least_squares(self):
        cal = calibration_from_points([(150, 10), (100, 35), (50, 60)])
        assert float(cal.y_to_value(75)) == pytest.approx(47.5)

    def test_log(self):
        cal = calibration_from_range(PANEL, 1, 1000, scale="log")
        assert cal.is_log and cal.scale_type == "log_user_supplied"
        assert float(cal.y_to_value(160.25)) == pytest.approx(1)
        assert float(cal.y_to_value(20.25)) == pytest.approx(1000)
        assert float(cal.y_to_value(90.25)) == pytest.approx(10 ** 1.5)

    def test_inverse(self):
        for cal in (calibration_from_range(PANEL, -5, 20), calibration_from_range(PANEL, 0.1, 100, scale="log")):
            rows = np.array([25.0, 80.5, 150.0])
            np.testing.assert_allclose(cal.value_to_y(cal.y_to_value(rows)), rows)

    def test_inverted_axis_is_flagged(self):
        cal = calibration_from_points([(50, 0), (150, 100)])  # values increase down the image
        assert cal.orientation_flipped

    @pytest.mark.parametrize(
        "points,scale",
        [
            ([(100, 5)], "linear"),  # one point
            ([(100, 5), (100, 9)], "linear"),  # same pixel
            ([(100, 5), (50, 5)], "linear"),  # same value
            ([(100, 0), (50, 10)], "log"),  # zero on a log axis
            ([(100, 1), (50, 10)], "cubic"),  # unknown scale
        ],
    )
    def test_bad_points(self, points, scale):
        with pytest.raises(ValueError):
            calibration_from_points(points, scale=scale)

    def test_normalised(self):
        cal = normalised_calibration(PANEL)
        assert not cal.is_calibrated
        assert float(cal.y_to_value(160.25)) == pytest.approx(0)
        assert float(cal.y_to_value(20.25)) == pytest.approx(1)

    def test_round_trip_through_dict(self):
        cal = calibration_from_range(PANEL, 0, 300)
        assert Calibration(**cal.to_dict()) == cal

    def test_legacy_scale_types_still_work(self):
        log = Calibration(a=-0.005, b=3.0, scale_type="log_ocr_value")
        np.testing.assert_allclose(log.y_to_value(np.array([0, 200, 400, 600])), [1000, 100, 10, 1], rtol=1e-6)


class TestXCalibration:
    def test_from_range(self):
        cal = x_calibration_from_range(PANEL, 0, 7)
        assert float(apply_x(cal, 40.5)) == pytest.approx(0)
        assert float(apply_x(cal, 389.5)) == pytest.approx(7)

    def test_from_points_log(self):
        cal = x_calibration_from_points([(100, 1), (300, 100)], scale="log")
        assert float(apply_x(cal, 200)) == pytest.approx(10)

    def test_bad_points(self):
        with pytest.raises(ValueError):
            x_calibration_from_points([(100, 1)])


class TestPropagation:
    def _panel(self, pid, x0, scale_type, a=-2.0, b=400.0):
        return {
            "panel_id": pid, "x0": x0, "x1": x0 + 100, "y_top": 50, "y_bottom": 200,
            "y_calibration": Calibration(a=a, b=b, scale_type=scale_type, confidence="high").to_dict(),
        }

    def test_unlabelled_sibling_inherits(self):
        panels = [self._panel(1, 0, "ocr_value"), self._panel(2, 200, "normalised_0_to_1", a=-1 / 150, b=200 / 150)]
        out = propagate_y_calibration_across_rows(panels)
        cal = out[1]["y_calibration"]
        assert cal["scale_type"] == "propagated_from_panel_1" and cal["method"] == "propagated"
        assert cal["a"] * 50 + cal["b"] == pytest.approx(300)
        assert cal["a"] * 200 + cal["b"] == pytest.approx(0)

    def test_calibrated_sibling_is_left_alone(self):
        panels = [self._panel(1, 0, "ocr_value"), self._panel(2, 200, "log_ocr_value", a=-0.01, b=3.0)]
        out = propagate_y_calibration_across_rows(panels)
        assert out[1]["y_calibration"]["scale_type"] == "log_ocr_value"
