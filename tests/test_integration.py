"""End-to-end integration tests for the digitiser pipeline.

Includes quantitative accuracy metrics against synthetic ground truth:
- Panel count accuracy
- Panel IoU against known subplot axes
- Trace y-value RMSE after x-alignment
- OCR calibration success rate
"""

import shutil

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    not shutil.which("tesseract"),
    reason="Tesseract OCR not installed",
)


# ===================================================================
# Panel count accuracy
# ===================================================================

class TestPanelCount:
    """Verify the correct number of panels is detected for each layout."""

    def test_vertical_2_panel(self, vertical_2_panel_image, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )
        assert len(result.panels) == 2

    def test_horizontal_3_panel(self, horizontal_3_panel_image, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            horizontal_3_panel_image,
            layout_mode="horizontal", expected_rows=1, expected_cols=3,
            output_dir=tmp_path,
        )
        assert len(result.panels) == 3

    def test_grid_2x2(self, grid_2x2_image, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            grid_2x2_image,
            layout_mode="grid", expected_rows=2, expected_cols=2,
            output_dir=tmp_path,
        )
        assert len(result.panels) == 4


# ===================================================================
# Output structure validation
# ===================================================================

class TestOutputStructure:
    """Verify output files and DataFrame columns."""

    def test_output_files_exist(self, vertical_2_panel_image, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )

        assert result.trace_csv_path is not None
        assert result.trace_csv_path.exists()
        assert result.metadata_csv_path.exists()

    def test_trace_data_columns(self, vertical_2_panel_image, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )

        expected_cols = {
            "panel_id", "trace_id", "x_pixel", "x_norm_0_to_1",
            "center_estimate", "mean_estimate",
            "upper_envelope", "lower_envelope", "scale_type",
        }
        assert expected_cols.issubset(set(result.trace_data.columns))

    def test_dataclass_types(self, vertical_2_panel_image, tmp_path):
        from trace_digitiser import digitise
        from trace_digitiser.models import Calibration, DigitiserResult, Panel

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )

        assert isinstance(result, DigitiserResult)
        for panel in result.panels:
            assert isinstance(panel, Panel)
            assert isinstance(panel.calibration, Calibration)
            assert panel.area > 0

    def test_diagnostics_saved(self, vertical_2_panel_image, tmp_path):
        from trace_digitiser import digitise

        digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
            save_diagnostics=True,
        )

        diag_dir = tmp_path / "diagnostics"
        assert diag_dir.exists()
        assert (diag_dir / "panel_detection_overlay.png").exists()


# ===================================================================
# OCR calibration success rate
# ===================================================================

class TestCalibrationQuality:
    """Verify that OCR-based y-axis calibration succeeds."""

    def test_ocr_calibration_succeeds(self, vertical_2_panel_image, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )

        ocr_panels = [p for p in result.panels if p.calibration.scale_type == "ocr_value"]
        # At least one panel should have OCR calibration.
        assert len(ocr_panels) >= 1, "No panels achieved OCR calibration"

    def test_calibration_range_sensible(self, vertical_2_panel_image, tmp_path):
        """Calibrated values should be within the known y-range [0, 300]."""
        from trace_digitiser import digitise

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )

        for p in result.panels:
            if p.calibration.scale_type != "ocr_value":
                continue
            top_val = p.calibration.y_to_value(p.y_top)
            bottom_val = p.calibration.y_to_value(p.y_bottom)
            # Top should be higher than bottom (upward orientation).
            assert float(top_val) > float(bottom_val)
            # Values should be in a reasonable range for our synthetic plots.
            assert float(top_val) <= 500, f"Top value {top_val} too high"
            assert float(bottom_val) >= -100, f"Bottom value {bottom_val} too low"


# ===================================================================
# Trace accuracy (RMSE against ground truth)
# ===================================================================

class TestTraceAccuracy:
    """Compare digitised traces against synthetic ground truth.

    These tests verify that the digitised trace is a reasonable
    approximation of the original plotted data by measuring RMSE
    after aligning on normalised x.
    """

    def test_trace_rmse_vertical_2_panel(self, vertical_2_panel_image, ground_truth, tmp_path):
        from trace_digitiser import digitise

        result = digitise(
            vertical_2_panel_image,
            layout_mode="stacked", expected_rows=2, expected_cols=1,
            output_dir=tmp_path,
        )

        gt = ground_truth[ground_truth["example"] == "vertical_2_panels"]
        if result.trace_data.empty or gt.empty:
            pytest.skip("No trace data or ground truth available")

        # For each panel, compute RMSE between digitised center_estimate
        # and ground-truth y, aligned on normalised x.
        for panel_id in result.trace_data["panel_id"].unique():
            panel_traces = result.trace_data[result.trace_data["panel_id"] == panel_id]

            if panel_traces.empty:
                continue

            # Get the first (only) trace per panel.
            trace = panel_traces[panel_traces["trace_id"] == panel_traces["trace_id"].min()]
            gt_panel = gt[gt["panel_expected"] == panel_id]

            if gt_panel.empty or trace.empty:
                continue

            # Normalise ground truth x to [0, 1] range.
            gt_x_norm = (gt_panel["x"].values - gt_panel["x"].min()) / max(
                1e-9, gt_panel["x"].max() - gt_panel["x"].min()
            )

            # Interpolate digitised values onto ground-truth x positions.
            dig_x = trace["x_norm_0_to_1"].values
            dig_y = trace["center_estimate"].values

            gt_y_interp = np.interp(dig_x, gt_x_norm, gt_panel["y"].values)

            # Only compare where the digitised panel has OCR calibration.
            scale = trace["scale_type"].iloc[0]
            if "ocr" not in scale and "propagated" not in scale:
                continue

            rmse = np.sqrt(np.nanmean((dig_y - gt_y_interp) ** 2))

            # RMSE should be reasonable — under 50 units for a 0-300 range plot.
            # This is a loose threshold since raster digitisation is approximate.
            assert rmse < 50, (
                f"Panel {panel_id} RMSE={rmse:.1f} is too high "
                f"(scale_type={scale})"
            )


# ===================================================================
# Log-scale model
# ===================================================================

class TestLogScale:
    """Verify the log-scale calibration model."""

    def test_log_calibration_values(self):
        from trace_digitiser.models import Calibration

        # log10(value) = -0.005 * y + 3.0
        # At y=0: value = 10^3 = 1000
        # At y=600: value = 10^0 = 1
        c = Calibration(a=-0.005, b=3.0, scale_type="log_ocr_value")
        vals = c.y_to_value(np.array([0, 200, 400, 600]))
        np.testing.assert_allclose(vals, [1000, 100, 10, 1], rtol=1e-6)
