"""Tests for the digitise() pipeline that do not need OCR.

Axis values are supplied, as a careful user would; accuracy is measured
against the data the figure was drawn from (trace_digitiser.validation).
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from trace_digitiser import DigitiserWarning, digitise
from trace_digitiser.io import load_image
from trace_digitiser.validation import compare


def run(fig, **kwargs):
    """Digitise a rendered figure with its axis limits supplied and no files written."""
    spec = fig.spec
    options = dict(x_min=spec.xlim[0], x_max=spec.xlim[1], ocr=False, write_outputs=False)
    if spec.yscale == "linear":
        options.update(y_min=spec.ylim[0], y_max=spec.ylim[1])
    options.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DigitiserWarning)
        return digitise(fig.path, **options)


class TestAccuracy:
    """Error against known data, in percent of the y-axis range."""

    @pytest.mark.parametrize(
        "case,limit",
        [
            ("A1_box_grid_blue", 0.3),
            ("A4_box_grid_negative", 0.3),
            ("B1_box_nogrid_ticks_out", 0.3),  # 1.x: 1.1 % (plot area widened by tick marks)
            ("B2_L_axes_ticks_out", 0.4),  # 1.x: plot not found
            ("C9_black_line_grid", 0.4),  # 1.x: trace not found
            ("C11_grey_line", 0.4),  # 1.x: trace not found
            ("C12_pastel_pink", 0.3),
            ("D_dpi60", 1.0),  # 1.x: trace not found
            ("D_dpi300", 0.3),
            ("E_jpeg_q30", 0.4),
            ("E_noise_sigma8_jpeg85", 0.4),
            ("E_blur1_jpeg80", 0.4),
            ("F_linewidth_0p5", 0.3),  # 1.x: trace not found
            ("F_linewidth_2p5", 0.3),
        ],
    )
    def test_single_trace(self, render, case, limit):
        fig = render(case)
        metrics, _ = compare(run(fig), fig.truth)
        row = metrics.iloc[0]
        assert row["status"] == "ok"
        assert row["n_traces_found"] == 1
        assert row["rmse_pct"] <= limit
        assert abs(row["x_err_left_pct"]) <= 0.25 and abs(row["x_err_right_pct"]) <= 0.25
        assert row["corr"] > 0.999

    @pytest.mark.parametrize(
        "case", ["G1_stacked_2", "G2_horizontal_3", "G3_grid_2x2", "H1_two_traces_blue_red", "H2_three_traces"]
    )
    def test_panels_and_traces(self, render, case):
        fig = render(case)
        metrics, _ = compare(run(fig), fig.truth)
        assert (metrics["status"] == "ok").all()
        assert (metrics["n_traces_found"] == metrics["n_traces_true"]).all()
        assert (metrics["rmse_pct"] <= 0.4).all()

    def test_log_axis_with_limits(self, render):
        fig = render("I1_log_y")
        result = run(fig, y_min=1, y_max=1000, y_scale="log")
        assert result.panels[0].calibration.is_log
        metrics, _ = compare(result, fig.truth)
        assert metrics.iloc[0]["rmse_pct"] <= 0.4

    def test_dense_record_envelopes(self, render):
        """Many samples per pixel column: the envelopes and their midpoint are what can be recovered."""
        fig = render("K2_dense_144dpi")
        metrics, columns = compare(run(fig), fig.truth)
        assert (metrics["status"] == "ok").all()
        assert (metrics["rmse_lower_pct"] <= 1.5).all()
        assert (metrics["corr"] > 0.9).all()
        # The centre estimate is not the mean of the samples: for bursty,
        # skewed data it lies well above it.  Tested so that nobody "fixes"
        # the documentation.
        assert (metrics["bias_bin_mean_pct"] > 1.0).all()
        assert (columns["center_estimate"] >= columns["lower_envelope"] - 1e-9).all()
        assert (columns["center_estimate"] <= columns["upper_envelope"] + 1e-9).all()


class TestManualSettings:
    def test_plot_area_by_hand(self, render):
        fig = render("A1_box_grid_blue")
        result = run(fig, plot_area=fig.box())
        assert result.panels[0].source == "user_plot_area"
        metrics, _ = compare(result, fig.truth)
        assert metrics.iloc[0]["rmse_pct"] <= 0.3

    def test_plot_area_list_for_two_panels(self, render):
        fig = render("G1_stacked_2")
        result = run(fig, plot_area=[fig.box(0), fig.box(1)])
        assert len(result.panels) == 2 and result.n_traces == 2

    def test_reference_points(self, render):
        fig = render("A1_box_grid_blue")
        p = fig.panel
        mid_row = 0.5 * (p["top"] + p["bottom"])
        mid_col = 0.5 * (p["left"] + p["right"])
        result = run(
            fig, y_min=None, y_max=None, x_min=None, x_max=None,
            y_ref=[(p["bottom"], 0), (mid_row, 50)], x_ref=[(p["left"], 0), (mid_col, 5)],
        )
        cal = result.panels[0].calibration
        assert cal.scale_type == "user_reference"
        metrics, _ = compare(result, fig.truth)
        assert metrics.iloc[0]["rmse_pct"] <= 0.3

    def test_reference_points_take_precedence_over_limits(self, render):
        fig = render("A1_box_grid_blue")
        p = fig.panel
        result = run(fig, y_min=0, y_max=1, y_ref=[(p["bottom"], 0), (p["top"], 100)])
        assert float(result.panels[0].calibration.y_to_value(p["top"])) == pytest.approx(100)

    def test_reference_points_need_panel_numbers_when_several_panels(self, render):
        fig = render("G1_stacked_2")
        with pytest.raises(ValueError, match="dict keyed by panel number"):
            run(fig, y_ref=[(100, 0), (50, 10)])
        p1, p2 = fig.truth["panels"]
        result = run(
            fig,
            y_min=None,
            y_max=None,
            y_ref={1: [(p1["bottom"], 0), (p1["top"], 100)], 2: [(p2["bottom"], 0), (p2["top"], 100)]},
        )
        assert all(p.calibration.scale_type == "user_reference" for p in result.panels)

    def test_trace_colours(self, render):
        fig = render("H1_two_traces_blue_red")
        result = run(fig, trace_colours=["#d62728"])
        assert result.n_traces == 1

    def test_exclude(self, render):
        fig = render("A1_box_grid_blue")
        left, top, right, bottom = fig.box()
        result = run(fig, exclude=(left + 100, top, left + 200, bottom))
        gap = result.trace_data[(result.trace_data.x_pixel > left + 105) & (result.trace_data.x_pixel < left + 195)]
        assert gap["interpolated"].all() and (gap["n_pixels"] == 0).all()


class TestOutputs:
    def test_columns(self, render):
        data = run(render("A1_box_grid_blue")).trace_data
        expected = {
            "panel_id", "trace_id", "x_pixel", "x_norm_0_to_1", "x_value", "center_estimate", "mean_estimate",
            "upper_envelope", "lower_envelope", "scale_type", "trace_median_rgb", "trace_pixel_count",
            "n_pixels", "interpolated",
        }
        assert expected.issubset(data.columns)
        assert data["x_value"].is_monotonic_increasing
        assert data[["center_estimate", "upper_envelope", "lower_envelope"]].notna().all().all()
        assert (data["upper_envelope"] >= data["lower_envelope"]).all()

    def test_x_positions_use_subpixel_edges(self, render):
        fig = render("A1_box_grid_blue")
        data = run(fig).trace_data
        p = fig.panel
        expected = (data["x_pixel"] - p["left"]) / (p["right"] - p["left"]) * 10
        assert np.abs(data["x_value"] - expected).max() < 0.02

    def test_nothing_written_when_asked(self, render, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run(render("A1_box_grid_blue"))
        assert list(tmp_path.iterdir()) == []
        assert result.trace_csv_path is None and result.provenance_path is None

    def test_files_written(self, render, tmp_path):
        result = run(render("A1_box_grid_blue"), write_outputs=True, output_dir=tmp_path, save_overlay=True)
        for path in (result.trace_csv_path, result.metadata_csv_path, result.provenance_path, result.overlay_path):
            assert path is not None and path.exists() and path.parent == tmp_path
        assert len(pd.read_csv(result.trace_csv_path)) == len(result.trace_data)
        meta = pd.read_csv(result.metadata_csv_path)
        assert {"plot_left", "plot_bottom", "y_calibration_method", "axes"}.issubset(meta.columns)
        overlay = load_image(result.overlay_path)
        assert overlay.shape[0] >= result.rgb.shape[0]

    def test_diagnostics(self, render, tmp_path):
        result = run(render("A1_box_grid_blue"), write_outputs=True, output_dir=tmp_path, save_diagnostics=True)
        assert (tmp_path / "diagnostics" / "panel_detection_overlay.png").exists()
        assert (tmp_path / "diagnostics" / "trace_mask_panel_1_trace_1.png").exists()
        assert result.overlay_path.exists()

    def test_provenance(self, render, tmp_path):
        fig = render("A1_box_grid_blue")
        result = run(fig, write_outputs=True, output_dir=tmp_path)
        record = json.loads(result.provenance_path.read_text())
        assert record == json.loads(json.dumps(result.provenance))
        assert len(record["image"]["sha256"]) == 64
        assert record["image"]["width"] == fig.truth["width"]
        assert record["parameters"]["y_max"] == fig.spec.ylim[1]
        assert record["panels"][0]["y_calibration"]["method"] == "axis_limits"
        assert record["panels"][0]["traces"][0]["columns"] == len(result.trace_data)
        assert record["software"]["numpy"]
        assert record["trace_digitiser_version"]

    def test_image_array_input(self, render):
        fig = render("A1_box_grid_blue")
        spec = fig.spec
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DigitiserWarning)
            result = digitise(
                load_image(fig.path), y_min=spec.ylim[0], y_max=spec.ylim[1], ocr=False, write_outputs=False
            )
        assert result.image_path is None
        pd.testing.assert_frame_equal(
            result.trace_data.drop(columns=["x_value"], errors="ignore"),
            run(fig).trace_data.drop(columns=["x_value"]),
        )

    def test_overlay_method(self, render, tmp_path):
        result = run(render("A1_box_grid_blue"))
        image = result.overlay(tmp_path / "check.png")
        assert (tmp_path / "check.png").exists()
        assert image.ndim == 3 and not np.array_equal(image, np.full_like(image, 255))

    def test_plot_and_resample(self, render, tmp_path):
        result = run(render("A1_box_grid_blue"))
        assert result.plot(output_path=tmp_path / "plot.png", show=False) is not None
        assert (tmp_path / "plot.png").exists()
        resampled = result.resample(interval=0.5)
        assert 15 <= len(resampled) <= 25  # ten x units in steps of a half


class TestWarnings:
    def test_clean_run_has_no_warnings(self, render):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DigitiserWarning)
            result = digitise(render("A1_box_grid_blue").path, y_min=0, y_max=100, x_min=0, x_max=10, ocr=False,
                              write_outputs=False)
        assert result.warnings == []

    def test_uncalibrated_axes_are_reported(self, render):
        with pytest.warns(DigitiserWarning):
            result = digitise(render("A1_box_grid_blue").path, ocr=False, write_outputs=False)
        assert any("y axis is not calibrated" in w for w in result.warnings)
        assert any("x axis" in w and "not calibrated" in w for w in result.warnings)
        cal = result.panels[0].calibration
        assert not cal.is_calibrated
        assert result.trace_data["center_estimate"].between(-0.05, 1.05).all()
        assert "x_value" not in result.trace_data

    def test_open_axes_with_limits_are_flagged(self, render):
        with pytest.warns(DigitiserWarning, match="no line along the top"):
            result = digitise(render("B2_L_axes_ticks_out").path, y_min=0, y_max=100, x_min=0, x_max=10, ocr=False,
                              write_outputs=False)
        assert any("right-hand" in w for w in result.warnings)

    def test_no_plot_found(self, simple_rgb):
        with pytest.warns(DigitiserWarning, match="No plot area"):
            result = digitise(simple_rgb, ocr=False, write_outputs=False)
        assert result.panels == [] and result.trace_data.empty and result.n_traces == 0

    def test_no_trace_found(self):
        frame_only = np.full((200, 300, 3), 255, dtype=np.uint8)
        frame_only[30, 40:261] = frame_only[170, 40:261] = 0
        frame_only[30:171, 40] = frame_only[30:171, 260] = 0
        with pytest.warns(DigitiserWarning, match="no trace was found"):
            result = digitise(frame_only, y_min=0, y_max=1, x_min=0, x_max=1, ocr=False, write_outputs=False)
        assert len(result.panels) == 1 and result.n_traces == 0


class TestArguments:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"y_min": 0},  # y_max missing
            {"x_max": 5},
            {"y_scale": "cubic"},
            {"layout_mode": "diagonal"},
            {"achromatic": "sometimes"},
            {"plot_area": (1, 2, 3)},
            {"y_ref": [(100, 5)]},
        ],
    )
    def test_bad_arguments(self, render, kwargs):
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DigitiserWarning)
                digitise(render("A1_box_grid_blue").path, ocr=False, write_outputs=False, **kwargs)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            digitise(tmp_path / "nope.png", write_outputs=False)

    def test_bad_array(self):
        with pytest.raises(ValueError):
            digitise(np.zeros((10, 10)), write_outputs=False)

    def test_total_days_shorthand(self, render):
        result = run(render("A1_box_grid_blue"), x_min=None, x_max=None, total_days=7)
        assert result.trace_data["x_value"].max() == pytest.approx(7, abs=0.05)
