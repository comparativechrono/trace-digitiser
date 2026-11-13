"""Tests for the validation tools (trace_digitiser.validation)."""

import numpy as np
import pytest

from trace_digitiser.validation import (
    FigureSpec,
    axis_cases,
    compare,
    default_cases,
    drawn_envelope,
    render_figure,
    round_trip,
    run_axis_validation,
    run_validation,
    summarise,
    summarise_axes,
    synthetic_activity,
)

from .conftest import needs_tesseract


class TestDrawnEnvelope:
    def test_straight_line(self):
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 100.0])
        lower, upper, mean = drawn_envelope(x, y, np.array([5.0]), 0.5)
        assert lower[0] == pytest.approx(45) and upper[0] == pytest.approx(55)
        assert np.isnan(mean[0])  # no sample falls in the window

    def test_spike_inside_window(self):
        x = np.arange(11, dtype=float)
        y = np.zeros(11)
        y[5] = 80
        lower, upper, mean = drawn_envelope(x, y, np.array([5.0, 8.0]), 0.6)
        assert upper[0] == 80 and lower[0] == pytest.approx(32)  # the spike's flanks leave the window at 32
        assert mean[0] == 80
        assert upper[1] == 0 and lower[1] == 0

    def test_dense_window_mean(self):
        x = np.linspace(0, 1, 101)
        y = np.where(np.arange(101) % 2 == 0, 10.0, 30.0)
        lower, upper, mean = drawn_envelope(x, y, np.array([0.5]), 0.1)
        assert lower[0] == 10 and upper[0] == 30
        assert mean[0] == pytest.approx(20, abs=1)


class TestRenderFigure:
    def test_truth_geometry_matches_image(self, tmp_path):
        """The recorded plot area must be where the axes really are in the image."""
        from trace_digitiser.io import load_image

        truth = render_figure(FigureSpec("geom", style="box", panel_w_px=300, panel_h_px=150), tmp_path)
        rgb = load_image(tmp_path / truth["image"])
        assert rgb.shape[:2] == (truth["height"], truth["width"])
        p = truth["panels"][0]
        dark = rgb.sum(axis=2) < 300
        # The frame's sides are the columns (rows) that are dark along the whole plot.
        inner_rows = slice(int(p["top"]) + 5, int(p["bottom"]) - 5)
        inner_cols = slice(int(p["left"]) + 5, int(p["right"]) - 5)
        cols = np.flatnonzero(dark[inner_rows, :].mean(axis=0) > 0.9)
        rows = np.flatnonzero(dark[:, inner_cols].mean(axis=1) > 0.9)
        assert abs(cols.min() - p["left"]) <= 1 and abs(cols.max() - p["right"]) <= 1
        assert abs(rows.min() - p["top"]) <= 1 and abs(rows.max() - p["bottom"]) <= 1
        assert p["right"] - p["left"] == pytest.approx(300) and p["bottom"] - p["top"] == pytest.approx(150)

    def test_custom_data_and_formats(self, tmp_path):
        x = np.linspace(0, 1, 50)
        truth = render_figure(
            FigureSpec("custom", series="custom", xlim=(0, 1), ylim=(0, 2), fmt="jpg", noise_sigma=3),
            tmp_path, data=[[(x, 1 + 0.5 * np.sin(6 * x))]],
        )
        assert truth["image"].endswith(".jpg") and (tmp_path / truth["image"]).exists()
        assert (tmp_path / "custom.truth.json").exists()
        assert truth["panels"][0]["traces"][0]["y"][0] == pytest.approx(1.0)

    def test_unknown_style(self, tmp_path):
        with pytest.raises(ValueError):
            render_figure(FigureSpec("bad", style="cubist"), tmp_path)


class TestSuite:
    def test_default_cases_are_unique_and_cover_groups(self):
        cases = default_cases()
        names = [c.name for c in cases]
        assert len(names) == len(set(names)) == 44
        assert {c.group[0] for c in cases} == set("ABCDEFGHIJKL")

    def test_run_and_summarise(self, tmp_path):
        chosen = [c for c in default_cases() if c.name in ("A1_box_grid_blue", "C9_black_line_grid", "G1_stacked_2")]
        results = run_validation(tmp_path, cases=chosen, modes=("limits",))
        assert (tmp_path / "results.csv").exists()
        assert len(results) == 4  # 1 + 1 + 2 traces
        assert results["recovered"].all()
        summary = summarise(results)
        total = summary[summary["group"] == "ALL"].iloc[0]
        assert total["traces"] == 4 and total["found"] == 4 and total["recovered"] == 4
        assert total["declined"] == 0 and total["inaccurate"] == 0
        assert total["median_rmse_pct"] < 0.3

    def test_a_failing_digitiser_is_scored_as_failing(self, tmp_path):
        """The harness must not flatter: a digitiser that returns nothing recovers nothing."""
        import pandas as pd

        from trace_digitiser.models import DigitiserResult

        def useless(path, **kwargs):
            return DigitiserResult(
                None, np.zeros((1, 1, 3), np.uint8), [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )

        def crashing(path, **kwargs):
            raise RuntimeError("boom")

        chosen = [c for c in default_cases() if c.name == "A1_box_grid_blue"]
        for fn, status in ((useless, "panel_missed"), (crashing, "crash")):
            results = run_validation(tmp_path / status, cases=chosen, modes=("limits",), digitise_fn=fn)
            assert results["status"].tolist() == [status]
            assert not results["recovered"].any()

    def test_wrong_scale_is_not_recovered(self, render):
        """A result with the right shape but the wrong axis values must not count as recovered."""
        import warnings

        from trace_digitiser import digitise

        fig = render("A1_box_grid_blue")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = digitise(fig.path, y_min=0, y_max=50, x_min=0, x_max=10, ocr=False, write_outputs=False)
        metrics, _ = compare(result, fig.truth)
        assert metrics.iloc[0]["status"] == "ok" and not metrics.iloc[0]["recovered"]
        assert metrics.iloc[0]["rmse_pct"] > 10
        metrics["mode"], metrics["seconds"] = "limits", 0.0
        total = summarise(metrics).iloc[-1]
        assert total["inaccurate"] == 1 and total["declined"] == 0

    def test_an_unread_axis_is_declined_not_inaccurate(self, render):
        """Values left uncalibrated, with a warning, are counted apart from values that are off."""
        import warnings

        from trace_digitiser import digitise

        fig = render("A1_box_grid_blue")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = digitise(fig.path, x_min=0, x_max=10, ocr=False, write_outputs=False)
        metrics, _ = compare(result, fig.truth)
        assert not metrics.iloc[0]["recovered"]
        metrics["mode"], metrics["seconds"] = "limits", 0.0
        total = summarise(metrics).iloc[-1]
        assert total["found"] == 1 and total["declined"] == 1 and total["inaccurate"] == 0
        assert np.isnan(total["median_rmse_pct"])


class TestAxisReading:
    def test_cases(self):
        cases = axis_cases()
        names = [c.name for c in cases]
        assert len(names) == len(set(names)) == 375
        # Only typefaces that ship with matplotlib, so that the figures are the same on every machine.
        assert {c.font for c in cases} == {"DejaVu Sans", "DejaVu Serif", "STIXGeneral"}
        assert {c.yscale for c in cases} == {"linear", "log"}

    @needs_tesseract
    def test_run_and_summarise(self, tmp_path):
        chosen = [c for c in axis_cases() if c.group == "sans_150dpi_9pt"][:6]
        results = run_axis_validation(tmp_path, cases=chosen)
        assert (tmp_path / "axis_results.csv").exists()
        assert len(results) == 12  # a y axis and an x axis for each figure
        assert (results["outcome"] == "right").all(), results[["case", "axis", "outcome"]]
        assert (results["confidence"] == "high").all()
        total = summarise_axes(results).iloc[-1]
        assert total["axes"] == 12 and total["right_high"] == 12 and total["wrong_high"] == 0

    def test_summary_counts_each_kind_of_outcome(self):
        import pandas as pd

        results = pd.DataFrame(
            {
                "outcome": ["right", "right", "declined", "wrong", "wrong", "crash"],
                "confidence": ["high", "medium", None, "medium", "high", None],
                "small_print": [False, True, True, True, False, False],
            }
        )
        summary = summarise_axes(results).set_index("label height")
        assert summary.loc["all"].tolist() == [6, 1, 1, 1, 1, 1, 1]
        assert summary.loc["under 8 px", ["axes", "right_flagged", "declined", "wrong_flagged"]].tolist() == [
            3,
            1,
            1,
            1,
        ]
        assert summary.loc["8 px or more", ["axes", "right_high", "wrong_high", "failed"]].tolist() == [3, 1, 1, 1]


class TestRoundTrip:
    def test_smooth_series(self, tmp_path):
        x = np.linspace(0, 24, 200)
        y = 5 + 3 * np.sin(2 * np.pi * x / 24)
        out = round_trip(x, y, tmp_path, name="sine", ylim=(0, 10))
        row = out["metrics"].iloc[0]
        assert row["status"] == "ok" and row["recovered"]
        assert row["rmse_pct"] < 0.5
        assert out["image"].exists() and out["overlay"].exists()
        assert (tmp_path / "sine_digitised_vs_truth.csv").exists()
        cols = out["columns"]
        assert np.abs(cols["center_estimate"] - cols["truth_center"]).max() < 0.2  # of a 0-10 axis

    def test_dense_series_reports_bin_mean_bias(self, tmp_path):
        """A week of minute data in a 348-pixel plot: the result tracks the data but is not its mean."""
        t, activity = synthetic_activity(days=7, seed=3)
        out = round_trip(
            t, 300 * activity, tmp_path, name="dense", xlim=(0, 7), ylim=(0, 300),
            panel_px=(348, 137), dpi=72, linewidth=0.6, style="paper", fmt="jpg", fontsize=6.5,
            yticks=(0, 50, 100, 150, 200, 250, 300), colour="#578dd2",
        )
        row = out["metrics"].iloc[0]
        assert row["status"] == "ok"
        assert row["corr_bin_mean"] > 0.85  # follows the ups and downs ...
        assert row["bias_bin_mean_pct"] > 2  # ... but sits above the mean activity

    @pytest.mark.parametrize(
        "x,y",
        [([0, 1], [1]), ([0, 0, 1], [1, 2, 3]), ([0, 1, 2], [1, np.nan, 3]), ([2, 1, 0], [1, 2, 3])],
    )
    def test_bad_input(self, tmp_path, x, y):
        with pytest.raises(ValueError):
            round_trip(x, y, tmp_path)


def test_synthetic_activity_is_reproducible_and_bounded():
    t1, a1 = synthetic_activity(days=2, seed=5)
    t2, a2 = synthetic_activity(days=2, seed=5)
    np.testing.assert_array_equal(a1, a2)
    assert len(t1) == 2 * 24 * 60 and t1[0] == 0 and t1[-1] < 2
    assert a1.min() >= 0 and a1.max() <= 1
    first_half = (t1 % 1) < 0.5  # "night" comes first
    assert a1[first_half].mean() > 2 * a1[~first_half].mean()
