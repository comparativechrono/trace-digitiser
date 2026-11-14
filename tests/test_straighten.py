"""Tests for straightening tilted figures (trace_digitiser.straighten)."""

import json
import warnings

import numpy as np
import pytest
from PIL import Image

from trace_digitiser import digitise
from trace_digitiser.cli import main
from trace_digitiser.frames import find_frames
from trace_digitiser.io import load_image
from trace_digitiser.straighten import estimate_tilt, straighten
from trace_digitiser.validation import compare

from .conftest import needs_published_figure


def tilted(fig, degrees):
    """The rendered figure *fig* turned anticlockwise by *degrees*, as an RGB array."""
    image = Image.open(fig.path).convert("RGB")
    return np.asarray(image.rotate(degrees, resample=Image.BICUBIC, fillcolor=(255, 255, 255)))


def quiet(image, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return digitise(image, write_outputs=False, **kwargs)


class TestEstimateTilt:
    @pytest.mark.parametrize(
        "case", ["A1_box_grid_blue", "B2_L_axes_ticks_out", "C9_black_line_grid", "D_dpi60", "G3_grid_2x2",
                 "E_noise_sigma8_jpeg85", "K1_dense_72dpi"]
    )
    def test_upright_figures_are_left_alone(self, render, case):
        assert estimate_tilt(load_image(render(case).path)) == 0.0

    @needs_published_figure
    def test_published_figure_is_upright(self, published_figure):
        assert estimate_tilt(load_image(published_figure)) == 0.0

    @pytest.mark.parametrize("degrees", [0.5, 1.0, 2.0, -0.7, -3.3, 4.5])
    @pytest.mark.parametrize("case", ["A1_box_grid_blue", "B2_L_axes_ticks_out", "G1_stacked_2"])
    def test_tilt_is_measured(self, render, case, degrees):
        assert estimate_tilt(tilted(render(case), degrees)) == pytest.approx(degrees, abs=0.05)

    def test_nothing_to_judge_by(self, simple_rgb):
        assert estimate_tilt(simple_rgb) == 0.0
        speckled = simple_rgb.copy()
        speckled[np.random.default_rng(0).random(speckled.shape[:2]) > 0.98] = 0
        assert estimate_tilt(speckled) == 0.0

    def test_tilt_beyond_the_search_is_not_guessed(self, render):
        assert estimate_tilt(tilted(render("A1_box_grid_blue"), 9.0)) == 0.0


class TestStraighten:
    def test_axes_return_to_where_they_were(self, render):
        fig = render("A1_box_grid_blue")
        upright = straighten(tilted(fig, 1.5), 1.5)
        frames = find_frames(upright)
        assert len(frames) == 1
        left, top, right, bottom = fig.box()
        assert abs(frames[0].left - left) <= 1 and abs(frames[0].right - right) <= 1
        assert abs(frames[0].top - top) <= 1 and abs(frames[0].bottom - bottom) <= 1

    def test_zero_is_a_no_op(self, simple_rgb):
        assert straighten(simple_rgb, 0.0) is simple_rgb

    def test_corners_are_filled_with_the_page_colour(self):
        page = np.full((120, 200, 3), (238, 232, 213), dtype=np.uint8)  # a yellowed page
        turned = straighten(page, 3.0)
        assert turned.shape == page.shape
        assert (np.abs(turned.astype(int) - page.astype(int)) <= 1).all()


class TestDigitiseTiltedFigures:
    @pytest.mark.parametrize("degrees", [0.5, 2.0, -3.3])
    def test_a_tilted_figure_is_digitised_as_well_as_an_upright_one(self, render, degrees):
        fig = render("G1_stacked_2")
        limits = dict(y_min=0, y_max=100, x_min=0, x_max=10)
        upright, _ = compare(quiet(fig.path, **limits), fig.truth)
        result = quiet(tilted(fig, degrees), **limits)
        askew, _ = compare(result, fig.truth)
        assert askew["status"].eq("ok").all() and len(result.panels) == 2
        assert (askew["rmse_pct"] <= upright["rmse_pct"] + 0.15).all()
        assert result.provenance["straightening"]["turned_clockwise_by_degrees"] == pytest.approx(degrees, abs=0.05)
        assert any("has been straightened" in w for w in result.warnings)

    def test_without_straightening_it_fails(self, render):
        """What the option is for: left askew, the same figure is not digitised correctly."""
        fig = render("G1_stacked_2")
        result = quiet(tilted(fig, 2.0), y_min=0, y_max=100, x_min=0, x_max=10, straighten=False)
        metrics, _ = compare(result, fig.truth)
        assert not metrics["recovered"].any()

    def test_upright_figure_is_untouched(self, render):
        fig = render("A1_box_grid_blue")
        result = quiet(fig.path, y_min=0, y_max=100, x_min=0, x_max=10)
        assert result.warnings == []
        assert result.provenance["straightening"] == {"tilt_measured_degrees": 0.0, "turned_clockwise_by_degrees": 0.0}
        assert np.array_equal(result.rgb, load_image(fig.path))

    def test_positions_given_by_hand_stop_automatic_straightening(self, render):
        """Positions measured on the image as it is would no longer fit a straightened one."""
        fig = render("A1_box_grid_blue")
        image = tilted(fig, 1.0)
        result = quiet(image, plot_area=fig.box(), y_min=0, y_max=100)
        assert np.array_equal(result.rgb, image)
        assert any("tilted 1.00 degrees anticlockwise" in w and "not been straightened" in w for w in result.warnings)

    def test_angle_given_by_hand(self, render):
        fig = render("A1_box_grid_blue")
        result = quiet(tilted(fig, 1.0), straighten=1.0, plot_area=fig.box(), y_min=0, y_max=100, x_min=0, x_max=10)
        metrics, _ = compare(result, fig.truth)
        assert metrics.iloc[0]["recovered"]
        assert not any("straightened" in w for w in result.warnings)  # asked for, so nothing to report

    @pytest.mark.parametrize("value", ["sideways", 90, object()])
    def test_bad_values(self, simple_rgb, value):
        with pytest.raises(ValueError, match="straighten"):
            digitise(simple_rgb, straighten=value, write_outputs=False)

    def test_straightened_image_is_saved(self, render, tmp_path):
        fig = render("A1_box_grid_blue")
        scan = tmp_path / "scan.png"
        Image.fromarray(tilted(fig, 1.2)).save(scan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = digitise(scan, y_min=0, y_max=100, x_min=0, x_max=10, output_dir=tmp_path)
        assert result.straightened_path == tmp_path / "scan_automated_straightened.png"
        assert np.array_equal(load_image(result.straightened_path), result.rgb)
        record = json.loads(result.provenance_path.read_text())
        assert record["straightening"]["turned_clockwise_by_degrees"] == pytest.approx(1.2, abs=0.05)
        assert record["parameters"]["straighten"] == "auto"

    def test_command_line(self, render, tmp_path, capsys):
        fig = render("A1_box_grid_blue")
        scan = tmp_path / "scan.png"
        Image.fromarray(tilted(fig, 1.2)).save(scan)
        main([str(scan), "--y-min", "0", "--y-max", "100", "-o", str(tmp_path / "auto")])
        out = capsys.readouterr().out
        assert "Straightened:" in out and "CHECK: The figure was tilted 1.2" in out
        main([str(scan), "--y-min", "0", "--y-max", "100", "-o", str(tmp_path / "never"), "--straighten", "never"])
        assert "Straightened:" not in capsys.readouterr().out
        with pytest.raises(SystemExit):
            main([str(scan), "--straighten", "sideways"])
