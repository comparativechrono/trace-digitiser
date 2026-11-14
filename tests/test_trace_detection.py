"""Tests for trace_digitiser.trace_detection."""

import numpy as np
import pytest

from trace_digitiser.io import load_image
from trace_digitiser.panel_detection import find_plot_panels
from trace_digitiser.trace_detection import (
    circular_hue_groups,
    detect_trace_masks,
    hue_mask_from_group,
    parse_colour,
)


class TestCircularHueGroups:
    def test_empty(self):
        assert circular_hue_groups(np.array([])) == []

    def test_single_cluster(self):
        # 500 pixels all around hue=0.3 (green-ish)
        hues = np.random.default_rng(0).normal(0.3, 0.02, 500).clip(0, 1)
        groups = circular_hue_groups(hues, min_pixels=50)
        assert len(groups) >= 1

    def test_two_clusters(self):
        rng = np.random.default_rng(0)
        # Blue-ish and red-ish clusters
        hues = np.concatenate([
            rng.normal(0.6, 0.02, 300).clip(0, 1),
            rng.normal(0.0, 0.02, 300).clip(0, 1),
        ])
        groups = circular_hue_groups(hues, min_pixels=50)
        assert len(groups) >= 2

    def test_red_wraps_around(self):
        # Reds either side of hue 0 are one colour.
        rng = np.random.default_rng(1)
        hues = np.concatenate([rng.uniform(0.0, 0.02, 300), rng.uniform(0.98, 1.0, 300)])
        assert len(circular_hue_groups(hues, min_pixels=50)) == 1


class TestHueMask:
    def test_basic_mask(self):
        H = np.array([[0.1, 0.5], [0.9, 0.3]])
        group = {"bins": [3, 4], "lo": 3 / 36, "hi": 5 / 36, "n_bins": 36}
        mask = hue_mask_from_group(H, group)
        assert mask.shape == H.shape
        assert mask.dtype == bool
        assert mask.tolist() == [[True, False], [False, False]]


class TestParseColour:
    @pytest.mark.parametrize(
        "colour,expected",
        [
            ("#578dd2", (87, 141, 210)),
            ("578DD2", (87, 141, 210)),
            ("#fff", (255, 255, 255)),
            ((1, 2, 3), (1, 2, 3)),
            ([0, 0, 0], (0, 0, 0)),
            ("black", (0, 0, 0)),  # colour names, as matplotlib knows them
            ("steelblue", (70, 130, 180)),
            ("tab:blue", (31, 119, 180)),
        ],
    )
    def test_accepts(self, colour, expected):
        assert parse_colour(colour) == expected

    @pytest.mark.parametrize("colour", ["blueish", "#12345", "#gggggg", (1, 2), (0, 0, 300)])
    def test_rejects(self, colour):
        with pytest.raises(ValueError):
            parse_colour(colour)


def traces_in(fig, **kwargs):
    rgb = load_image(fig.path)
    panel = find_plot_panels(rgb)[0][0]
    return rgb, panel, detect_trace_masks(rgb, panel, **kwargs)


def column_coverage(mask) -> float:
    return float(mask.any(axis=0).mean())


class TestDetectTraceMasks:
    @pytest.mark.parametrize(
        "case",
        ["A1_box_grid_blue", "C2_colour_d62728", "C6_colour_17becf", "C12_pastel_pink", "E_jpeg_q30", "D_dpi60"],
    )
    def test_one_coloured_trace_and_nothing_else(self, render, case):
        _, _, masks = traces_in(render(case))
        assert len(masks) == 1  # 1.x reported a second, all-white "trace" in about half of such figures
        assert masks[0]["kind"] == "colour"
        assert column_coverage(masks[0]["mask"]) > 0.95

    def test_trace_colour_is_reported(self, render):
        fig = render("A1_box_grid_blue")
        _, _, masks = traces_in(fig)
        assert np.abs(np.array(masks[0]["median_rgb"]) - np.array(fig.panel["traces"][0]["rgb"])).max() <= 40

    def test_one_pixel_line_survives(self, render):
        """A 0.5 pt line is one pixel wide; nothing may erode it away."""
        _, _, masks = traces_in(render("F_linewidth_0p5"))
        assert len(masks) == 1
        assert column_coverage(masks[0]["mask"]) > 0.95

    @pytest.mark.parametrize("case", ["C9_black_line_grid", "C10_black_line_nogrid", "C11_grey_line"])
    def test_black_and_grey_traces(self, render, case):
        _, _, masks = traces_in(render(case))
        assert len(masks) == 1
        assert masks[0]["kind"] == "achromatic"
        assert column_coverage(masks[0]["mask"]) > 0.9

    def test_gridlines_are_not_a_trace(self, render):
        """Grey gridlines and a frame, with the coloured trace ignored: no black trace may be invented."""
        fig = render("A1_box_grid_blue")
        rgb = load_image(fig.path)
        panel = find_plot_panels(rgb)[0][0]
        masks = detect_trace_masks(rgb, panel, colours=["#000000"])
        assert masks == []

    @pytest.mark.parametrize("case,n", [("H1_two_traces_blue_red", 2), ("H2_three_traces", 3)])
    def test_several_colours(self, render, case, n):
        _, _, masks = traces_in(render(case))
        assert len(masks) == n
        assert all(column_coverage(m["mask"]) > 0.9 for m in masks)

    def test_explicit_colours_select_and_order(self, render):
        fig = render("H1_two_traces_blue_red")
        _, _, masks = traces_in(fig, colours=["#d62728", "#1f77b4"])
        assert [m["trace_id"] for m in masks] == [1, 2]
        assert masks[0]["median_rgb"][0] > masks[0]["median_rgb"][2]  # first is the red one
        assert masks[1]["median_rgb"][2] > masks[1]["median_rgb"][0]
        _, _, only_red = traces_in(fig, colours=[(214, 39, 40)])
        assert len(only_red) == 1

    def test_exclude_region(self, render):
        fig = render("A1_box_grid_blue")
        left, top, right, bottom = fig.box()
        middle = (left + 0.4 * (right - left), top, left + 0.6 * (right - left), bottom)
        _, panel, masks = traces_in(fig, exclude=[middle])
        cols = np.flatnonzero(masks[0]["mask"].any(axis=0)) + panel["x0"]
        assert not ((cols > middle[0] + 1) & (cols < middle[2] - 1)).any()
        assert (cols < middle[0]).any() and (cols > middle[2]).any()

    def test_achromatic_never(self, render):
        _, _, masks = traces_in(render("C9_black_line_grid"), achromatic="never")
        assert masks == []

    def test_coloured_trace_followed_down_to_the_axis(self, render):
        """A dense trace resting on the x axis must not leave empty columns at the bottom."""
        fig = render("K1_dense_72dpi")
        _, _, masks = traces_in(fig)
        assert len(masks) == 1
        assert column_coverage(masks[0]["mask"]) > 0.97

    def test_frame_is_not_trace(self, render):
        """No mask pixel may lie on the axis lines of a black-trace plot."""
        fig = render("C10_black_line_nogrid")
        _, panel, masks = traces_in(fig)
        mask = masks[0]["mask"]
        assert not mask[:, 0].any() and not mask[-1, :].any() and not mask[0, :].any() and not mask[:, -1].any()

    def test_bad_arguments(self, render):
        fig = render("A1_box_grid_blue")
        rgb = load_image(fig.path)
        panel = find_plot_panels(rgb)[0][0]
        with pytest.raises(ValueError):
            detect_trace_masks(rgb, panel, achromatic="sometimes")
        with pytest.raises(ValueError):
            detect_trace_masks(rgb, panel, exclude=[(1, 2, 3)])
