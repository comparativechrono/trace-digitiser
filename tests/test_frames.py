"""Tests for plot-area detection from axis lines (trace_digitiser.frames)."""

import numpy as np
import pytest

from trace_digitiser.frames import darkness_map, detect_axis_lines, find_frames
from trace_digitiser.io import load_image
from trace_digitiser.panel_detection import find_plot_panels

from .conftest import needs_published_figure


def _errors(frame, box):
    left, top, right, bottom = box
    return [frame.left - left, frame.top - top, frame.right - right, frame.bottom - bottom]


class TestDarknessMap:
    def test_white_page_has_no_ink(self, simple_rgb):
        assert darkness_map(simple_rgb).max() == 0

    def test_grey_paper_is_background(self):
        # An off-white scan: the paper itself must not count as ink.
        page = np.full((200, 300, 3), 215, dtype=np.uint8)
        page[100, 50:250] = 0
        dark = darkness_map(page)
        assert dark[20, 20] < 5
        assert dark[100, 150] > 150


class TestFindFrames:
    @pytest.mark.parametrize(
        "case",
        [
            "A1_box_grid_blue",
            "B1_box_nogrid_ticks_out",
            "B3_box_nogrid_ticks_in",
            "D_dpi60",
            "D_dpi300",
            "E_jpeg_q30",
            "E_noise_sigma8_jpeg85",
            "E_blur1_jpeg80",
        ],
    )
    def test_single_frame_subpixel(self, render, case):
        fig = render(case)
        frames = find_frames(load_image(fig.path))
        assert len(frames) == 1
        assert frames[0].kind == "box"
        # Within 0.6 px on every side (matplotlib snaps axis lines to the pixel grid).
        assert max(abs(e) for e in _errors(frames[0], fig.box())) <= 0.6

    def test_tick_marks_do_not_widen_the_plot_area(self, render):
        """Outward ticks at the corners lengthen the axis lines; the plot area must not grow."""
        fig = render("B1_box_nogrid_ticks_out")
        frame = find_frames(load_image(fig.path))[0]
        assert abs(frame.left - fig.box()[0]) <= 0.6  # 1.x placed this edge a tick length too far left

    def test_l_shaped_axes(self, render):
        fig = render("B2_L_axes_ticks_out")
        frames = find_frames(load_image(fig.path))
        assert len(frames) == 1
        frame = frames[0]
        assert frame.kind == "L"
        assert not frame.has_top and not frame.has_right
        assert max(abs(e) for e in _errors(frame, fig.box())) <= 1.0

    @pytest.mark.parametrize("case,n", [("G1_stacked_2", 2), ("G2_horizontal_3", 3), ("G3_grid_2x2", 4)])
    def test_multi_panel_reading_order(self, render, case, n):
        fig = render(case)
        frames = find_frames(load_image(fig.path))
        assert len(frames) == n
        for i, frame in enumerate(frames):
            assert max(abs(e) for e in _errors(frame, fig.box(i))) <= 0.6

    def test_axis_under_dense_trace(self, render):
        """A dense trace resting on the x axis hides the axis line; the frame must still be found."""
        fig = render("K1_dense_72dpi")
        frames = find_frames(load_image(fig.path))
        assert len(frames) == 2
        for i, frame in enumerate(frames):
            assert max(abs(e) for e in _errors(frame, fig.box(i))) <= 1.2

    def test_gridlines_and_ticks_located(self, render):
        fig = render("A1_box_grid_blue")
        frame = find_frames(load_image(fig.path))[0]
        p = fig.panel
        height = p["bottom"] - p["top"]
        expected = [p["top"] + height * f for f in (0.25, 0.5, 0.75)]  # y ticks at 75, 50, 25
        for row in expected:
            assert min(abs(g - row) for g in frame.gridlines_h) <= 1.0
        assert len(frame.ticks_y) >= 3

    def test_gridline_beside_the_frame_is_not_the_frame(self, render):
        """Gridlines at ticks just inside each end of the axes must not be taken for the axes."""
        from trace_digitiser.validation import FigureSpec

        fig = render(
            FigureSpec("gridline_near_edges", xlim=(0, 14), xticks=tuple(np.arange(0.5, 14, 1.0)),
                       ylim=(0, 300), yticks=(8, 100, 200, 292), panel_w_px=445, panel_h_px=300)
        )
        frames = find_frames(load_image(fig.path))
        assert len(frames) == 1
        assert max(abs(e) for e in _errors(frames[0], fig.box())) <= 0.6

    @pytest.mark.parametrize("name", ["horizontal_3_panels_noisy.jpg", "grid_2x2_panels_clean.png"])
    def test_gridline_under_a_trace_is_still_found(self, test_figures_dir, name):
        """A trace that keeps crossing a gridline hides much of it; it is judged on the part in view."""
        frames = find_frames(load_image(test_figures_dir / name))
        assert len(frames) in (3, 4)
        for frame in frames:
            # y ticks at 0, 100, 200, 300: two gridlines between the axes, a third of the height apart.
            assert len(frame.gridlines_h) == 2
            third = (frame.bottom - frame.top) / 3
            assert abs(frame.gridlines_h[0] - (frame.top + third)) <= 1.5
            assert abs(frame.gridlines_h[1] - (frame.top + 2 * third)) <= 1.5
            assert len(frame.gridlines_v) == 14

    def test_blank_image_has_no_frames(self, simple_rgb):
        assert find_frames(simple_rgb) == []

    def test_border_around_figure_is_not_a_plot(self):
        """A box drawn around the whole figure is dropped in favour of the plot inside it."""
        img = np.full((300, 400, 3), 255, dtype=np.uint8)
        for a, b, c, d in ((5, 5, 395, 295), (60, 40, 360, 240)):  # outer border, inner frame
            img[b, a : c + 1] = img[d, a : c + 1] = 0
            img[b : d + 1, a] = img[b : d + 1, c] = 0
        frames = find_frames(img)
        assert len(frames) == 1
        assert abs(frames[0].left - 60) <= 0.5 and abs(frames[0].bottom - 240) <= 0.5

    def test_coloured_lines_are_not_axes(self):
        """A long flat coloured line is a trace, not an axis."""
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        img[100, 20:280] = (30, 90, 200)
        horizontal, _ = detect_axis_lines(img)
        assert horizontal == []


class TestFindPlotPanels:
    def test_no_hint_needed(self, render):
        fig = render("G3_grid_2x2")
        panels, _, _ = find_plot_panels(load_image(fig.path))
        assert [p["panel_id"] for p in panels] == [1, 2, 3, 4]
        assert all(p["source"] == "axis_lines" for p in panels)

    def test_single_picks_largest(self, render):
        fig = render("G1_stacked_2")
        panels, _, _ = find_plot_panels(load_image(fig.path), layout_mode="single")
        assert len(panels) == 1

    def test_count_mismatch_is_reported(self, render):
        fig = render("A1_box_grid_blue")
        notes: list[str] = []
        panels, _, _ = find_plot_panels(load_image(fig.path), expected_panels=3, notes=notes)
        assert len(panels) == 1
        assert any("Expected 3" in n for n in notes)

    def test_nothing_found_is_reported(self, simple_rgb):
        notes: list[str] = []
        panels, _, _ = find_plot_panels(simple_rgb, notes=notes)
        assert panels == []
        assert any("plot_area" in n for n in notes)

    def test_user_plot_area(self, render):
        fig = render("A1_box_grid_blue")
        box = fig.box()
        panels, _, _ = find_plot_panels(load_image(fig.path), plot_area=[box])
        assert len(panels) == 1
        assert panels[0]["source"] == "user_plot_area"
        assert panels[0]["edges"] == pytest.approx(
            {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]}
        )

    @pytest.mark.parametrize("bad", [(10, 10, 5, 50), (0, 0, 10_000, 50), (10, 10, 50)])
    def test_bad_plot_area_rejected(self, simple_rgb, bad):
        with pytest.raises(ValueError):
            find_plot_panels(simple_rgb, plot_area=[bad])

    def test_invalid_layout_mode(self, simple_rgb):
        with pytest.raises(ValueError):
            find_plot_panels(simple_rgb, layout_mode="diagonal")

    @needs_published_figure
    def test_published_figure_frames(self, published_figure):
        """Edges measured by hand from the pixel values of the published image."""
        panels, _, _ = find_plot_panels(load_image(published_figure))
        assert len(panels) == 2
        expected = [(40.5, 22.6, 388.2, 159.3), (40.5, 218.4, 388.3, 354.8)]
        for panel, (left, top, right, bottom) in zip(panels, expected):
            e = panel["edges"]
            assert abs(e["left"] - left) <= 0.7
            assert abs(e["top"] - top) <= 0.7
            assert abs(e["right"] - right) <= 0.7
            assert abs(e["bottom"] - bottom) <= 0.7
