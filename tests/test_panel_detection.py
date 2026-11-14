"""Tests for trace_digitiser.panel_detection."""

import pytest

from trace_digitiser.panel_detection import (
    find_plot_panels,
    infer_expected_panels,
    panel_quality,
    split_y_clusters,
)


class TestInferExpectedPanels:
    def test_explicit(self):
        assert infer_expected_panels(expected_panels=5) == 5

    def test_rows_cols(self):
        assert infer_expected_panels(expected_rows=2, expected_cols=3) == 6

    def test_single(self):
        assert infer_expected_panels(layout_mode="single") == 1

    def test_auto_unknown(self):
        assert infer_expected_panels(layout_mode="auto") is None


class TestSplitYClusters:
    def test_single_cluster(self):
        result = split_y_clusters([100, 120, 140], image_height=600)
        assert len(result) == 1

    def test_two_clusters(self):
        result = split_y_clusters([100, 120, 140, 400, 420, 440], image_height=600)
        assert len(result) == 2


class TestPanelQuality:
    def test_higher_score_better(self):
        a = {"score": 5, "x0": 0, "x1": 100, "y_top": 0, "y_bottom": 100}
        b = {"score": 3, "x0": 0, "x1": 100, "y_top": 0, "y_bottom": 100}
        assert panel_quality(a) > panel_quality(b)


@pytest.mark.skipif(
    not pytest.importorskip("cv2", reason="OpenCV required"),
    reason="OpenCV not available",
)
class TestFindPlotPanels:
    def test_vertical_2_panels(self, vertical_2_panel_image):
        from trace_digitiser.io import load_image

        rgb = load_image(vertical_2_panel_image)
        panels, _, _ = find_plot_panels(rgb, layout_mode="stacked", expected_rows=2, expected_cols=1)
        assert len(panels) == 2
        # Panels should be ordered top-to-bottom.
        assert panels[0]["y_top"] < panels[1]["y_top"]

    def test_horizontal_3_panels(self, horizontal_3_panel_image):
        from trace_digitiser.io import load_image

        rgb = load_image(horizontal_3_panel_image)
        panels, _, _ = find_plot_panels(rgb, layout_mode="horizontal", expected_rows=1, expected_cols=3)
        assert len(panels) == 3
        # Panels should be ordered left-to-right.
        assert panels[0]["x0"] < panels[1]["x0"] < panels[2]["x0"]

    def test_grid_2x2(self, grid_2x2_image):
        from trace_digitiser.io import load_image

        rgb = load_image(grid_2x2_image)
        panels, _, _ = find_plot_panels(rgb, layout_mode="grid", expected_rows=2, expected_cols=2)
        assert len(panels) == 4
