"""Tests for trace_digitiser.geometry."""

from trace_digitiser.geometry import box_area, box_iou, contains_box, merge_contiguous


class TestMergeContiguous:
    def test_empty(self):
        assert merge_contiguous([]) == []

    def test_single(self):
        assert merge_contiguous([5]) == [(5, 5)]

    def test_two_groups(self):
        assert merge_contiguous([1, 2, 3, 10, 11]) == [(1, 3), (10, 11)]

    def test_custom_gap(self):
        assert merge_contiguous([1, 2, 5, 6], max_gap=3) == [(1, 6)]

    def test_unsorted_input(self):
        assert merge_contiguous([11, 3, 1, 2, 10]) == [(1, 3), (10, 11)]


class TestBoxHelpers:
    def test_area(self):
        p = {"x0": 10, "x1": 110, "y_top": 20, "y_bottom": 70}
        assert box_area(p) == 100 * 50

    def test_area_degenerate(self):
        p = {"x0": 10, "x1": 5, "y_top": 20, "y_bottom": 70}
        assert box_area(p) == 0

    def test_iou_identical(self):
        p = {"x0": 0, "x1": 100, "y_top": 0, "y_bottom": 100}
        assert box_iou(p, p) == 1.0

    def test_iou_no_overlap(self):
        a = {"x0": 0, "x1": 50, "y_top": 0, "y_bottom": 50}
        b = {"x0": 100, "x1": 150, "y_top": 100, "y_bottom": 150}
        assert box_iou(a, b) == 0.0

    def test_contains(self):
        outer = {"x0": 0, "x1": 200, "y_top": 0, "y_bottom": 200}
        inner = {"x0": 10, "x1": 190, "y_top": 10, "y_bottom": 190}
        assert contains_box(outer, inner) is True
        assert contains_box(inner, outer) is False
