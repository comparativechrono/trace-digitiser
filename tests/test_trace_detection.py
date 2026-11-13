"""Tests for trace_digitiser.trace_detection."""

import numpy as np

from trace_digitiser.trace_detection import circular_hue_groups, hue_mask_from_group


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


class TestHueMask:
    def test_basic_mask(self):
        H = np.array([[0.1, 0.5], [0.9, 0.3]])
        group = {"bins": [3, 4], "lo": 3 / 36, "hi": 5 / 36, "n_bins": 36}
        mask = hue_mask_from_group(H, group)
        assert mask.shape == H.shape
        assert mask.dtype == bool
