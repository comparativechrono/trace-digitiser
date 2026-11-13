"""Tests for image loading and output helpers."""

import numpy as np
import pytest
from PIL import Image

from trace_digitiser.io import build_panel_metadata, load_image
from trace_digitiser.provenance import _plain, sha256_file


class TestLoadImage:
    def test_rgb(self, tmp_path):
        Image.new("RGB", (30, 20), (10, 20, 30)).save(tmp_path / "a.png")
        img = load_image(tmp_path / "a.png")
        assert img.shape == (20, 30, 3) and img.dtype == np.uint8
        assert img[0, 0].tolist() == [10, 20, 30]

    def test_transparent_background_becomes_white(self, tmp_path):
        """A figure saved with a transparent background must not load as black."""
        rgba = np.zeros((20, 30, 4), dtype=np.uint8)  # black, fully transparent
        rgba[5, :, :] = (255, 0, 0, 255)  # one opaque red line
        Image.fromarray(rgba, "RGBA").save(tmp_path / "t.png")
        img = load_image(tmp_path / "t.png")
        assert img[0, 0].tolist() == [255, 255, 255]
        assert img[5, 0].tolist() == [255, 0, 0]

    def test_greyscale_and_palette(self, tmp_path):
        Image.new("L", (10, 10), 128).save(tmp_path / "g.png")
        assert load_image(tmp_path / "g.png")[0, 0].tolist() == [128, 128, 128]
        Image.new("RGB", (10, 10), (0, 255, 0)).convert("P").save(tmp_path / "p.png")
        assert load_image(tmp_path / "p.png")[0, 0].tolist() == [0, 255, 0]

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.png")

    def test_not_an_image(self, tmp_path):
        (tmp_path / "x.png").write_text("hello")
        with pytest.raises(ValueError, match="Cannot read"):
            load_image(tmp_path / "x.png")
        (tmp_path / "x.pdf").write_text("%PDF-1.4")
        with pytest.raises(ValueError, match="exported as an image"):
            load_image(tmp_path / "x.pdf")


def test_panel_metadata_columns_are_backward_compatible():
    panel = {
        "panel_id": 1, "x0": 1, "x1": 9, "y_top": 2, "y_bottom": 8, "source": "s", "score": 1.0, "gridline_y": [2, 8],
        "y_calibration": {"a": -1.0, "b": 8.0, "scale_type": "ocr_value"},
    }
    meta = build_panel_metadata([panel])
    first = ["panel_id", "x0", "x1", "y_top", "y_bottom", "source", "score", "gridline_y", "y_calibration_a",
             "y_calibration_b", "scale_type", "x_calibration_a", "x_calibration_b", "x_scale_type", "x_labels"]
    assert list(meta.columns[: len(first)]) == first  # the 1.2.0 columns, in the 1.2.0 order


def test_provenance_helpers(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"abc")
    assert sha256_file(tmp_path / "f.bin") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    plain = _plain({"a": np.float64(1.5), "b": np.int64(2), "c": np.array([1, 2]), "d": float("nan"), 3: tmp_path})
    assert plain == {"a": 1.5, "b": 2, "c": [1, 2], "d": None, "3": str(tmp_path)}
