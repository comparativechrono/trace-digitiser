"""Tests for the command-line interface."""

import json

import pandas as pd
import pytest

from trace_digitiser.cli import _box, _build_parser, _point, main

from .conftest import needs_tesseract


class TestArgumentParsing:
    def test_box(self):
        assert _box("41,23,388,159") == (41.0, 23.0, 388.0, 159.0)
        assert _box("41.5, 23, 388, 159.25") == (41.5, 23.0, 388.0, 159.25)

    @pytest.mark.parametrize("text", ["1,2,3", "a,b,c,d", ""])
    def test_bad_box(self, text):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _box(text)

    def test_point(self):
        assert _point("159:0") == (159.0, 0.0)
        assert _point("22.7:-1.5e2") == (22.7, -150.0)

    @pytest.mark.parametrize("text", ["159", "159:", "a:b"])
    def test_bad_point(self, text):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _point(text)

    def test_repeatable_options(self):
        args = _build_parser().parse_args(
            ["fig.png", "--plot-area", "1,2,3,4", "--plot-area", "5,6,7,8", "--y-ref", "10:0", "--y-ref", "5:1",
             "--trace-colour", "ff0000", "--trace-color", "0000ff", "--exclude", "1,1,2,2"]
        )
        assert args.plot_area == [(1, 2, 3, 4), (5, 6, 7, 8)]
        assert args.y_ref == [(10, 0), (5, 1)]
        assert args.trace_colours == ["ff0000", "0000ff"]

    def test_defaults_match_1x(self):
        """Options that existed in 1.2.0 keep their names and defaults."""
        args = _build_parser().parse_args(["fig.png", "--layout", "stacked", "--rows", "2", "--cols", "1",
                                           "--y-min", "0", "--y-max", "300", "--total-days", "7", "--plot",
                                           "--sampling-interval", "auto", "--x-units", "days", "--save-diagnostics"])
        assert (args.layout_mode, args.expected_rows, args.expected_cols) == ("stacked", 2, 1)
        assert (args.y_min, args.y_max, args.total_days) == (0, 300, 7)
        assert args.plot and args.save_diagnostics and args.sampling_interval == "auto"


class TestMain:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.startswith("trace-digitiser ")

    def test_no_arguments_prints_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1
        assert "usage:" in capsys.readouterr().out

    def test_digitise_with_limits(self, render, tmp_path, capsys):
        fig = render("A1_box_grid_blue")
        main([str(fig.path), "-o", str(tmp_path), "--y-min", "0", "--y-max", "100", "--x-min", "0", "--x-max", "10",
              "--no-ocr", "--overlay", "--plot"])
        out = capsys.readouterr().out
        assert "Panels: 1, Traces: 1" in out
        assert "y 0 to 100, from the limits given" in out
        assert "CHECK:" not in out
        stem = fig.path.stem
        data = pd.read_csv(tmp_path / f"{stem}_automated_digitised_trace_by_column.csv")
        assert data["x_value"].between(-0.1, 10.1).all()
        assert (tmp_path / f"{stem}_automated_overlay.png").exists()
        assert (tmp_path / f"{stem}_trace_plot.png").exists()
        record = json.loads((tmp_path / f"{stem}_automated_provenance.json").read_text())
        assert record["parameters"]["ocr"] is False

    def test_manual_plot_area_and_references(self, render, tmp_path, capsys):
        fig = render("A1_box_grid_blue")
        left, top, right, bottom = fig.box()
        main([str(fig.path), "-o", str(tmp_path), "--no-ocr", "--plot-area", f"{left},{top},{right},{bottom}",
              "--y-ref", f"{bottom}:0", "--y-ref", f"{top}:100", "--x-ref", f"{left}:0", "--x-ref", f"{right}:10"])
        out = capsys.readouterr().out
        assert "from the reference points given" in out

    def test_warnings_are_shown(self, render, tmp_path, capsys):
        main([str(render("A1_box_grid_blue").path), "-o", str(tmp_path), "--no-ocr"])
        out = capsys.readouterr().out
        assert "NOT CALIBRATED" in out
        assert "CHECK: Panel 1: the y axis is not calibrated" in out

    def test_missing_file_fails(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / "missing.png"), "-o", str(tmp_path)])
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_bad_combinations_rejected(self, render, tmp_path):
        path = str(render("A1_box_grid_blue").path)
        for bad in (["--y-min", "0"], ["--y-ref", "10:0"], ["--x-max", "3"]):
            with pytest.raises(SystemExit) as exc:
                main([path, "-o", str(tmp_path), *bad])
            assert exc.value.code == 2

    def test_batch_continues_after_a_bad_file(self, render, tmp_path, capsys):
        good = render("A1_box_grid_blue").path
        bad = tmp_path / "not_an_image.png"
        bad.write_text("this is not an image")
        with pytest.raises(SystemExit) as exc:
            main([str(bad), str(good), "-o", str(tmp_path), "--no-ocr"])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "Cannot read" in captured.err
        assert "Panels: 1, Traces: 1" in captured.out

    def test_generate_test_figures(self, tmp_path, capsys):
        main(["--generate-test-figures", "-o", str(tmp_path)])
        assert (tmp_path / "synthetic_ground_truth.csv").exists()
        assert len(list(tmp_path.glob("*_noisy.jpg"))) == 4

    @needs_tesseract
    def test_fully_automatic(self, render, tmp_path, capsys):
        main([str(render("A1_box_grid_blue").path), "-o", str(tmp_path)])
        out = capsys.readouterr().out
        assert "read from tick labels (high confidence)" in out
        assert "CHECK:" not in out
