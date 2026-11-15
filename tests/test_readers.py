"""The readers of points, bars, rows, heat maps and polar plots, on figures drawn here from known values."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from trace_digitiser import (
    FigureResult,
    describe_figure,
    digitise_bars,
    digitise_heatmap,
    digitise_points,
    digitise_polar,
    digitise_rows,
    pixels_to_values,
)
from trace_digitiser.axis_labels import tesseract_available
from trace_digitiser.markers import colour_mask, find_markers, marker_radius, under_ink

warnings.simplefilter("ignore")


def save(fig: Figure, path) -> str:
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=100)
    return str(path)


def axes_box(fig: Figure, ax) -> tuple[float, float, float, float]:
    """The plot area of *ax* in image pixels (left, top, right, bottom)."""
    FigureCanvasAgg(fig).draw()
    h = fig.get_figheight() * fig.dpi
    b = ax.bbox
    return (b.x0, h - b.y1, b.x1, h - b.y0)


def frame(ax) -> None:
    for s in ax.spines.values():
        s.set_linewidth(1.0)


# --------------------------------------------------------------------- points


@pytest.fixture(scope="module")
def scatter(tmp_path_factory):
    rng = np.random.default_rng(3)
    x = np.arange(1, 13, dtype=float)
    y = 20 + 10 * np.sin(x / 2) + rng.normal(0, 1, x.size)
    err = 1.5 + rng.uniform(0, 2, x.size)
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=err, fmt="o-", color="tab:red", ecolor="0.3", capsize=3, markersize=6)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 40)
    frame(ax)
    path = save(fig, tmp_path_factory.mktemp("pts") / "scatter.png")
    return path, x, y, err


def test_points_found_with_their_error_bars(scatter):
    path, x, y, err = scatter
    r = digitise_points(path, x_min=0, x_max=13, y_min=0, y_max=40, write_outputs=False)
    assert isinstance(r, FigureResult)
    d = r.data.sort_values("x_value")
    assert len(d) == len(x)
    assert np.abs(d.x_value.to_numpy() - x).max() < 0.1
    assert np.abs(d.y_value.to_numpy() - y).max() < 0.3
    assert "y_error_low" in d
    assert np.abs(d.y_error_low.to_numpy() - (y - err)).max() < 0.4
    assert np.abs(d.y_error_high.to_numpy() - (y + err)).max() < 0.4
    assert "x_error_low" not in d  # no horizontal bars


def test_points_with_colour_given_and_none_found(scatter):
    path, x, *_ = scatter
    r = digitise_points(path, colours=["tab:red"], x_min=0, x_max=13, y_min=0, y_max=40, error_bars="none",
                        write_outputs=False)
    assert len(r.data) == len(x)
    assert "y_error_low" not in r.data
    r = digitise_points(path, colours=["#00ff00"], x_min=0, x_max=13, y_min=0, y_max=40, write_outputs=False)
    assert r.data.empty
    assert any("no markers" in w for w in r.warnings)


def test_markers_without_lines_and_overlapping(tmp_path):
    fig = Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    pts = [(1, 1), (2, 3), (3, 2), (3.12, 2.1), (4, 4)]  # two of them overlap
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "s", color="black", markersize=8)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    path = save(fig, tmp_path / "m.png")
    r = digitise_points(path, x_min=0, x_max=5, y_min=0, y_max=5, write_outputs=False)
    got = sorted(zip(r.data.x_value.round(1), r.data.y_value.round(1)))
    assert len(got) == 5
    assert (r.data.shared > 1).sum() == 2


def test_marker_radius_of_lines_only():
    m = np.zeros((60, 60), bool)
    m[30, 5:55] = True
    m[10:50, 20] = True
    assert marker_radius(m) is None


def test_find_markers_box_and_exclude():
    rgb = np.full((50, 80, 3), 255, np.uint8)
    for cx in (15, 40, 65):
        yy, xx = np.ogrid[:50, :80]
        rgb[(yy - 25) ** 2 + (xx - cx) ** 2 <= 16] = (0, 0, 200)
    found = find_markers(rgb, (0, 0, 79, 49), [(0, 0, 200)])
    assert [round(m.x) for m in found] == [15, 40, 65]
    found = find_markers(rgb, (0, 0, 79, 49), [(0, 0, 200)], exclude=[(55, 0, 79, 49)])
    assert [round(m.x) for m in found] == [15, 40]


def test_markers_under_a_black_bar():
    # the mean's bar of a dot plot, drawn across two dots: the dots go on under it, the bar beyond them does not
    rgb = np.full((40, 90, 3), 255, np.uint8)
    yy, xx = np.ogrid[:40, :90]
    for cx in (25, 60):
        rgb[(yy - 20) ** 2 + (xx - cx) ** 2 <= 64] = (236, 71, 57)
    rgb[18:22, 5:85] = 0
    colour = (236, 71, 57)
    hidden = under_ink(rgb, colour, colour_mask(rgb, colour))
    assert hidden[18:22, 20:30].all() and hidden[18:22, 55:65].all()
    assert not hidden[:, :15].any() and not hidden[:, 36:50].any() and not hidden[:, 70:].any()
    found = find_markers(rgb, (0, 0, 89, 39), [colour])
    assert [(round(m.x), round(m.y)) for m in found] == [(25, 20), (60, 20)]
    assert not under_ink(rgb, (0, 0, 0), colour_mask(rgb, (0, 0, 0))).any()  # black markers: the bar is theirs


# --------------------------------------------------------------------- bars


@pytest.fixture(scope="module")
def barchart(tmp_path_factory):
    values = np.array([12.0, -7.5, 20.0, 4.0, -15.0])
    errs = np.array([2.0, 1.5, 3.0, 1.0, 2.5])
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(1, 6), values, yerr=errs, color="#4c72b0", capsize=3, label="mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-25, 30)
    ax.set_xlim(0.4, 5.6)
    ax.legend(loc="upper right")
    frame(ax)
    path = save(fig, tmp_path_factory.mktemp("bars") / "bars.png")
    return path, values, errs


def test_bars_values_and_error_bars(barchart):
    path, values, errs = barchart
    r = digitise_bars(path, y_min=-25, y_max=30, x_min=0.4, x_max=5.6, write_outputs=False)
    d = r.data.sort_values("x_pixel")
    assert len(d) == 5  # the legend's patch is left out
    assert np.abs(d.value.to_numpy() - values).max() < 0.3
    assert np.abs(d.x_value.to_numpy() - np.arange(1, 6)).max() < 0.05
    assert np.allclose(d.baseline, 0, atol=1e-6)
    half = 0.5 * (d.error_high - d.error_low).to_numpy()
    assert np.abs(half - errs).max() < 0.4
    assert any("do not reach the baseline" in w for w in r.warnings)


def test_bars_horizontal(tmp_path):
    values = np.array([3.0, 7.0, 5.0])
    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.barh([1, 2, 3], values, color="#dd8452")
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 3.6)
    frame(ax)
    path = save(fig, tmp_path / "h.png")
    r = digitise_bars(path, orientation="horizontal", x_min=0, x_max=10, y_min=0.4, y_max=3.6, error_bars=False,
                      write_outputs=False)
    d = r.data.sort_values("y_value")
    assert np.abs(d.value.to_numpy() - values).max() < 0.1


# --------------------------------------------------------------------- rows


@pytest.fixture(scope="module")
def actogram(tmp_path_factory):
    rng = np.random.default_rng(5)
    n_rows, n_bins = 8, 24
    vals = rng.uniform(0, 1, (n_rows, n_bins))
    vals[:, 0] = 0.0
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    for i in range(n_rows):
        ax.fill_between(np.arange(n_bins), i * 1.1, i * 1.1 + vals[i], step="mid", color="#4C72B0", alpha=0.6)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylim(-0.5, n_rows * 1.1 + 0.5)
    ax.set_yticks([i * 1.1 for i in range(n_rows)])
    ax.set_yticklabels([str(2000 + i) for i in range(n_rows)])
    frame(ax)
    path = save(fig, tmp_path_factory.mktemp("rows") / "acto.png")
    lo, hi = -0.5, n_rows * 1.1 + 0.5
    return path, vals, (lo, hi)


def test_rows_heights(actogram):
    path, vals, (lo, hi) = actogram
    n_rows, n_bins = vals.shape
    r = digitise_rows(path, row_values=[i * 1.1 for i in range(n_rows)], y_min=lo, y_max=hi, bins=n_bins,
                      row_spacing_value=1.1, direction="up", write_outputs=False)
    d = r.data
    assert len(d) == n_rows * n_bins
    got = d.sort_values(["row", "bin"]).value.to_numpy().reshape(n_rows, n_bins)
    # rows are numbered from the top: the last row drawn (the highest) is row 1
    err = np.abs(got[::-1] - vals)
    assert np.median(err) < 0.03
    assert np.percentile(err, 95) < 0.08


def test_rows_need_their_positions(tmp_path):
    fig = Figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.fill_between([0, 1, 2], 0, [0.2, 0.8, 0.5], color="tab:blue")
    ax.set_xticks([])
    ax.set_yticks([])
    path = save(fig, tmp_path / "r.png")
    r = digitise_rows(path, write_outputs=False)
    assert r.data.empty
    assert any("where its rows are" in w for w in r.warnings)


# --------------------------------------------------------------------- heat maps


@pytest.fixture(scope="module")
def heat(tmp_path_factory):
    rng = np.random.default_rng(1)
    m = np.round(rng.uniform(-1, 1, (6, 8)), 2)
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(m, cmap="viridis", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)
    path = save(fig, tmp_path_factory.mktemp("heat") / "heat.png")
    return path, m


def test_heatmap_from_colormap(heat):
    path, m = heat
    r = digitise_heatmap(path, colormap="viridis", vmin=-1, vmax=1, rows=6, cols=8, write_outputs=False)
    big = r.data.groupby("panel_id").size().idxmax()
    d = r.data[r.data.panel_id == big].sort_values(["row", "col"])
    got = d.value.to_numpy().reshape(6, 8)
    assert np.abs(got - m).max() < 0.03


def test_heatmap_cells_found_without_rows_and_cols(heat):
    path, m = heat
    r = digitise_heatmap(path, colormap="viridis", vmin=-1, vmax=1, write_outputs=False)
    big = r.data.groupby("panel_id").size().idxmax()
    d = r.data[r.data.panel_id == big]
    assert d.row.max() == 6 and d.col.max() == 8


@pytest.mark.skipif(not tesseract_available(), reason="needs Tesseract")
def test_heatmap_colour_bar_read_from_its_labels(heat):
    path, m = heat
    r = digitise_heatmap(path, rows=6, cols=8, write_outputs=False)
    assert r.provenance["colour_scale"]["method"] == "tick_labels"
    d = r.data.sort_values(["row", "col"])
    got = d.value.to_numpy().reshape(6, 8)
    assert np.abs(got - m).max() < 0.05


def test_heatmap_colour_bar_ends_given(heat, tmp_path):
    path, m = heat
    from trace_digitiser.heatmap import find_colourbar
    from trace_digitiser.io import load_image
    from trace_digitiser.panel_detection import find_plot_panels

    rgb = load_image(path)
    bar = find_colourbar(rgb, find_plot_panels(rgb)[0])
    assert bar is not None and bar["vertical"]
    e = bar["edges"]
    r = digitise_heatmap(path, colourbar=(e["left"], e["top"], e["right"], e["bottom"]), colourbar_values=(-1, 1),
                         rows=6, cols=8, write_outputs=True, output_dir=tmp_path, save_overlay=True)
    got = r.data.sort_values(["row", "col"]).value.to_numpy().reshape(6, 8)
    assert np.abs(got - m).max() < 0.05
    assert r.csv_path.exists() and r.overlay_path.exists() and r.provenance_path.exists()


# --------------------------------------------------------------------- polar


def test_polar_points(tmp_path):
    theta = np.radians([20, 80, 150, 230, 300])
    radius = np.array([2.0, 4.0, 6.0, 8.0, 9.5])
    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, radius, "o", color="tab:green", markersize=7)
    ax.set_rlim(0, 10)
    ax.set_rgrids([2, 4, 6, 8, 10])
    path = save(fig, tmp_path / "polar.png")
    r = digitise_polar(path, outer_value=10, write_outputs=False)
    d = r.data.sort_values("r_value")
    assert len(d) == 5
    assert np.abs(d.r_value.to_numpy() - radius).max() < 0.15
    da = (d.angle_degrees.to_numpy() - np.degrees(theta) + 180) % 360 - 180
    assert np.abs(da).max() < 1.0
    rings = r.provenance["polar"]["rings"]
    assert len(rings) >= 4
    r2 = digitise_polar(path, r_ref=[(rings[0], 2), (rings[3], 8)], zero="north", clockwise=True,
                        write_outputs=False)
    d2 = r2.data.sort_values("r_value")
    assert np.abs(d2.r_value.to_numpy() - radius).max() < 0.15
    clock = (90 - np.degrees(theta)) % 360
    assert np.abs((d2.angle_degrees.to_numpy() - clock + 180) % 360 - 180).max() < 1.0


# --------------------------------------------------------------------- helpers


def test_pixels_to_values():
    out = pixels_to_values([(41, 159), (388, 23)], x_ref=[(41, 0), (388, 7)], y_ref=[(159, 0), (23, 300)])
    assert out[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert out[1] == pytest.approx((7.0, 300.0), abs=1e-9)
    out = pixels_to_values([(100, 50)], y_ref=[(150, 1), (50, 1000)], y_scale="log")
    assert out[0][0] is None and out[0][1] == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        pixels_to_values([(0, 0)], x_ref=[(1, 1)])


def test_describe_figure(scatter):
    path, *_ = scatter
    info = describe_figure(path, ocr=False)
    json.dumps(info)
    assert info["width"] == 600 and info["height"] == 400
    assert len(info["panels"]) == 1
    p = info["panels"][0]
    assert len(p["plot_area"]) == 4
    assert any(c["hex"] for c in p["colours"])
    assert info["polar"] is None


@pytest.mark.skipif(not tesseract_available(), reason="needs Tesseract")
def test_describe_figure_reads_the_axes(scatter):
    path, *_ = scatter
    p = describe_figure(path)["panels"][0]
    assert p["y_axis"]["ticks"] and p["x_axis"]["ticks"]
    ys = [v for _, v in p["y_axis"]["ticks"]]
    assert min(ys) == 0 and max(ys) == 40


# --------------------------------------------------------------------- the command line


def test_command_line_subcommands(barchart, scatter, tmp_path, capsys):
    from trace_digitiser.cli import main

    path, values, _ = barchart
    main(["bars", path, "--y-min", "-25", "--y-max", "30", "-o", str(tmp_path), "--overlay"])
    out = capsys.readouterr().out
    assert "Found: 5 bars" in out
    d = pd.read_csv(tmp_path / "bars_bars.csv")
    assert np.abs(d.sort_values("x_pixel").value.to_numpy() - values).max() < 0.3
    assert (tmp_path / "bars_bars_overlay.png").exists()
    spath, x, *_ = scatter
    main(["points", spath, "--x-min", "0", "--x-max", "13", "--y-min", "0", "--y-max", "40", "-o", str(tmp_path)])
    assert len(pd.read_csv(tmp_path / "scatter_points.csv")) == len(x)
    main(["describe", spath, "--no-ocr", "-o", str(tmp_path / "d.json")])
    assert json.loads((tmp_path / "d.json").read_text())["panels"]


def test_command_line_image_named_like_a_subcommand(tmp_path, monkeypatch):
    """An image file that happens to be called "points" is still digitised as before."""
    from trace_digitiser.cli import SUBCOMMANDS

    assert "points" in SUBCOMMANDS
    monkeypatch.chdir(tmp_path)
    (tmp_path / "points").write_bytes(b"not an image")
    from trace_digitiser.cli import main

    with pytest.raises(SystemExit):
        main(["points", "--no-ocr"])  # taken as an image, which cannot be read


def test_angle_conventions_round_trip():
    # zero north, clockwise: a point straight to the right is at 90 degrees
    theta = math.degrees(math.atan2(0, 1))
    assert (theta - 90) % 360 == 270  # anticlockwise from north
    assert (-(theta - 90)) % 360 == 90  # clockwise from north


# --------------------------------------------------------------------- a second y axis


@pytest.mark.skipif(not tesseract_available(), reason="needs Tesseract")
def test_lines_against_the_right_hand_axis(tmp_path):
    from trace_digitiser import digitise

    x = np.linspace(0, 10, 200)
    left = 200 + 100 * np.sin(x / 2)
    right = 0.4 + 0.3 * np.cos(x / 3)
    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(x, left, color="#1f77b4", lw=1.5)
    ax.set_ylim(50, 350)
    ax2 = ax.twinx()
    ax2.plot(x, right, color="#ff7f0e", lw=1.5)
    ax2.set_ylim(0, 1)
    fig.subplots_adjust(left=0.12, right=0.88)
    path = save(fig, tmp_path / "twin.png")
    r = digitise(path, trace_colours=["#ff7f0e"], y_axis="right", write_outputs=False)
    cal = r.panels[0].calibration
    assert cal.method == "tick_labels"
    top, bottom = (float(cal.y_to_value(r.panels[0].edges[k])) for k in ("top", "bottom"))
    assert abs(top - 1.0) < 0.01 and abs(bottom) < 0.01
    d = r.trace_data.dropna(subset=["x_value"])
    want = 0.4 + 0.3 * np.cos(d["x_value"].to_numpy() / 3)
    assert np.median(np.abs(d["center_estimate"].to_numpy() - want)) < 0.01
    # the left-hand axis is still the default
    r = digitise(path, trace_colours=["#1f77b4"], write_outputs=False)
    top = float(r.panels[0].calibration.y_to_value(r.panels[0].edges["top"]))
    assert abs(top - 350) < 3
    with pytest.raises(ValueError):
        digitise(path, y_axis="middle", write_outputs=False)
