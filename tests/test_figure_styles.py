"""Figure styles found in published figures (GraphPad Prism, MATLAB) that 2.1.0 could not read, drawn here from known
values: black, grey and hatched bars on an axis that does not start at 0, with error bars of their own colour;
markers on a fitted line of their own colour, with error bars of their own colour; a shaded band behind a line; a
plot area given a few pixels off its thick axis lines."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from trace_digitiser import describe_figure, digitise, digitise_bars, digitise_points
from trace_digitiser.axis_labels import tesseract_available

warnings.simplefilter("ignore")


def save(fig: Figure, path, dpi=100) -> str:
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi)
    return str(path)


def axes_box(fig: Figure, ax, dpi=100) -> tuple[float, float, float, float]:
    FigureCanvasAgg(fig).draw()
    h = fig.get_figheight() * dpi
    b = ax.get_window_extent()
    s = dpi / fig.dpi
    return (b.x0 * s, h - b.y1 * s, b.x1 * s, h - b.y0 * s)


@pytest.fixture(scope="module")
def prism_bars(tmp_path_factory):
    """Bars in the style of Prism: black, grey and hatched, on an axis from 23 to 25 h, thick axis lines, error bars
    drawn upwards only in each bar's own colour."""
    means = np.array([24.21, 24.02, 24.06, 24.80, 23.72, 23.65])
    sems = np.array([0.04, 0.04, 0.03, 0.04, 0.07, 0.06])
    fig = Figure(figsize=(5, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    styles = [("0.55", "black", "///"), ("0.55", "black", "///"), ("black", "black", None),
              ("#2036fb", "#2036fb", None), ("white", "#2036fb", "///"), ("white", "#2036fb", "///")]
    for i, (m, e, (face, edge, hatch)) in enumerate(zip(means, sems, styles)):
        ax.bar(i, m - 22.5, bottom=22.5, color=face, edgecolor=edge, hatch=hatch, linewidth=1.5, width=0.7)
        ax.errorbar(i, m, yerr=[[0], [e]], color=edge, capsize=6, elinewidth=1.5, capthick=1.5)
    ax.set_ylim(22.5, 25.5)
    ax.set_yticks([23, 24, 25])
    ax.set_xlim(-0.6, 5.6)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["a", "b", "c", "d", "e", "f"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(3)
    path = save(fig, tmp_path_factory.mktemp("prism") / "bars.png", dpi=200)
    return path, means, sems, fig, ax


def test_black_grey_and_hatched_bars_on_an_axis_from_23(prism_bars):
    path, means, sems, *_ = prism_bars
    axis = {} if tesseract_available() else {"y_min": 22.5, "y_max": 25.5}
    r = digitise_bars(path, write_outputs=False, **axis)
    d = r.data.sort_values("x_pixel")
    assert len(d) == 6
    assert np.abs(d.value.to_numpy() - means).max() < 0.03
    # error bars of the bars' own colour, upwards: error_high is where they end
    got = d.dropna(subset=["error_high"])
    assert len(got) >= 4
    idx = [list(d.index).index(i) for i in got.index]
    assert np.abs((got.error_high - got.value).to_numpy() - sems[idx]).max() < 0.03


def test_plot_area_given_off_the_thick_axis_lines(prism_bars):
    path, means, _, fig, ax = prism_bars
    left, top, right, bottom = axes_box(fig, ax, dpi=200)
    # a box drawn by hand, a few pixels off the (6-pixel) axis lines
    r = digitise_bars(path, plot_area=[(left - 3, top + 2, right - 2, bottom + 3)], write_outputs=False,
                      **({} if tesseract_available() else {"y_min": 22.5, "y_max": 25.5}))
    d = r.data.sort_values("x_pixel")
    assert len(d) == 6
    assert np.abs(d.value.to_numpy() - means).max() < 0.04


@pytest.mark.skipif(not tesseract_available(), reason="needs Tesseract")
def test_given_plot_area_reads_the_tick_labels_beyond_a_thick_axis(prism_bars):
    path, _, _, fig, ax = prism_bars
    left, top, right, bottom = axes_box(fig, ax, dpi=200)
    info = describe_figure(path, plot_area=[(left + 2, top, right, bottom - 2)])
    y = info["panels"][0]["y_axis"]
    assert sorted(v for lab in y["labels"] for v in lab["values"][:1]) == [23.0, 24.0, 25.0]


def test_markers_on_a_fitted_line_with_error_bars_of_their_own_colour(tmp_path):
    """MATLAB style: small square markers on a thick fitted line of the same colour, error bars of the same colour."""
    x = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
    y = np.array([-2.2, -2.4, -1.8, -1.2, -0.2])
    err = np.array([1.3, 1.25, 1.15, 1.1, 1.1])
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=err, fmt="s", color="#2c2c90", markersize=6, capsize=4, elinewidth=1.5)
    fit = np.polyfit(x, y, 1)
    ax.plot([8, 16], np.polyval(fit, [8, 16]), color="#2c2c90", linewidth=3.5)
    ax.set_xlim(6, 18)
    ax.set_ylim(-4, 4)
    path = save(fig, tmp_path / "pts.png", dpi=200)
    r = digitise_points(path, x_min=6, x_max=18, y_min=-4, y_max=4, write_outputs=False)
    d = r.data.sort_values("x_value")
    near = [d.iloc[int(np.argmin(np.abs(d.x_value.to_numpy() - xv)))] for xv in x]
    assert all(abs(p.x_value - xv) < 0.15 and abs(p.y_value - yv) < 0.1 for p, xv, yv in zip(near, x, y))
    with_bars = [p for p in near if np.isfinite(p.get("y_error_high", np.nan))]
    assert len(with_bars) >= 4
    good = 0
    for p in with_bars:
        k = int(np.argmin(np.abs(x - p.x_value)))
        good += abs((p.y_error_high - p.y_error_low) / 2 - err[k]) < 0.15
    # where the fitted line starts at a marker and crosses its error bar, that bar can stop at the line
    assert good >= 4


def test_a_shaded_band_is_not_a_trace(tmp_path):
    t = np.linspace(0, 72, 200)
    v = 35 + 25 * np.sin(2 * np.pi * t / 24)
    fig = Figure(figsize=(6, 3), dpi=100)
    ax = fig.add_subplot(111)
    for start in (0, 24, 48):
        ax.axvspan(start, start + 12, color="#f2e8c4", lw=0)
    ax.plot(t, v, color="black", lw=1.5)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 75)
    path = save(fig, tmp_path / "band.png")
    r = digitise(path, x_min=0, x_max=72, y_min=0, y_max=75, write_outputs=False)
    d = r.trace_data
    assert d["trace_id"].nunique() == 1
    got = d.dropna(subset=["center_estimate"])
    want = 35 + 25 * np.sin(2 * np.pi * got["x_value"].to_numpy() / 24)
    assert np.median(np.abs(got["center_estimate"].to_numpy() - want)) < 0.5
