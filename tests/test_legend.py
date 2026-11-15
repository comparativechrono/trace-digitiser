"""Rows of text and legends inside a plot (trace_digitiser.legend)."""

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from trace_digitiser.legend import find_legend, find_text_rows

from .conftest import REPO


def _render(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def _inside(ax, rgb):
    """The plot area of *ax* in the rendered image, as (x0, y0, x1, y1), a few pixels in from its frame."""
    b = ax.get_window_extent()
    h = rgb.shape[0]
    return int(b.x0) + 3, int(h - b.y1) + 3, int(b.x1) - 3, int(h - b.y0) - 3


def test_a_row_of_words_is_found():
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    ax.plot([0, 1], [0, 1], color="#1f77b4")
    ax.text(0.55, 0.2, "Knockout mice", fontsize=11, color="black", transform=ax.transAxes)
    rgb = _render(fig)
    rows = find_text_rows(rgb, _inside(ax, rgb))
    assert len(rows) == 1
    x0, y0, x1, y1 = _inside(ax, rgb)
    r = rows[0]
    assert r.x0 > x0 + 0.5 * (x1 - x0) and r.y0 > y0 + 0.6 * (y1 - y0)  # where the words are


def test_a_legend_is_found_and_the_lines_are_not():
    fig = Figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
    t = np.linspace(0, 10, 200)
    ax.plot(t, np.sin(t), color="#d62728", label="Wild type")
    ax.plot(t, np.cos(t), color="#1f77b4", label="Knockout")
    ax.set_ylim(-1.2, 3)
    ax.legend(loc="upper left", frameon=False)
    rgb = _render(fig)
    boxes = find_legend(rgb, _inside(ax, rgb))
    assert boxes
    top = min(b[1] for b in boxes)
    x0, y0, x1, y1 = _inside(ax, rgb)
    assert max(b[3] for b in boxes) < y0 + 0.45 * (y1 - y0)  # all in the top part, where the legend is
    assert top >= y0


DENSE = REPO / "validation" / "dense-traces" / "figures" / "locomotor_mouse_1-PMC-sized-red.jpg"


@pytest.mark.skipif(not DENSE.exists(), reason="validation figure not available")
def test_the_steep_strokes_of_a_dense_trace_are_not_text():
    # a week of 5-min samples drawn as one line, 3.7 samples to a pixel column: where the trace drops to the axis
    # and rises again, its upright strokes stand on the gridline like letters on a line
    rgb = np.asarray(Image.open(DENSE).convert("RGB"))
    assert find_text_rows(rgb, (20, 20, 571, 237)) == []
