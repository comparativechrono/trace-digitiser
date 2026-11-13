"""Shared fixtures for trace_digitiser tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
PUBLISHED_FIGURE = REPO / "examples" / "lopatina2014_fig4.jpg"
REFERENCE = Path(__file__).resolve().parent / "data" / "lopatina2014_fig4_reference.csv"

needs_tesseract = pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract OCR not installed")
needs_published_figure = pytest.mark.skipif(not PUBLISHED_FIGURE.exists(), reason="example figure not available")


# ---------------------------------------------------------------------------
# Figures drawn from known data (see trace_digitiser.validation)
# ---------------------------------------------------------------------------

class Rendered:
    """A rendered validation figure: ``.path`` of the image, ``.truth`` record, ``.spec``."""

    def __init__(self, spec, directory: Path):
        from trace_digitiser.validation import render_figure

        self.spec = spec
        self.truth = render_figure(spec, directory)
        self.path = directory / self.truth["image"]

    @property
    def panel(self) -> dict:
        return self.truth["panels"][0]

    def box(self, i: int = 0) -> tuple[float, float, float, float]:
        """True plot area of panel *i*: (x_left, y_top, x_right, y_bottom)."""
        p = self.truth["panels"][i]
        return (p["left"], p["top"], p["right"], p["bottom"])


@pytest.fixture(scope="session")
def render(tmp_path_factory):
    """Factory: ``render(name)`` or ``render(spec)`` -> :class:`Rendered` (cached per session)."""
    from trace_digitiser.validation import FigureSpec, default_cases

    directory = tmp_path_factory.mktemp("rendered")
    by_name = {c.name: c for c in default_cases()}
    cache: dict[str, Rendered] = {}

    def make(spec) -> Rendered:
        if isinstance(spec, str):
            spec = by_name[spec]
        assert isinstance(spec, FigureSpec)
        if spec.name not in cache:
            cache[spec.name] = Rendered(spec, directory)
        return cache[spec.name]

    return make


# ---------------------------------------------------------------------------
# The original noisy multi-panel figures (trace_digitiser.synthetic)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_figures_dir(tmp_path_factory) -> Path:
    """Generate synthetic test figures once per test session."""
    from trace_digitiser.synthetic import generate_noisy_example_plots

    out = tmp_path_factory.mktemp("test_figures")
    generate_noisy_example_plots(output_dir=out, seed=42, show=False)
    return out


@pytest.fixture(scope="session")
def ground_truth(test_figures_dir) -> pd.DataFrame:
    """Load the synthetic ground-truth CSV."""
    return pd.read_csv(test_figures_dir / "synthetic_ground_truth.csv")


@pytest.fixture(scope="session")
def vertical_2_panel_image(test_figures_dir) -> Path:
    """Path to the vertical 2-panel noisy test image."""
    return test_figures_dir / "vertical_2_panels_noisy.jpg"


@pytest.fixture(scope="session")
def horizontal_3_panel_image(test_figures_dir) -> Path:
    """Path to the horizontal 3-panel noisy test image."""
    return test_figures_dir / "horizontal_3_panels_noisy.jpg"


@pytest.fixture(scope="session")
def grid_2x2_image(test_figures_dir) -> Path:
    """Path to the 2×2 grid noisy test image."""
    return test_figures_dir / "grid_2x2_panels_noisy.jpg"


@pytest.fixture(scope="session")
def wide_2x3_image(test_figures_dir) -> Path:
    """Path to the 2×3 grid noisy test image."""
    return test_figures_dir / "wide_colours_2x3_noisy.jpg"


@pytest.fixture
def simple_rgb() -> np.ndarray:
    """A minimal 200×400 white image for unit-testing geometry helpers."""
    return np.ones((200, 400, 3), dtype=np.uint8) * 255


# ---------------------------------------------------------------------------
# A published figure (CC BY; see examples/README.md)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def published_figure() -> Path:
    return PUBLISHED_FIGURE


@pytest.fixture(scope="session")
def published_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE)
