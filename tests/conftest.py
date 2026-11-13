"""Shared fixtures for trace_digitiser tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


@pytest.fixture
def simple_rgb() -> np.ndarray:
    """A minimal 200×400 white image for unit-testing geometry helpers."""
    return np.ones((200, 400, 3), dtype=np.uint8) * 255
