"""Synthetic test-figure generation.

Creates noisy raster plots with known ground truth for testing the
digitiser pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

# ===================================================================
# Noise injection
# ===================================================================

def add_image_noise(
    path: str | Path,
    out_path: str | Path | None = None,
    noise_sigma: float = 8,
    salt_pepper: float = 0.002,
    jpeg_quality: int = 86,
    seed: int = 0,
) -> Path:
    """Add Gaussian noise, salt-and-pepper, blur, and optional JPEG compression."""
    rng = np.random.default_rng(seed)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.int16)

    noise = rng.normal(0, noise_sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    if salt_pepper > 0:
        mask = rng.random(arr.shape[:2])
        arr[mask < salt_pepper / 2] = 0
        arr[(mask >= salt_pepper / 2) & (mask < salt_pepper)] = 255

    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.15))

    if out_path is None:
        out_path = path
    out_path = Path(out_path)

    if out_path.suffix.lower() in [".jpg", ".jpeg"]:
        img.save(out_path, quality=jpeg_quality, subsampling=1)
    else:
        img.save(out_path)

    return out_path


# ===================================================================
# Synthetic series
# ===================================================================

def make_synthetic_series(
    n: int = 900,
    seed: int = 0,
    baseline: float = 55,
    amplitude: float = 40,
    spike_scale: float = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an actigraphy-like synthetic time-series.

    Combines sinusoidal structure, baseline variation, random noise,
    and sparse spikes.  Returns ``(x, y)`` arrays.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 14, n)

    phase = rng.uniform(0, 2 * np.pi)
    y = baseline + 0.0  # ensure float
    y = y + amplitude * np.maximum(0, np.sin(2 * np.pi * x + phase))
    y = y + 0.7 * amplitude * np.maximum(0, np.sin(4 * np.pi * x + phase / 2))
    y = y + 15 * np.sin(2 * np.pi * x / 7 + phase)

    y = y + rng.normal(0, 10, n)

    spike_positions = rng.choice(np.arange(20, n - 20), size=max(8, n // 55), replace=False)
    for pos in spike_positions:
        width = rng.integers(3, 16)
        height = rng.uniform(35, spike_scale)
        lo = max(0, pos - width)
        hi = min(n, pos + width)
        shape = np.exp(-0.5 * ((np.arange(lo, hi) - pos) / max(1, width / 3)) ** 2)
        y[lo:hi] += height * shape

    return x, np.clip(y, 0, 300)


# ===================================================================
# Multi-layout test generation
# ===================================================================

def generate_noisy_example_plots(
    output_dir: str | Path = "test_figures",
    seed: int = 123,
    show: bool = False,
) -> tuple[list[Path], Path]:
    """Create several synthetic layouts with ground truth.

    Returns ``(generated_image_paths, ground_truth_csv_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed)

    colours = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#17becf",  # cyan
        "#e377c2",  # pink
        "#bcbd22",  # olive/yellow-green
    ]

    layouts = [
        ("vertical_2_panels", 2, 1, (8.5, 7.0)),
        ("horizontal_3_panels", 1, 3, (12.0, 3.4)),
        ("grid_2x2_panels", 2, 2, (10.0, 7.0)),
        ("wide_colours_2x3", 2, 3, (13.0, 6.4)),
    ]

    all_truth_rows: list[dict] = []
    generated_paths: list[Path] = []

    for layout_index, (name, nrows, ncols, figsize) in enumerate(layouts):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(wspace=0.28, hspace=0.35)

        for panel_i, ax in enumerate(axes.flat):
            colour = colours[(layout_index * 3 + panel_i) % len(colours)]
            x, y = make_synthetic_series(
                n=900,
                seed=seed + layout_index * 20 + panel_i,
                baseline=float(rng.uniform(35, 80)),
                amplitude=float(rng.uniform(25, 55)),
                spike_scale=float(rng.uniform(60, 150)),
            )

            ax.plot(x, y, color=colour, linewidth=1.15, alpha=0.95)
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 300)
            ax.set_yticks([0, 100, 200, 300])
            ax.set_xticks(np.arange(0.5, 14.5, 1.0))

            labels = []
            for day in range(1, 8):
                labels.extend([f"{day}N", f"{day}D"])
            ax.set_xticklabels(labels, fontsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, color="0.82", linewidth=0.7)
            ax.set_title(f"Panel {panel_i + 1}", fontsize=9)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_color("0.15")

            for xx, yy in zip(x, y):
                all_truth_rows.append(
                    {
                        "example": name,
                        "panel_expected": panel_i + 1,
                        "x": xx,
                        "y": yy,
                        "colour": colour,
                    }
                )

        clean_path = output_dir / f"{name}_clean.png"
        noisy_path = output_dir / f"{name}_noisy.jpg"
        fig.savefig(clean_path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        add_image_noise(
            clean_path,
            noisy_path,
            noise_sigma=8 + layout_index * 2,
            salt_pepper=0.0015 + layout_index * 0.0006,
            jpeg_quality=84,
            seed=seed + layout_index,
        )
        generated_paths.append(noisy_path)

    truth = pd.DataFrame(all_truth_rows)
    truth_path = output_dir / "synthetic_ground_truth.csv"
    truth.to_csv(truth_path, index=False)

    if show:
        for p in generated_paths:
            img = Image.open(p)
            plt.figure(figsize=(10, 5))
            plt.imshow(img)
            plt.axis("off")
            plt.title(str(p))
            plt.show()

    return generated_paths, truth_path
