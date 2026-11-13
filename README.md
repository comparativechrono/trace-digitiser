# trace-digitiser

Template-free computer-vision digitisation pipeline for raster scientific line plots.

Given a raster image of a scientific figure (PNG, JPEG, etc.), this tool detects plot panels, reads y-axis tick labels via OCR, segments coloured traces, and exports digitised data as CSV files — all without requiring manual calibration points, known trace colours, or panel coordinates. The user supplies only a high-level layout hint (e.g. "stacked", "horizontal", "grid").

## Installation

Requires Python 3.10+ and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

```bash
# Install Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Install Tesseract (macOS)
brew install tesseract

# Install the package
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick start

### Python API

```python
from trace_digitiser import digitise

result = digitise(
    "figure.jpg",
    layout_mode="stacked",
    expected_rows=2,
    expected_cols=1,
    output_dir="outputs",
)

# Digitised traces as a DataFrame
print(result.trace_data.head())

# Panel metadata
for panel in result.panels:
    print(f"Panel {panel.panel_id}: {panel.width}×{panel.height}px, "
          f"calibration={panel.calibration.scale_type}")
```

### Command line

```bash
# Single image
trace-digitiser figure.jpg --layout stacked --rows 2 --cols 1 -o outputs/

# Batch processing
trace-digitiser figures/*.jpg --layout auto -o results/

# Generate synthetic test figures
trace-digitiser --generate-test-figures -o test_figures/
```

### Layout modes

| Mode | Use case | Example |
|------|----------|---------|
| `single` | One plot panel | `--layout single` |
| `stacked` | Vertically stacked panels | `--layout stacked --rows 2 --cols 1` |
| `horizontal` | Side-by-side panels | `--layout horizontal --rows 1 --cols 3` |
| `grid` | Row×column subplot grid | `--layout grid --rows 2 --cols 2` |
| `auto` | Unconstrained detection | `--layout auto` |

## Output files

For input `figure.jpg` with default settings:

| File | Description |
|------|-------------|
| `figure_automated_digitised_trace_by_column.csv` | One row per x-pixel column per trace, with `center_estimate`, `upper_envelope`, `lower_envelope` |
| `figure_automated_panel_metadata.csv` | Panel coordinates, calibration parameters, detected x-labels |
| `figure_automated_digitised_summary_by_label.csv` | Summary statistics per detected x-label interval (written only if labels are found) |

## Project structure

```
trace_digitiser/
├── pyproject.toml
├── src/trace_digitiser/
│   ├── __init__.py          # Public API: digitise()
│   ├── models.py            # Dataclasses: Panel, Calibration, Trace, DigitiserResult
│   ├── io.py                # Image loading, CSV output
│   ├── geometry.py          # Box area/IoU/containment helpers
│   ├── line_detection.py    # Horizontal and vertical line detectors
│   ├── panel_detection.py   # Candidate generation, deduplication, layout selection
│   ├── calibration.py       # Y-axis OCR calibration (linear + log) and cross-row propagation
│   ├── x_calibration.py     # X-axis OCR calibration (numeric ticks)
│   ├── ocr.py               # Tesseract OCR wrappers for tick and x-label reading
│   ├── trace_detection.py   # HSV/CIELAB colour segmentation + achromatic trace detection
│   ├── digitise.py          # Column-by-column trace digitisation
│   ├── summarise.py         # Interval-label summarisation
│   ├── diagnostics.py       # Debug overlays and diagnostic file output
│   ├── synthetic.py         # Synthetic test-figure generation
│   └── cli.py               # Command-line interface
└── tests/
    ├── conftest.py           # Shared fixtures (synthetic images)
    ├── test_geometry.py
    ├── test_panel_detection.py
    ├── test_trace_detection.py
    └── test_integration.py   # Panel count, calibration quality, trace RMSE
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=trace_digitiser

# Lint
ruff check src/ tests/
```

## Limitations

This tool digitises visible pixels from raster images. It does not recover original raw data. Key limitations:

- **Linear and log y-axes supported** — broken, dual, and other nonlinear axes are not supported yet.
- **X-axis calibration is best-effort** — numeric x-ticks are OCR'd when possible; otherwise x-values are normalised 0–1 or labelled by interval.
- **Black/grey traces partially supported** — achromatic trace detection is available but works best when traces have enough contrast against the background.
- **Requires visible structure** — axes, borders, or gridlines must be present for panel detection.
- **OCR fragility** — small, rotated, or low-contrast tick labels may fail.

## Changelog

### v0.1.0

**Refactored from notebook to installable package** with 15 modules, dataclasses, CLI, and test suite.

**Panel detection improvements:**
- Hint-aware `split_y_clusters` — uses the user's `expected_rows` to split evenly-spaced gridlines at the largest gaps, even when the inter-panel gap is only marginally wider than intra-panel gaps.
- Post-hoc panel subdivision — `apply_layout_hint` can split oversized candidates at their gridlines when initial detection yields fewer panels than expected.
- Improved grid detection — properly cross-matches row/column structure.

**Trace detection improvements:**
- CIELAB clustering mode (`use_lab=True`) — clusters non-background pixels in perceptually uniform colour space using mini-batch k-means.
- Achromatic trace detection — a separate pass detects black/grey traces by looking for low-saturation, horizontally continuous structures distinct from grid/axis lines.

**Calibration improvements:**
- Log-scale y-axis detection — when OCR'd tick values are better explained by `log10(y_value) = a * y_pixel + b`, the calibration uses log scale automatically.
- X-axis calibration — OCRs numeric x-axis ticks, fits `x_value = a * x_pixel + b`, and adds `x_value` column to the trace CSV.

**Infrastructure:**
- Diagnostic overlays save to files via `save_diagnostics=True`.
- Tesseract error handling — graceful recovery from OCR crashes on degenerate crops.
- No Colab dependency.
- Quantitative test suite with panel count, calibration quality, and trace RMSE metrics against synthetic ground truth.
