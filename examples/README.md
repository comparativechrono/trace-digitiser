# Examples

| File | What it shows |
|------|---------------|
| `digitise_published_figure.py` | Digitising a two-panel figure from a published paper, with the axis limits typed in. |
| `validate_with_known_data.py` | Finding out how accurately a figure of a given kind *can* be digitised, by drawing data you know and digitising the drawing. |
| `walkthrough.ipynb` | The same two things as a notebook, with the pictures. |
| `lopatina2014_fig4.jpg` | The published figure used by the scripts, the notebook and the test suite. |

Run a script from anywhere; it writes to `examples/output/` unless given another directory:

```bash
python examples/digitise_published_figure.py
python examples/validate_with_known_data.py
```

## Source of the figure

`lopatina2014_fig4.jpg` is Figure 4 of

> Lopatina O, *et al.* (2014). Anxiety- and depression-like behavior in mice lacking the
> CD157/BST1 gene, a risk factor for Parkinson's disease. *Frontiers in Behavioral
> Neuroscience* 8:133. <https://doi.org/10.3389/fnbeh.2014.00133>

"Spontaneous homecage activity in CD157+/+ (A) and CD157−/− (B) mice", recorded for seven
days. Copyright © 2014 the authors of that article, which is an open-access article
distributed under the terms of the Creative Commons Attribution License (CC BY); the licence
is linked from the article. The file here is the figure as published, unaltered
(397 × 376 pixels; SHA-256 `c6627d607e01f5d0acfb746883d9081062c07ce0f30cfcdc5a7c5d8da57b78d3`).
Images that the examples derive from it (the overlay with the digitised traces drawn on top)
are adaptations of the figure and carry the same attribution and licence.

`tests/data/lopatina2014_fig4_reference.csv` holds values digitised from this figure by
trace-digitiser, which the test suite uses to detect unintended changes. They are a reading
of a small image, not the authors' data.

The MIT licence of this repository covers the code, not this figure.
