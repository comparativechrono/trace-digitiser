"""The summary figure: two traces read from figures beside the authors' own numbers, and how the rhythm measures of
every series with data compare.

    python summary_figure.py RESULTS_DIR OUT.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORIGINAL, DIGITISED = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EXAMPLES = [("PMC9465459-fig2", "A locomotor", "Control", "Re et al. 2022, Fig 2A (control)", "beam breaks"),
            ("PMC6379088-fig6", "A LMA", "KO", "Hoekstra et al. 2019, Fig 6A (Cirbp KO)", "movements per hour")]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main(res, out):
    res = Path(res)
    sampled = pd.read_csv(res / "sampled.csv")
    comp = pd.read_csv(res / "comparisons.csv")
    comp = comp[comp["true_cos_p"].notna()]
    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.35], hspace=0.65, wspace=0.28)
    for i, (case, panel, series, title, unit) in enumerate(EXAMPLES):
        ax = fig.add_subplot(gs[i, :])
        s = sampled[(sampled.case == case) & (sampled.panel == panel) & (sampled.series == series)].sort_values("x")
        ax.plot(s.x, s.original, color=ORIGINAL, linewidth=2, label="authors' data")
        ax.plot(s.x, s.digitised, color=DIGITISED, linewidth=2, label="read from the figure")
        r = comp[(comp.case == case) & (comp.panel == panel) & (comp.series == series)].iloc[0]
        ax.set_title(f"{'AB'[i]}  {title}: r = {r['r']:.2f}, mean absolute error {r['mae_pct']:.1f}% of range",
                     loc="left", fontsize=9, color=INK)
        ax.set_xlabel("hours", fontsize=8, color=MUTED)
        ax.set_ylabel(unit, fontsize=8, color=MUTED)
        style(ax)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax = fig.add_subplot(gs[2, 0])
    x, y = comp["true_dark_fraction"], comp["dig_dark_fraction"]
    lo, hi = 0.35, 0.9
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    ax.scatter(x, y, s=36, color=DIGITISED, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("from the authors' data", fontsize=8, color=MUTED)
    ax.set_ylabel("from the figure", fontsize=8, color=MUTED)
    ax.set_title(f"C  Share of activity in the dark phase ({len(comp)} series)", loc="left", fontsize=9, color=INK)
    style(ax)
    ax = fig.add_subplot(gs[2, 1])
    d = comp["acrophase_diff_h"].to_numpy(float)
    order = np.argsort(d)
    ax.axvspan(-1, 1, color=GRID, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.scatter(d[order], np.arange(len(d)), s=36, color=DIGITISED, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("acrophase from the figure minus from the data (h)", fontsize=8, color=MUTED)
    ax.set_title(f"D  24-h cosinor acrophase ({int((np.abs(d) <= 1).sum())} of {len(d)} within 1 h)", loc="left",
                 fontsize=9, color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main(*sys.argv[1:])
