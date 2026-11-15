"""Plot each digitised series over the series behind it, one subplot per series.

    python plot_case.py DIGITISED.csv TRUTH.csv OUT.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main(dig, truth, out):
    d = pd.read_csv(dig)
    t = pd.read_csv(truth) if truth != "-" else None
    keys = list(d.groupby(["panel", "series"]).groups)
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 1.9 * len(keys)), squeeze=False)
    for ax, (panel, series) in zip(axes[:, 0], keys):
        g = d[(d.panel == panel) & (d.series == series)].sort_values("x")
        ax.plot(g.x, g.y, color="#c2410c", lw=1, label="digitised")
        if t is not None:
            tt = t[(t.panel == panel) & (t.series == series)].sort_values("x")
            ax.errorbar(tt.x, tt.y, yerr=tt.get("sem"), color="#1d4ed8", lw=1, marker="o", ms=2.5, elinewidth=0.6,
                        label="original data")
        ax.set_title(f"{panel}: {series}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:])
