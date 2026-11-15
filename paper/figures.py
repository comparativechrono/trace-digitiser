"""The figures of Hearn & Lloyd-Owen (revised): Fig 1 (reading a figure), Fig 2 (the activity-trace validation),
Fig 3 (the CD157 re-analysis) and S1 Fig (the pale trace and the gridlines), as PLOS ONE asks for them: TIFF, RGB,
LZW-compressed, 300 dpi, at most 7.5 in wide; a PNG of each beside it, for looking at.

    python paper/figures.py OUT_DIR

Run validation/activity-traces/run.py and validation/activity-traces/cd157.py first. The two screenshots of the
workbench (paper/workbench-marked.png, paper/workbench-both-colours.png) were taken with workbench_shots.js (see
paper/README.md).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
AT = HERE.parent / "validation" / "activity-traces"
sys.path.insert(0, str(AT))
import cd157  # noqa: E402

# the reference palette's categorical slots (see the dataviz notes): authors' data / wild type blue, read orange,
# knockout red; text and rules in ink, muted and grid greys
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
for f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
    p = Path("/usr/share/fonts/truetype/liberation") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "savefig.facecolor": "white"})
FIG4 = AT / "figures" / "PMC4001052-fig4.jpg"
CFG4 = json.loads((AT / "configs" / "PMC4001052-fig4.json").read_text())


def style(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    if grid:
        ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", color=INK)


def save(fig, out, name):
    png = out / f"{name}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    w_in = im.width / 300
    assert w_in <= 7.5 + 1e-6, f"{name} is {w_in:.2f} in wide"
    assert im.height / 300 <= 8.75 + 1e-6, f"{name} is {im.height / 300:.2f} in tall"
    im.save(out / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    print(f"{name}: {im.width} x {im.height} px ({w_in:.2f} x {im.height / 300:.2f} in)")


def panel_crop(panel, x0_h, x1_h):
    """Pixels of one plot area of Lopatina et al. 2014 Fig 4 (PMC's copy) between two times (h), and the extent of
    that crop in (hours, cm/s)."""
    p = next(q for q in CFG4["panels"] if q["name"] == panel)
    left, top, right, bottom = p["plot_area"]
    (px0, v0), (px1, v1) = p["x_ref"]
    to_px = lambda h: px0 + (h - v0) * (px1 - px0) / (v1 - v0)  # noqa: E731
    c0, c1 = int(np.floor(to_px(x0_h))), int(np.ceil(to_px(x1_h)))
    r0, r1 = int(np.floor(top)), int(np.ceil(bottom))
    rgb = np.asarray(Image.open(FIG4).convert("RGB"))[r0:r1 + 1, c0:c1 + 1]
    to_h = lambda c: v0 + (c - px0) * (v1 - v0) / (px1 - px0)  # noqa: E731
    to_v = lambda r: (bottom - r) / (bottom - top) * 300.0  # noqa: E731
    return rgb, [to_h(c0 - 0.5), to_h(c1 + 0.5), to_v(r1 + 0.5), to_v(r0 - 0.5)], p


def fig1(out):
    """Reading a figure: the workbench, and the steps on the wild type's trace of Lopatina et al. 2014 Fig 4A."""
    fig = plt.figure(figsize=(7.5, 7.6))
    shot = np.asarray(Image.open(HERE / "workbench-marked.png").convert("RGB"))
    ax = fig.add_axes([0.03, 0.41, 0.97, 0.59])
    ax.imshow(shot, interpolation="lanczos")
    ax.set_axis_off()
    fig.text(0.002, 0.998, "A", fontsize=10, fontweight="bold", va="top", ha="left", color=INK)
    rgb, ext, p = panel_crop("A CD157+/+", 0, 72)
    colour = np.array([int(p["series"]["CD157+/+"][i:i + 2], 16) for i in (1, 3, 5)], float)
    dist = np.sqrt(((rgb.astype(float) - colour) ** 2).sum(axis=2))
    mask = dist <= 60  # the line reader's colour tolerance for a colour the user gives
    d = pd.read_csv(AT / "digitised" / "PMC4001052-fig4.csv")
    d = d[(d.series == "CD157+/+") & (d.x >= 0) & (d.x <= 72)]
    hx, hy = cd157.series("PMC4001052-fig4")["CD157+/+"]
    keep = hx <= 72
    titles = ["B  The figure (Fig 4A, days 1-3)", "C  Pixels of the clicked colour",
              "D  Values per column and per hour"]
    for i in range(3):
        ax = fig.add_axes([0.075 + i * 0.315, 0.075, 0.27, 0.26])
        if i == 0:
            ax.imshow(rgb, extent=ext, aspect="auto", interpolation="nearest")
        elif i == 1:
            show = np.full(mask.shape + (3,), 255, np.uint8)
            show[mask] = (20, 60, 140)
            ax.imshow(show, extent=ext, aspect="auto", interpolation="nearest")
        else:
            faded = (255 - 0.28 * (255 - rgb.astype(float))).astype(np.uint8)
            ax.imshow(faded, extent=ext, aspect="auto", interpolation="nearest")
            ax.plot(d.x, d.y, color=ORANGE, lw=0.8, label="per column")
            ax.plot(hx[keep], hy[keep], "o", ms=2.6, color=INK, mec="white", mew=0.4, label="per hour")
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="none", handlelength=1.0,
                      borderpad=0.25, fontsize=6.5, handletextpad=0.3, labelspacing=0.2)
        ax.set_xlim(0, 72)
        ax.set_ylim(0, 300)
        ax.set_xticks([0, 24, 48, 72])
        ax.set_yticks([0, 100, 200, 300])
        ax.set_xlabel("hours from the first lights-off")
        if i == 0:
            ax.set_ylabel("activity (cm/s)")
        else:
            ax.set_yticklabels([])
        ax.set_title(titles[i], loc="left", color=INK, pad=4)
        style(ax, grid=False)
    save(fig, out, "Fig1")


def fig2(out):
    """The activity-trace validation: two traces as read beside the authors' data, and the dark-phase share and
    acrophase of all 29 series with data."""
    sampled = pd.read_csv(AT / "results" / "sampled.csv")
    comp = pd.read_csv(AT / "results" / "comparisons.csv")
    comp = comp[comp["true_cos_p"].notna()]
    examples = [("PMC9465459-fig2", "A locomotor", "Control", "Re et al. 2022, Fig 2A (control mice)", "beam breaks"),
                ("PMC6379088-fig6", "A LMA", "KO", "Hoekstra et al. 2019, Fig 6A (Cirbp knockout)",
                 "movements per hour")]
    fig = plt.figure(figsize=(7.5, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.45], hspace=0.62, wspace=0.3, left=0.085, right=0.98,
                          top=0.95, bottom=0.075)
    for i, (case, panel, series, title, unit) in enumerate(examples):
        ax = fig.add_subplot(gs[i, :])
        s = sampled[(sampled.case == case) & (sampled.panel == panel) & (sampled.series == series)].sort_values("x")
        ax.plot(s.x, s.original, color=BLUE, lw=1.4, label="authors' data")
        ax.plot(s.x, s.digitised, color=ORANGE, lw=1.4, label="read from the figure")
        r = comp[(comp.case == case) & (comp.panel == panel) & (comp.series == series)].iloc[0]
        ax.set_title(f"{title}: r = {r['r']:.2f}, mean absolute error {r['mae_pct']:.1f}% of the range",
                     loc="left", color=INK, pad=4)
        ax.set_xlabel("hours")
        ax.set_ylabel(unit)
        style(ax)
        letter(ax, "AB"[i])
        if i == 0:
            ax.legend(frameon=False, loc="upper right", ncol=2)
    ax = fig.add_subplot(gs[2, 0])
    lo, hi = 0.35, 0.9
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
    ax.scatter(comp["true_dark_fraction"], comp["dig_dark_fraction"], s=22, color=ORANGE, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("from the authors' data")
    ax.set_ylabel("from the figure")
    n05 = int((comp["dark_fraction_diff"].abs() <= 0.05).sum())
    ax.set_title(f"Share of activity in the dark ({n05} of {len(comp)} within 0.05)", loc="left", color=INK, pad=4)
    style(ax)
    letter(ax, "C")
    ax = fig.add_subplot(gs[2, 1])
    dd = comp["acrophase_diff_h"].to_numpy(float)
    order = np.argsort(dd)
    ax.axvspan(-1, 1, color=SHADE, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter(dd[order], np.arange(len(dd)), s=22, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks([])
    ax.set_ylabel(f"the {len(dd)} series, in order")
    ax.set_xlabel("acrophase from the figure minus from the data (h)")
    n1 = int((np.abs(dd) <= 1).sum())
    ax.set_title(f"24-h cosinor acrophase ({n1} of {len(dd)} within 1 h)", loc="left", color=INK, pad=4)
    style(ax)
    ax.grid(axis="y", visible=False)
    letter(ax, "D")
    save(fig, out, "Fig2")


def fig3(out):
    """The CD157 re-analysis: each genotype's hourly values drawn over the published panel, the mean day, and the
    acrophase and dark-phase share of each day, from both copies of the figure."""
    read = cd157.series("PMC4001052-fig4-gridlines-left-out")
    days = pd.read_csv(AT / "results" / "cd157_daily.csv")
    days = days[days.reading == "primary"]
    fig = plt.figure(figsize=(7.5, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.42, left=0.075, right=0.985,
                          top=0.955, bottom=0.07)
    names = {"CD157+/+": "CD157+/+ (wild type)", "CD157-/-": "CD157−/− (knockout)"}
    for i, (panel, name) in enumerate(cd157.GENOTYPES):
        ax = fig.add_subplot(gs[i, :])
        rgb, ext, _ = panel_crop(panel, 0, 168)
        ax.imshow(rgb, extent=ext, aspect="auto", interpolation="lanczos")
        x, y = read[name]
        ax.plot(x, y, color=INK, lw=0.9)
        ax.set_xlim(0, 168)
        ax.set_ylim(0, 300)
        ax.set_xticks(range(0, 169, 24))
        ax.set_yticks([0, 100, 200, 300])
        ax.set_ylabel("activity (cm/s)")
        ax.set_title(f"{names[name]}: Lopatina et al. 2014, Fig 4{'AB'[i]}; black, the hourly values read from it",
                     loc="left", color=INK, pad=4)
        style(ax, grid=False)
        letter(ax, "AB"[i])
        if i == 1:
            ax.set_xlabel("hours from the first lights-off (the figure's 1N, 1D, ... 7D)")
    colours = {"CD157+/+": BLUE, "CD157-/-": RED}
    ax = fig.add_subplot(gs[2, 0])
    ax.axvspan(0, 12, color=SHADE, lw=0)
    for _, name in cd157.GENOTYPES:
        x, y = read[name]
        week = y.reshape(7, 24)
        ax.fill_between(np.arange(24), np.nanmin(week, 0), np.nanmax(week, 0), color=colours[name], alpha=0.15, lw=0)
        ax.plot(np.arange(24), np.nanmean(week, 0), color=colours[name], lw=1.5, label=names[name].split(" ")[0])
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_ylim(0, None)
    ax.set_xlabel("hours from lights-off")
    ax.set_ylabel("activity (cm/s)")
    ax.set_title("Mean day (band: range of 7 days)", loc="left", color=INK, pad=4)
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    letter(ax, "C")
    for j, (col, ylab, title, ylim) in enumerate([
            ("acrophase_h_after_dark_onset", "acrophase (h from lights-off)", "24-h cosinor acrophase, each day",
             (0, 24)),
            ("dark_share", "share of the day's activity in the dark", "Dark-phase share, each day",
             (0.3, 0.8))]):
        ax = fig.add_subplot(gs[2, 1 + j])
        if j == 0:
            ax.axhspan(0, 12, color=SHADE, lw=0)
        else:
            ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        for copy, marker in zip(cd157.COPIES, ("o", "s")):
            for _, name in cd157.GENOTYPES:
                g = days[(days["copy"] == copy) & (days.genotype == name)].sort_values("day")
                filled = marker == "o"
                ax.plot(g.day + (-0.12 if filled else 0.12), g[col], marker=marker, ms=4.2, lw=0,
                        color=colours[name], mfc=colours[name] if filled else "white", mec=colours[name], mew=1.0,
                        label=f"{names[name].split(' ')[0]}, {copy.split(' (')[0]}")
        ax.set_ylim(*ylim)
        if j == 0:
            ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_xticks(range(1, 8))
        ax.set_xlim(0.5, 7.5)
        ax.set_xlabel("day")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", color=INK, pad=4)
        style(ax)
        letter(ax, "DE"[j])
        if j == 0:
            ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.2, handletextpad=0.1,
                      columnspacing=0.6, borderaxespad=0.2)
    save(fig, out, "Fig3")


def s1_fig(out):
    """The pale trace and the gridlines: where the line reader took a gridline or the frame for the knockout's trace,
    and what the workbench draws when both colours are given for both panels."""
    fig = plt.figure(figsize=(7.5, 7.4))
    d_read = pd.read_csv(AT / "digitised" / "PMC4001052-fig4.csv")
    d_left = pd.read_csv(AT / "digitised" / "PMC4001052-fig4-gridlines-left-out.csv")
    bad = pd.read_csv(AT / "results" / "cd157_misread.csv")
    bad = bad[bad.case == "PMC4001052-fig4"]
    windows = [(-0.6, 3), (14.5, 20), (65.5, 71), (162, 168)]
    for i, (a, b) in enumerate(windows):
        ax = fig.add_axes([0.07 + i * 0.235, 0.66, 0.2, 0.3])
        rgb, ext, _ = panel_crop("B CD157-/-", a, b)
        ax.imshow(rgb, extent=ext, aspect="auto", interpolation="nearest")
        for dd, colour, lab, ms in ((d_read, ORANGE, "as read", 3.4), (d_left, INK, "gridlines left out", 2.2)):
            g = dd[(dd.series == "CD157-/-") & (dd.x >= a) & (dd.x <= b)]
            ax.plot(g.x, g.y, "o", ms=ms, color=colour, mec="white", mew=0.3, label=lab)
        g = bad[(bad.x_h >= a) & (bad.x_h <= b)]
        ax.plot(g.x_h, g.y_read, "x", ms=7, mew=1.2, color=RED, label="value on grey")
        ax.set_xlim(a, b)
        ax.set_ylim(0, 300)
        ax.set_yticks([0, 100, 200, 300])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("activity (cm/s)")
        ax.set_xlabel("hours")
        style(ax, grid=False)
        if i == 0:
            letter(ax, "A", x=-0.25)
        if i == 3:
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=6.5,
                      handletextpad=0.2, borderpad=0.3)
    shot = np.asarray(Image.open(HERE / "workbench-both-colours.png").convert("RGB"))
    ax = fig.add_axes([0.03, 0.0, 0.97, 0.58])
    ax.imshow(shot, interpolation="lanczos")
    ax.set_axis_off()
    fig.text(0.002, 0.585, "B", fontsize=10, fontweight="bold", va="top", ha="left", color=INK)
    save(fig, out, "S1_Fig")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out")
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(out)
    fig3(out)
    s1_fig(out)
