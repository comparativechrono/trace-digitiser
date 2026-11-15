"""The figures of Hearn & Lloyd-Owen (revised): Fig 1 (reading a figure), Fig 2 (the activity-trace validation),
Fig 3 (the CD157 re-analysis) and S1 Fig (reading a dense trace), as PLOS ONE asks for them: TIFF, RGB,
LZW-compressed, 300 dpi, at most 7.5 in wide; a PNG of each beside it, for looking at.

    python paper/figures.py OUT_DIR

Run validation/activity-traces/run.py, validation/activity-traces/cd157.py and validation/dense-traces/run.py first.
The screenshot of the workbench (paper/workbench-marked.png) was taken with workbench_shots.js (see paper/README.md).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
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
    """The CD157 re-analysis: each genotype's trace as read (each pixel column from bottom to top) with its hourly
    values, the mean day, and the acrophase and dark-phase share of each day, from both copies of the figure."""
    case = cd157.COPIES[next(iter(cd157.COPIES))]["left out"]
    cols = cd157.columns(case)
    read = cd157.series(case)
    days = pd.read_csv(AT / "results" / "cd157_daily.csv")
    days = days[days.reading == "primary"]
    fig = plt.figure(figsize=(7.5, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.42, left=0.075, right=0.985,
                          top=0.955, bottom=0.07)
    names = {"CD157+/+": "CD157+/+ (wild type)", "CD157-/-": "CD157−/− (knockout)"}
    colours = {"CD157+/+": BLUE, "CD157-/-": RED}
    for i, (_, name) in enumerate(cd157.GENOTYPES):
        ax = fig.add_subplot(gs[i, :])
        for k in range(7):
            ax.axvspan(24 * k, 24 * k + 12, color=SHADE, lw=0, zorder=0)
        g = cols[(cols.series == name) & ~cols.interpolated.astype(bool)]
        ax.vlines(g.x, g.bottom, g.top, color=colours[name], alpha=0.55, lw=0.55, zorder=1)
        x, y = read[name]
        ax.plot(x, y, color=INK, lw=0.9, zorder=2)
        ax.set_xlim(0, 168)
        ax.set_ylim(0, 300)
        ax.set_xticks(range(0, 169, 24))
        ax.set_yticks([0, 100, 200, 300])
        ax.set_ylabel("activity (cm/s)")
        ax.set_title(f"{names[name]}, Lopatina et al. 2014, Fig 4{'AB'[i]}, as read: each pixel column from bottom "
                     "to top, and the hourly values (black)", loc="left", color=INK, pad=4)
        style(ax, grid=False)
        letter(ax, "AB"[i])
        if i == 1:
            ax.set_xlabel("hours from the first lights-off (the figure's 1N, 1D, ... 7D; grey, dark)")
    ax = fig.add_subplot(gs[2, 0])
    ax.axvspan(0, 12, color=SHADE, lw=0)
    for _, name in cd157.GENOTYPES:
        x, y = read[name]
        week = y.reshape(7, 24)
        ax.fill_between(np.arange(24) + 0.5, np.nanmin(week, 0), np.nanmax(week, 0), color=colours[name],
                        alpha=0.15, lw=0)
        ax.plot(np.arange(24) + 0.5, np.nanmean(week, 0), color=colours[name], lw=1.5,
                label=names[name].split(" ")[0])
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
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


STAT = {"top": ("top", "#eb6834"), "midpoint": ("midpoint", INK), "mean of pixels": ("mean of pixels", "#1baf7a"),
        "bottom": ("bottom", "#6250d6")}  # orange, ink, aqua, violet: the column statistics, in every panel
DT = HERE.parent / "validation" / "dense-traces"
SHORT_READING = {"bins on the hour": "bins on hour", "nearest the hour": "nearest hour", "mean of pixels": "mean"}


def s1_fig(out):
    """Reading a dense trace: what each pixel column of the CD157 figure holds, real records drawn the same way and
    read back, and the CD157 result in every reading."""
    fig = plt.figure(figsize=(7.5, 8.6))
    case = cd157.COPIES[next(iter(cd157.COPIES))]["left out"]
    cols = cd157.columns(case)
    cfg = json.loads((AT / "configs" / f"{case}.json").read_text())
    # A, B: a stretch of each panel with the four values of every column
    for i, ((panel, name), (a, b), ymax) in enumerate(zip(cd157.GENOTYPES, [(26, 38), (24, 36)], [250, 150])):
        ax = fig.add_axes([0.075 + i * 0.475, 0.775, 0.42, 0.19])
        rgb, ext, p = panel_crop(panel, a, b)
        faded = (255 - 0.55 * (255 - rgb.astype(float))).astype(np.uint8)
        ax.imshow(faded, extent=ext, aspect="auto", interpolation="nearest", zorder=0)
        if i == 1:  # the rows and columns left out of panel B, in value units
            left, top, right, bottom = p["plot_area"]
            for x0, y0, x1, y1 in next(q for q in cfg["panels"] if q["name"] == panel)["exclude"]:
                if x1 - x0 > 50:
                    v0, v1 = (bottom - y1) / (bottom - top) * 300, (bottom - y0) / (bottom - top) * 300
                    ax.axhspan(v0, v1, color="white", alpha=0.85, lw=0, zorder=1)
        g = cols[(cols.series == name) & (cols.x >= a) & (cols.x <= b)]
        for st, mk, ms in (("top", "_", 6), ("bottom", "_", 6), ("midpoint", "o", 2.4)):
            lab, c = STAT[st]
            ax.plot(g.x, g[st], mk, ms=ms, mew=1.3, color=c, label=lab, zorder=3)
        ax.set_xlim(a, b)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("hours from the first lights-off")
        if i == 0:
            ax.set_ylabel("activity (cm/s)")
        ax.set_title(f"{'Wild type' if i == 0 else 'Knockout'} (Fig 4{'AB'[i]}), hours {a}-{b}", loc="left",
                     color=INK, pad=4)
        style(ax, grid=False)
        letter(ax, "AB"[i], x=-0.1 if i == 0 else -0.04)
        if i == 0:
            ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5,
                      handletextpad=0.2, columnspacing=0.8, borderpad=0.3)
    # C: one of the test's images; D: its hourly values, true and read
    rec, size = "locomotor, mean of 3", "PMC-sized"
    im = np.asarray(Image.open(DT / "figures" / "locomotor_mean_of_3-PMC-sized-blue.jpg").convert("RGB"))
    r = pd.read_csv(DT / "results" / "readings.csv")
    meta = r[(r.record == rec) & (r["size"] == size) & (r.colour == "blue")].iloc[0]
    m, w7, h = 20, 551, 217
    ax = fig.add_axes([0.075, 0.585, 0.9, 0.12])
    days3 = int(round(m + w7 * 72 / 168))
    ax.imshow(im[m - 2:m + h + 3, m - 1:days3 + 1], extent=[0, 72, 0, meta.y_max * (h + 5) / h], aspect="auto",
              interpolation="nearest")
    ax.set_xlim(0, 72)
    ax.set_xticks(range(0, 73, 12))
    ax.set_yticks([])
    ax.set_title("Test: the mean of three mice's home-cage activity in 5-min bins (Hirano et al. 2026), drawn as "
                 "Fig 4 is drawn (first 3 of 7 days)", loc="left", color=INK, pad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    letter(ax, "C", x=-0.01)
    hh = pd.read_csv(DT / "results" / "hourly.csv")
    hh = hh[(hh.record == rec) & (hh["size"] == size) & (hh.colour == "blue") & (hh.hour < 72)]
    ax = fig.add_axes([0.075, 0.385, 0.9, 0.15])
    t = hh[hh.statistic == "midpoint"]
    ax.plot(t.hour + 0.5, t.true, color="#9a9a96", lw=3.0, solid_capstyle="butt", label="the record's hourly means")
    for st in ("top", "midpoint", "bottom"):
        e = hh[hh.statistic == st]
        lab, c = STAT[st]
        ax.plot(e.hour + 0.5, e.read, color=c, lw=0.9, label=f"{lab} of the columns, hourly")
    ax.set_xlim(0, 72)
    ax.set_xticks(range(0, 73, 12))
    ax.set_ylim(0, 1.25 * np.nanmax(hh.read))
    ax.set_xlabel("hours")
    ax.set_ylabel("counts per 5 min")
    ax.set_title("Read back: hourly means of the record and of each column statistic", loc="left", color=INK, pad=4)
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=1.0,
              bbox_to_anchor=(0.5, 1.0))
    style(ax)
    letter(ax, "D", x=-0.01)
    # E: the test summarised over its 36 readings
    order = ["top", "midpoint", "mean of pixels", "bottom"]
    short = ["top", "mid", "mean", "bottom"]
    specs = [("level", "level, read ÷ true", True), ("dark_share_error", "dark share,\nread − true", False),
             ("acrophase_error_h", "|acrophase error| (h)", False)]
    for k, (col, lab, logy) in enumerate(specs):
        ax = fig.add_axes([0.075 + k * 0.17, 0.1, 0.1, 0.2])
        for j, st in enumerate(order):
            v = r.loc[r.statistic == st, col].to_numpy(float)
            if col == "acrophase_error_h":
                v = np.abs(v)
            jit = (np.random.default_rng(j).random(len(v)) - 0.5) * 0.4
            ax.plot(np.full(len(v), j) + jit, v, "o", ms=2.0, color=STAT[st][1], alpha=0.75, mew=0)
            ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=INK, lw=1.1)
        ax.axhline(1.0 if logy else 0.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        if logy:
            ax.set_yscale("log")
            ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
            ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(range(4))
        ax.set_xticklabels(short, rotation=90, fontsize=6.5)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylabel(lab, fontsize=7)
        style(ax)
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_title("Test, 36 readings", loc="left", color=INK, pad=4)
            letter(ax, "E", x=-0.35)
    # F: the CD157 result in every reading
    sm = pd.read_csv(AT / "results" / "cd157_summary.csv")
    readings = list(dict.fromkeys(sm.reading))
    rcol = {"top": STAT["top"][1], "mean of pixels": STAT["mean of pixels"][1], "bottom": STAT["bottom"][1]}
    for k, (fn, lab, ylim) in enumerate([
            (lambda q: q.acrophase_ko_minus_wt_h, "acrophase, −/− minus +/+ (h)", (-8, 0)),
            (lambda q: q.dark_share_ko - q.dark_share_wt, "dark share, −/− minus +/+", (0, 0.25))]):
        ax = fig.add_axes([0.635 + k * 0.205, 0.1, 0.15, 0.2])
        for j, rd in enumerate(readings):
            for cp, mk, dx in zip(cd157.COPIES, ("o", "s"), (-0.15, 0.15)):
                q = sm[(sm.reading == rd) & (sm["copy"] == cp)].iloc[0]
                c = rcol.get(rd, INK)
                ax.plot(j + dx, fn(q), mk, ms=3.8, color=c, mfc=c if mk == "o" else "white", mec=c, mew=1.0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(readings)))
        ax.set_xticklabels([SHORT_READING.get(x, x) for x in readings], rotation=90, fontsize=6.5)
        ax.set_xlim(-0.6, len(readings) - 0.4)
        ax.set_ylabel(lab, fontsize=7)
        style(ax)
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_title("CD157, every reading", loc="left", color=INK, pad=4)
            letter(ax, "F", x=-0.33)
            ax.text(0.03, 0.04, "● PMC copy  □ journal copy", transform=ax.transAxes, fontsize=6, color=MUTED)
    save(fig, out, "S1_Fig")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out")
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(out)
    fig3(out)
    s1_fig(out)
