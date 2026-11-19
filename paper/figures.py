"""The figures of Hearn & Lloyd-Owen (revised), as PLOS ONE asks for them: TIFF, RGB, LZW-compressed, 300 dpi, at
most 7.5 in wide; a PNG of each beside it, for looking at.

    Fig 1   reading a figure                       S1 Fig  Validation 1: the figures of Hearn & Whitmore (2026)
    Fig 2   the activity-trace validation           S2 Fig  Validation 2: the development figures
    Fig 3   the CD157 re-analysis                   S3 Fig  Validation 2: the figures held out
                                                    S4 Fig  Validation 3: every series with data
                                                    S5 Fig  Validation 4: dense traces
                                                    S6 Fig  the columns of the CD157 figure, and every analysis of it

    python paper/figures.py OUT_DIR --workbench PATH_TO_DIGITISER_WORKBENCH

Run validation/published-figures/run.py --digitised, validation/activity-traces/run.py, cd157.py and
validation/dense-traces/run.py first. S1 Fig reads the figures of Validation 1 again, from a checkout of
digitiser-workbench (data/examples/hearn-whitmore-2026); without --workbench it is left out. The screenshot of the
workbench (paper/workbench-marked.png) was taken with workbench_shots.js (see paper/README.md).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
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


def tag(ax, s, x=0.0, y=1.02):
    """A short label above a panel, beside its letter (a genotype, say)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left", color=INK)


def fig1(out):
    """Reading a figure: the workbench, and the steps on the wild-type trace of Lopatina et al. 2014 Fig 4A."""
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
            ax.plot(d.x, d.y, color=ORANGE, lw=0.8, label="Column midpoint")
            ax.plot(hx[keep], hy[keep], "o", ms=2.6, color=INK, mec="white", mew=0.4, label="Hourly mean")
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="none", handlelength=1.0,
                      borderpad=0.25, fontsize=6.5, handletextpad=0.3, labelspacing=0.2)
        ax.set_xlim(0, 72)
        ax.set_ylim(0, 300)
        ax.set_xticks([0, 24, 48, 72])
        ax.set_yticks([0, 100, 200, 300])
        ax.set_xlabel("Time from first lights-off (h)")
        if i == 0:
            ax.set_ylabel("Activity (cm/s)")
        else:
            ax.set_yticklabels([])
        letter(ax, "BCD"[i])
        style(ax, grid=False)
    save(fig, out, "Fig1")


def fig2(out):
    """The activity-trace validation: two digitised traces beside the authors' data, and the dark-phase share and
    acrophase of all 29 series with data."""
    sampled = pd.read_csv(AT / "results" / "sampled.csv")
    comp = pd.read_csv(AT / "results" / "comparisons.csv")
    comp = comp[comp["true_cos_p"].notna()]
    examples = [("PMC9465459-fig2", "A locomotor", "Control", "Beam breaks"),
                ("PMC6379088-fig6", "A LMA", "KO", "Movements per hour")]
    fig = plt.figure(figsize=(7.5, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.45], hspace=0.55, wspace=0.3, left=0.085, right=0.98,
                          top=0.95, bottom=0.075)
    for i, (case, panel, series, unit) in enumerate(examples):
        ax = fig.add_subplot(gs[i, :])
        s = sampled[(sampled.case == case) & (sampled.panel == panel) & (sampled.series == series)].sort_values("x")
        ax.plot(s.x, s.original, color=BLUE, lw=1.4, label="Authors' data")
        ax.plot(s.x, s.digitised, color=ORANGE, lw=1.4, label="Digitised")
        ax.set_xlabel("Time (h)")
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
    ax.set_xlabel("Dark-phase share (authors' data)")
    ax.set_ylabel("Dark-phase share (digitised)")
    style(ax)
    letter(ax, "C")
    ax = fig.add_subplot(gs[2, 1])
    dd = comp["acrophase_diff_h"].to_numpy(float)
    order = np.argsort(dd)
    ax.axvspan(-1, 1, color=SHADE, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.scatter(dd[order], np.arange(len(dd)), s=22, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks([])
    ax.set_ylabel("Series (ranked)")
    ax.set_xlabel("Acrophase difference (h)")
    style(ax)
    ax.grid(axis="y", visible=False)
    letter(ax, "D")
    save(fig, out, "Fig2")


GENO_LABEL = {"CD157+/+": "CD157$^{+/+}$", "CD157-/-": "CD157$^{−/−}$"}


def fig3(out):
    """The CD157 re-analysis: each genotype's digitised trace (each pixel column from its lowest to its highest trace
    pixel) with its hourly values, the mean day, and the acrophase and dark-phase share of each day, from both
    published versions of the figure: PubMed Central's and the smaller one on the journal's website."""
    case = cd157.COPIES[next(iter(cd157.COPIES))]["left out"]
    cols = cd157.columns(case)
    read = cd157.series(case)
    days = pd.read_csv(AT / "results" / "cd157_daily.csv")
    days = days[days.reading == "primary"]
    fig = plt.figure(figsize=(7.5, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.25], hspace=0.5, wspace=0.42, left=0.075, right=0.985,
                          top=0.955, bottom=0.07)
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
        ax.set_ylabel("Activity (cm/s)")
        style(ax, grid=False)
        letter(ax, "AB"[i])
        tag(ax, GENO_LABEL[name])
        if i == 1:
            ax.set_xlabel("Time from first lights-off (h)")
    ax = fig.add_subplot(gs[2, 0])
    ax.axvspan(0, 12, color=SHADE, lw=0)
    for _, name in cd157.GENOTYPES:
        x, y = read[name]
        week = y.reshape(7, 24)
        ax.fill_between(np.arange(24) + 0.5, np.nanmin(week, 0), np.nanmax(week, 0), color=colours[name],
                        alpha=0.15, lw=0)
        ax.plot(np.arange(24) + 0.5, np.nanmean(week, 0), color=colours[name], lw=1.5, label=GENO_LABEL[name])
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0, None)
    ax.set_xlabel("Time from lights-off (h)")
    ax.set_ylabel("Activity (cm/s)")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    letter(ax, "C")
    for j, (col, ylab, ylim) in enumerate([("acrophase_h_after_dark_onset", "Acrophase (h after lights-off)", (0, 24)),
                                           ("dark_share", "Dark-phase share", (0.3, 0.8))]):
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
                        color=colours[name], mfc=colours[name] if filled else "white", mec=colours[name], mew=1.0)
        ax.set_ylim(*ylim)
        if j == 0:
            ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_xticks(range(1, 8))
        ax.set_xlim(0.5, 7.5)
        ax.set_xlabel("Day")
        ax.set_ylabel(ylab)
        style(ax)
        letter(ax, "DE"[j])
        if j == 0:  # the genotypes are told by colour (panel C); the two versions of the figure, by marker
            versions = [matplotlib.lines.Line2D([], [], marker="o", ms=4.2, lw=0, color=INK, mfc=INK, mec=INK,
                                                label="PubMed Central version"),
                        matplotlib.lines.Line2D([], [], marker="s", ms=4.2, lw=0, color=INK, mfc="white", mec=INK,
                                                mew=1.0, label="Journal website version")]
            ax.legend(handles=versions, frameon=False, loc="upper center", ncol=1, fontsize=6.4,
                      handletextpad=0.2, borderaxespad=0.2)
    save(fig, out, "Fig3")


# ------------------------------------------------------------------------------------------- supporting figures
KNOWN_C, READ_C = BLUE, ORANGE  # known values (or the authors' data) and what was digitised, in every S figure
PF = HERE.parent / "validation" / "published-figures"


def key(fig, known_label="Known values", read_label="Digitised"):
    handles = [matplotlib.lines.Line2D([], [], color=KNOWN_C, lw=4, label=known_label),
               matplotlib.lines.Line2D([], [], color=READ_C, lw=4, label=read_label)]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, columnspacing=2.0)


def cell(fig, top, height, ncol, col, span=1, left=0.075, right=0.985, gap=0.07):
    """An axes in a grid of *ncol* columns, its top at *top* (figure fraction)."""
    w = (right - left - gap * (ncol - 1)) / ncol
    return fig.add_axes([left + col * (w + gap), top - height, w * span + gap * (span - 1), height])


def finish(ax, mark, title, xlabel=None, ylabel=None, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    style(ax, grid=grid)
    ax.tick_params(labelsize=6.5)
    if mark:
        letter(ax, mark)
    tag(ax, title)


def seen(d, column="center_estimate"):
    """A trace's values where it was seen: the columns filled in from their neighbours (under a legend, say) blank."""
    y = d[column].to_numpy(float)
    if "interpolated" in d:
        y = np.where(d["interpolated"].astype(str).str.lower().isin(["true", "1"]), np.nan, y)
    return y


def thousands(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))


# ---- S1 Fig: Validation 1, the figures of Hearn & Whitmore (2026)

def heat_pairs(data, known):
    """Each cell read from the six heat maps beside the mean of the known cells it covers, paired as the workbench's
    comparison (python/dwb/known.py) pairs them."""
    names = list(dict.fromkeys(known["panel"]))
    pairs = []
    for i, pid in enumerate(sorted(data["panel_id"].unique())[:6]):
        m = known[known.panel == names[i]].sort_values("lag_months").iloc[:, 2:].to_numpy(float)
        nrow, ncol = m.shape
        sub = data[data.panel_id == pid].dropna(subset=["value"])
        ncell = int(data[data.panel_id == pid]["col"].max())
        for c in sub.itertuples():
            lag = c.y_value if np.isfinite(c.y_value) else -20 + (nrow - c.row)
            j = int(round(lag)) + 20
            if not 0 <= j < nrow:
                continue
            a = int(np.floor((c.col - 1) / ncell * ncol))
            b = max(a + 1, int(np.ceil(c.col / ncell * ncol)))
            want = np.nanmean(m[j, a:b]) if np.isfinite(m[j, a:b]).any() else np.nan
            if np.isfinite(want):
                pairs.append((want, c.value))
    return np.array(pairs)


def s1_fig(out, workbench):
    """Validation 1: each figure of Hearn & Whitmore (2026), read with the settings of Table 1, against the values
    it shows (data/examples/hearn-whitmore-2026 of digitiser-workbench)."""
    import importlib.util

    # paper/tables.py by its path (validation/activity-traces, which is on the path for cd157, has a tables.py too)
    spec = importlib.util.spec_from_file_location("paper_tables", HERE / "tables.py")
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    examples = Path(workbench) / "data" / "examples" / "hearn-whitmore-2026"
    known = lambda name: pd.read_csv(examples / f"{name}.known.csv")  # noqa: E731
    reads = {}
    for figure, _what, _kind, _given, data, _ in tables.read_own(workbench):
        reads.setdefault(figure, []).append(data)
    fig = plt.figure(figsize=(7.5, 8.7))
    key(fig)
    L, R = 0.075, 0.985
    # A, B: Fig 1a, the births line and the peak-conception month line, over the 60 years
    k1a = known("figure-1a")
    lines = [("births", "Births per month"), ("peak_conception_month", "Peak-conception month")]
    for i, (col, ylab) in enumerate(lines):
        ax = fig.add_axes([L, 0.86 - i * 0.135, R - L, 0.085])
        d = reads["figure-1a"][i]
        ax.plot(k1a.decimal_year, k1a[col], color=KNOWN_C, lw=1.0)
        ax.plot(d.x_value, seen(d), color=READ_C, lw=0.6)
        ax.set_xlim(1955, 2015)
        ax.set_ylabel(ylab, fontsize=7)
        if i:
            ax.set_yticks([6, 8, 10, 12])
            ax.set_yticklabels(["Jun", "Aug", "Oct", "Dec"])
        else:
            thousands(ax)
        style(ax)
        letter(ax, "AB"[i], x=-0.01)
        tag(ax, "Fig 1a, births (blue line)" if i == 0 else
            "Fig 1a, peak-conception month (orange line, right-hand axis)")
    # C: the inset
    top3, top4, top5 = 0.665, 0.45, 0.235
    ax = cell(fig, top3, 0.135, 3, 0, gap=0.075)
    ins = k1a[k1a.in_inset.astype(str).str.lower() == "true"]
    d = reads["figure-1a"][2]
    ax.plot(ins.decimal_year, ins.births, color=KNOWN_C, lw=1.2)
    ax.plot(d.x_value, seen(d), color=READ_C, lw=0.7)
    ax.set_xlim(1972, 1979)
    thousands(ax)
    finish(ax, "C", "Fig 1a, inset (green line)", None, "Births per month")
    # D: Fig 1b, the bars
    ax = cell(fig, top3, 0.135, 3, 1, gap=0.075)
    kb = known("figure-1b")
    d = reads["figure-1b"][0].sort_values("x_pixel")
    x = np.arange(12)
    ax.bar(x - 0.2, kb.percent_difference, width=0.38, color=KNOWN_C)
    ax.bar(x[:len(d)] + 0.2, d.value.to_numpy(float)[:12], width=0.38, color=READ_C)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in kb.label], fontsize=6.5)
    finish(ax, "D", "Fig 1b, 12 bars", None, "Difference (%)")
    ax.grid(axis="x", visible=False)
    # E: Fig 2, the birthogram's cells
    ax = cell(fig, top3, 0.135, 3, 2, gap=0.075)
    d = reads["figure-2"][0].copy()
    d["row_year"] = np.round(d["row_value"]).astype(int)
    d["column"] = d["bin"].astype(int) - 1
    g = d.groupby(["row_year", "column"], as_index=False).agg(value=("value", "median"))
    m = g.merge(known("figure-2"), on=["row_year", "column"])
    ax.plot([0, 1.1], [0, 1.1], color=MUTED, lw=0.7)
    ax.scatter(m.normalised_births, m.value, s=2, color=READ_C, alpha=0.35, lw=0, rasterized=True)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    finish(ax, "E", f"Fig 2, {len(m):,} cells of the birthogram", "Known value", "Digitised value")
    # F, G: Fig 3a, the two spectra
    k3a = known("figure-3a")
    d = reads["figure-3a"][0]
    for i, (pid, era) in enumerate(zip(sorted(d.panel_id.unique()), ("1955-1974", "1976-2014"))):
        ax = cell(fig, top4, 0.135, 3, i, gap=0.075)
        kk = k3a[k3a.panel == era]
        g = d[(d.panel_id == pid) & (d.x_value >= 2) & (d.x_value <= 20)]
        ax.plot(kk.period_months, kk.amplitude, color=KNOWN_C, lw=1.2)
        ax.plot(g.x_value, seen(g), color=READ_C, lw=0.7)
        ax.set_xlim(2, 20)
        finish(ax, "FG"[i], f"Fig 3a, spectrum {era.replace('-', '–')}", "Period (months)",
               "Amplitude" if i == 0 else None)
    # H: Fig 3b, the polar plot
    w = (R - L - 0.075 * 2) / 3
    ax = fig.add_axes([L + 2 * (w + 0.075) + 0.015, top4 - 0.16, w - 0.03, 0.14], projection="polar")
    k3b = known("figure-3b")
    d = reads["figure-3b"][0]
    ax.plot(np.radians(k3b.angle_degrees), k3b.year, "o", ms=3.6, mfc="none", mec=KNOWN_C, mew=0.8)
    ax.plot(np.radians(d.angle_degrees), d.r_value, "o", ms=1.6, color=READ_C)
    ax.set_rlim(1950, 2016)
    ax.set_rticks([1955, 1975, 1995])
    ax.set_rlabel_position(22)
    ax.set_xticks(np.radians(np.arange(0, 360, 90)))
    ax.set_xticklabels(["0", "90", "180", "270"])
    ax.tick_params(labelsize=6, pad=0)
    ax.grid(color=GRID, lw=0.5)
    ax.spines["polar"].set_color(MUTED)
    ax.spines["polar"].set_linewidth(0.6)
    ax.text(-0.1, 1.17, "H", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    ax.text(-0.08, 1.17, "Fig 3b, 60 years", transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left")
    # I: Fig 3c, the forest plot
    ax = cell(fig, top5, 0.175, 4, 0, gap=0.065)
    k3c = known("figure-3c")
    d = reads["figure-3c"][0]
    ax.hlines(k3c.window_midpoint_year - 0.22, k3c.ci_low, k3c.ci_high, color=KNOWN_C, lw=0.5)
    ax.plot(k3c.circular_sd, k3c.window_midpoint_year - 0.22, "o", ms=2.0, color=KNOWN_C, mew=0)
    ax.hlines(d.y_value + 0.22, d.x_error_low, d.x_error_high, color=READ_C, lw=0.5)
    ax.plot(d.x_value, d.y_value + 0.22, "o", ms=1.6, color=READ_C, mew=0)
    ax.invert_yaxis()
    finish(ax, "I", "Fig 3c, 51 windows", "Circular SD (rad)", "Window midpoint")
    # J: Fig 4, the six heat maps
    ax = cell(fig, top5, 0.175, 4, 1, gap=0.065)
    p = heat_pairs(reads["figure-4"][0], known("figure-4"))
    ax.plot([-0.8, 0.8], [-0.8, 0.8], color=MUTED, lw=0.7)
    ax.scatter(p[:, 0], p[:, 1], s=1, color=READ_C, alpha=0.08, lw=0, rasterized=True)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    finish(ax, "J", f"Fig 4, {len(p):,} cells of 6 maps", "Known value", "Digitised value")
    # K, L: S3 Fig, its two panels of points
    ks3 = known("figure-s3")
    d = reads["figure-s3"][0]
    for i, (pid, era) in enumerate(zip(sorted(d.panel_id.unique()), ("1953-1974", "1976-2014"))):
        ax = cell(fig, top5, 0.175, 4, 2 + i, gap=0.065)
        kk = ks3[ks3.panel == era]
        g = d[d.panel_id == pid]
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(kk.lag_months, kk.correlation, "o", ms=3.4, mfc="none", mec=KNOWN_C, mew=0.8)
        ax.plot(g.x_value, g.y_value, "o", ms=1.6, color=READ_C)
        finish(ax, "KL"[i], f"S3 Fig, {era.replace('-', '–')}", "Lag (months)", "Correlation" if i == 0 else None)
    save(fig, out, "S1_Fig")


# ---- S2 Fig and S3 Fig: Validation 2, published figures with deposited data

def case_part(name, series=None):
    """What a case of Validation 2 read in the plot compared with *series* (or its only plot): the rows of
    validation/published-figures/digitised/<case>.csv for that panel."""
    folder = PF / "digitised"
    info = json.loads((folder / f"{name}.json").read_text())
    path = folder / f"{name}.csv"
    data = pd.read_csv(path) if path.stat().st_size > 1 else pd.DataFrame(columns=["panel_id"])
    p = info["parts"][0] if series is None else next(q for q in info["parts"] if series in (q["series"] or []))
    return data[data.panel_id == p["panel_id"]] if p["panel_id"] is not None else data.iloc[0:0]


def known_points(ax, pts, ms=4.2):
    a = np.array([[p[0], p[1]] for p in pts], float)
    ax.plot(a[:, 0], a[:, 1], "o", ms=ms, mfc="none", mec=KNOWN_C, mew=0.9, zorder=2)


def read_points(ax, d, ms=2.2):
    d = d.dropna(subset=["x_value", "y_value"])
    ax.plot(d.x_value, d.y_value, "o", ms=ms, color=READ_C, mew=0, zorder=3)


def read_lines(ax, d, scale=1.0, lw=0.8):
    """The traces read, where they were seen (the columns filled in between, or beyond, are left out)."""
    d = d.dropna(subset=["x_value"]).sort_values("x_value")
    for _, g in d.groupby("series"):
        ax.plot(g.x_value, seen(g, "y_value") / scale, color=READ_C, lw=lw, zorder=3)


def paired_bars(ax, known, d, width=0.38):
    """Known bar heights (left) beside those read (right), paired from left to right as the comparison pairs them."""
    k = np.array([v for _, v, _ in known], float)
    got = d.sort_values("x_pixel")["value"].to_numpy(float) if len(d) else np.zeros(0)
    x = np.arange(len(k))
    ax.bar(x - width / 2, k, width=width, color=KNOWN_C)
    n = min(len(k), len(got))
    ax.bar(x[:n] + width / 2, got[:n], width=width, color=READ_C)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(k) - 0.4)


FOLEY = [("foley-2C-25C", "2C, 25 °C, period"), ("foley-2C-30C", "2C, 30 °C, period"), ("foley-2D", "2D, period"),
         ("foley-2E", "2E, period"), ("foley-2F-period", "2F, period"), ("foley-2F-rhythmic", "2F, rhythmic"),
         ("foley-2G-period", "2G, period"), ("foley-2G-rhythmic", "2G, rhythmic"), ("foley-2H-period", "2H, period"),
         ("foley-2H-rhythmic", "2H, rhythmic")]


def s2_fig(out):
    """Validation 2, the development figures: Leypunskiy et al. (2017), Foley et al. (2019) and Pearl and Reed
    (1920), each read as in Table 2 (with the help listed there, where it was needed), against the authors' values."""
    known = json.loads((PF / "known.json").read_text())
    fig = plt.figure(figsize=(7.5, 8.75))
    key(fig)
    top1, top2, h = 0.935, 0.705, 0.165
    ax = cell(fig, top1, h, 4, 0)
    known_points(ax, known["leypunskiy-1C"]["in vivo"])
    read_points(ax, case_part("leypunskiy-1C-nothing"))
    ax.set(xlim=(0, 24), ylim=(0, 24), xticks=[0, 12, 24], yticks=[0, 12, 24])
    finish(ax, "A", "Leypunskiy, Fig 1C", "Day length (h)", "Peak time (h after dawn)")
    for i, s in enumerate(("18:6", "12:12", "6:18")):  # B-D: the three stacked plots of Fig 2B
        ax = cell(fig, top1, h, 4, 1 + i)
        known_points(ax, known["leypunskiy-2B"][s], ms=3.4)
        read_lines(ax, case_part("leypunskiy-2B-nothing", s))
        read_points(ax, case_part("leypunskiy-2B-points", s), ms=1.8)
        ax.set(xlim=(0, 72), ylim=(0, 75), xticks=[0, 24, 48, 72], yticks=[0, 25, 50, 75])
        finish(ax, "BCD"[i], f"Fig 2B, LD {s}", "Time (h)", "%P-KaiC" if i == 0 else None)
    ax = cell(fig, top2, h, 4, 0)
    for pts in known["leypunskiy-2C"].values():
        known_points(ax, pts, ms=3.6)
    read_points(ax, case_part("leypunskiy-2C-nothing"), ms=2.0)
    ax.set(xlim=(0, 24), ylim=(0, 24), xticks=[0, 12, 24], yticks=[0, 12, 24])
    finish(ax, "E", "Fig 2C", "Day length (h)", "Peak time (h after dawn)")
    ax = cell(fig, top2, h, 4, 1)
    known_points(ax, known["leypunskiy-5A"]["12 h pulses"])
    read_points(ax, case_part("leypunskiy-5A-nothing"))
    ax.set(xlim=(34, 62), ylim=(-12, 14))
    finish(ax, "F", "Fig 5A", "Dark pulse time (h)", "Phase shift (h)")
    ax = cell(fig, top2, h, 4, 2)
    for pts in known["leypunskiy-5B"].values():
        known_points(ax, pts)
    read_points(ax, case_part("leypunskiy-5B-colour-size"))
    ax.set(xlim=(6, 18), ylim=(-4, 4))
    finish(ax, "G", "Fig 5B", "Dark pulse length (h)")
    ax = cell(fig, top2, h, 4, 3)
    for pts in known["leypunskiy-5C"].values():
        known_points(ax, pts, ms=3.6)
    read_points(ax, case_part("leypunskiy-5C-colours"), ms=2.0)
    ax.set(xlim=(6, 18), ylim=(6, 18))
    finish(ax, "H", "Fig 5C", "Day length (h)", "Peak time (h)")
    # I: Pearl and Reed (1920), Fig 3: the census and the fitted curve, on a scanned page
    top3 = 0.475
    ax = cell(fig, top3, 0.385, 4, 0)
    pr = known["pearl-reed-3-curve"]["logistic"]
    ax.plot(pr["x"], np.asarray(pr["y"]) / 1e6, color=KNOWN_C, lw=1.6, zorder=1)
    known_points(ax, [(x, y / 1e6) for x, y, _ in known["pearl-reed-3-dots"]["census"]])
    read_lines(ax, case_part("pearl-reed-3-curve-ticks-exclude-gridline"), scale=1e6)
    d = case_part("pearl-reed-3-dots-ticks-colour-area")
    read_points(ax, d.assign(y_value=d.y_value / 1e6))
    ax.set(xlim=(1780, 1920), ylim=(0, 110))
    finish(ax, "I", "Pearl and Reed, Fig 3", "Year", "Population (millions)")
    # J: Foley et al. (2019), Fig 2C-H: ten bar charts, of periods (h) and of the percentage of flies rhythmic
    ncol, left, right, hj = 5, 0.33, 0.985, 0.15
    w = (right - left - 0.04 * (ncol - 1)) / ncol
    for j, (name, title) in enumerate(FOLEY):
        row, col = divmod(j, ncol)
        ax = fig.add_axes([left + col * (w + 0.04), top3 - row * (hj + 0.07) - hj, w, hj])
        paired_bars(ax, known[name], case_part(f"{name}-nothing"))
        ax.set_ylim(*((0, 100) if "rhythmic" in name else (23, 25.5) if name < "foley-2F" else (23, 27)))
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=6, length=2)
        ax.text(0.0, 1.03, title + (" (%)" if "rhythmic" in name else " (h)"), transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha="left")
        if j == 0:
            ax.text(-0.3, 1.03, "J", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    save(fig, out, "S2_Fig")


def dot_columns(ax, columns, d):
    """A dot plot: each column's known values (blue, left) and the dots read in it (orange, right), the dots split
    into columns at the widest gaps between them, as the comparison splits them."""
    cols = list(columns.items())
    d = d.dropna(subset=["y_value"])
    if len(d):  # the dots are coloured; the means and error bars drawn over them are black
        rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in d["colour"]])
        d = d[(rgb.max(axis=1) - rgb.min(axis=1)) > 60].sort_values("x_pixel")
    xs = d["x_pixel"].to_numpy(float)
    cuts = sorted(int(g) + 1 for g in np.argsort(-np.diff(xs))[: len(cols) - 1]) if len(xs) >= len(cols) else []
    rng = np.random.default_rng(0)
    for j, ((_name, values), idx) in enumerate(zip(cols, np.split(np.arange(len(xs)), cuts))):
        if values is not None:
            v = np.asarray(values, float)
            ax.plot(j - 0.2 + (rng.random(len(v)) - 0.5) * 0.25, v, "o", ms=2.6, mfc="none", mec=KNOWN_C, mew=0.6)
        else:
            ax.text(j, 0.02, "not\ncompared", transform=ax.get_xaxis_transform(), fontsize=5.5, ha="center",
                    va="bottom", color=MUTED)
        got = d.iloc[idx]["y_value"].to_numpy(float)
        ax.plot(j + 0.2 + (rng.random(len(got)) - 0.5) * 0.25, got, "o", ms=1.8, color=READ_C, mew=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Lost\n1st", "Lost\n2nd", "Base\n1st", "Base\n2nd", "Gain\n1st", "Gain\n2nd"], fontsize=6)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.axhline(0, color=MUTED, lw=0.5, ls=(0, (2, 2)))


def s3_fig(out):
    """Validation 2, the figures held out: Andreani et al. (2022), Figs 1 and 3."""
    known = json.loads((PF / "known.json").read_text())
    fig = plt.figure(figsize=(7.5, 8.75))
    key(fig)
    top1, h = 0.935, 0.15
    ax = cell(fig, top1, h, 2, 0)
    for pts in known["andreani-1B"].values():
        known_points(ax, pts, ms=3.6)
    read_points(ax, case_part("andreani-1B-area-ticks-colours"), ms=2.0)
    ax.set(xlim=(-0.3, 11.8), ylim=(-1, 40))
    finish(ax, "A", "Andreani, Fig 1B", "Time (ZT, h)", "Sleep (min)")
    ax = cell(fig, top1, h, 2, 1)
    for curve in known["andreani-1C"].values():
        ax.plot(curve["x"], curve["y"], color=KNOWN_C, lw=1.4, zorder=1)
    read_lines(ax, case_part("andreani-1C-nothing"), lw=0.7)
    ax.set(xlim=(0, 24), ylim=(0, 50))
    finish(ax, "B", "Fig 1C", "Time (CT, h)", "Beam crosses per 30 min")
    ax = cell(fig, 0.705, 0.15, 1, 0)
    paired_bars(ax, known["andreani-1E"], case_part("andreani-1E-area"))
    ax.set_ylim(0, 200)
    finish(ax, "C", "Fig 1E, 24 bars", None, "Sleep gain (min)")
    ax.grid(axis="x", visible=False)
    for i, p in enumerate("ABCD"):  # D-G: the four dot plots of Fig 3
        row, col = divmod(i, 2)
        ax = cell(fig, 0.47 - row * 0.235, 0.17, 2, col)
        dot_columns(ax, known[f"andreani-3{p}"], case_part(f"andreani-3{p}-colours"))
        ax.set_ylim(-100, 300)
        finish(ax, "DEFG"[i], f"Fig 3{p}", None, "Sleep (min)" if col == 0 else None)
        ax.grid(axis="x", visible=False)
    save(fig, out, "S3_Fig")


# ---- S4 Fig: Validation 3, every series with data

def panel_figure(figure, panel):
    """The figure a panel is in, from a paper's list of figures ("Fig 2b, 2f") and the panel's name ("f LMA"): the
    part whose letter is the panel's, or the first."""
    parts = [s.strip() for s in str(figure).split(",")]
    letter = str(panel).strip()[:1].lower()
    for s in parts:
        if s[-1:].lower() == letter:
            return s if s.startswith("Fig") else f"Fig {s}"
    return parts[0]


def s4_fig(out):
    """Validation 3: each of the 29 series with deposited data, digitised and as the authors' data give it, at the
    authors' time points; grouped by paper, in the order of S1 Table."""
    sampled = pd.read_csv(AT / "results" / "sampled.csv")
    comp = pd.read_csv(AT / "results" / "comparisons.csv")
    comp = comp[comp["true_cos_p"].notna()]
    papers = pd.read_csv(AT / "papers.csv").set_index("pmcid")
    order = {p: i for i, p in enumerate(papers.index)}
    comp = comp.assign(_o=comp.pmcid.map(order)).sort_values(["_o", "panel", "series"])
    fig = plt.figure(figsize=(7.5, 8.75))
    key(fig, "Authors' data")
    ncol, nrow = 3, 10
    left, right, top, bottom, gx = 0.06, 0.99, 0.955, 0.03, 0.055
    w = (right - left - gx * (ncol - 1)) / ncol
    hcell = (top - bottom) / nrow
    for k, (_, c) in enumerate(comp.iterrows()):
        row, col = divmod(k, ncol)
        ax = fig.add_axes([left + col * (w + gx), top - (row + 1) * hcell + 0.03, w, hcell - 0.052])
        s = sampled[(sampled.case == c.case) & (sampled.panel == c.panel) & (sampled.series == c.series)]
        s = s.sort_values("x")
        ax.plot(s.x, s.original, color=KNOWN_C, lw=1.0)
        ax.plot(s.x, s.digitised, color=READ_C, lw=0.8)
        span = float(s.x.max() - s.x.min()) if len(s) else 24.0
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(12 if span <= 60 else 24 if span <= 130 else 48))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        style(ax)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        p = papers.loc[c.pmcid]
        ax.text(0.0, 1.04, f"{p.first_author} {p.year} ({panel_figure(p.figure, c.panel)}): {c.panel}, {c.series}",
                transform=ax.transAxes, fontsize=5.8, va="bottom", ha="left", color=INK)
        ax.text(1.0, 1.04, f"r = {c.r:.2f}", transform=ax.transAxes, fontsize=5.8, va="bottom", ha="right", color=MUTED)
        if k + ncol >= len(comp):
            ax.set_xlabel("Time (h)", fontsize=6.5, labelpad=1)
    save(fig, out, "S4_Fig")


# ---- S5 Fig and S6 Fig: dense traces

STAT = {"top": ("Upper envelope", "#eb6834"), "midpoint": ("Midpoint", INK), "mean of pixels": ("Mean", "#1baf7a"),
        "bottom": ("Lower envelope", "#6250d6")}  # orange, ink, aqua, violet: the column statistics, in every panel
STAT_SHORT = {"top": "Upper", "midpoint": "Midpoint", "mean of pixels": "Mean", "bottom": "Lower"}
READING = {"primary": "Primary", "as read": "No exclusions", "bins on the hour": "Centred bins",
           "nearest the hour": "Nearest hour", "top": "Upper envelope", "mean of pixels": "Column mean",
           "bottom": "Lower envelope"}
DT = HERE.parent / "validation" / "dense-traces"


def s5_fig(out):
    """Validation 4: real records drawn as dense traces, as Fig 4 of Lopatina et al. (2014) is drawn, and digitised."""
    fig = plt.figure(figsize=(7.5, 8.6))
    r = pd.read_csv(DT / "results" / "readings.csv")
    hh = pd.read_csv(DT / "results" / "hourly.csv")
    # A: one of the test images; B: its hourly values, true and digitised
    rec, size = "locomotor, mean of 3", "PMC-sized"
    im = np.asarray(Image.open(DT / "figures" / "locomotor_mean_of_3-PMC-sized-blue.jpg").convert("RGB"))
    meta = r[(r.record == rec) & (r["size"] == size) & (r.colour == "blue")].iloc[0]
    m, w7, h = 20, 551, 217
    ax = fig.add_axes([0.075, 0.865, 0.9, 0.105])
    days3 = int(round(m + w7 * 72 / 168))
    ax.imshow(im[m - 2:m + h + 3, m - 1:days3 + 1], extent=[0, 72, 0, meta.y_max * (h + 5) / h], aspect="auto",
              interpolation="nearest")
    ax.set_xlim(0, 72)
    ax.set_xticks(range(0, 73, 12))
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    letter(ax, "A", x=-0.01)
    one = hh[(hh.record == rec) & (hh["size"] == size) & (hh.colour == "blue") & (hh.hour < 72)]
    ax = fig.add_axes([0.075, 0.67, 0.9, 0.14])
    t = one[one.statistic == "midpoint"]
    ax.plot(t.hour + 0.5, t.true, color="#9a9a96", lw=3.0, solid_capstyle="butt", label="Record")
    for st in ("top", "midpoint", "bottom"):
        e = one[one.statistic == st]
        ax.plot(e.hour + 0.5, e.read, color=STAT[st][1], lw=0.9, label=STAT[st][0])
    ax.set_xlim(0, 72)
    ax.set_xticks(range(0, 73, 12))
    ax.set_ylim(0, 1.25 * np.nanmax(one.read))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Counts per 5 min")
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=1.0,
              bbox_to_anchor=(0.5, 1.02))
    style(ax)
    letter(ax, "B", x=-0.01)
    # C: every record, its hourly means and the midpoints read from it (the PubMed Central size, blue)
    records = list(dict.fromkeys(r.record))
    for k, name in enumerate(records):
        row, col = divmod(k, 3)
        ax = fig.add_axes([0.075 + col * 0.315, 0.535 - row * 0.12, 0.27, 0.075])
        e = hh[(hh.record == name) & (hh["size"] == size) & (hh.colour == "blue") & (hh.statistic == "midpoint")]
        ax.plot(e.hour + 0.5, e.true, color="#9a9a96", lw=1.6, solid_capstyle="butt")
        ax.plot(e.hour + 0.5, e.read, color=INK, lw=0.6)
        ax.set_xlim(0, float(e.hour.max()) + 1 if len(e) else 168)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(24))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        style(ax)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        q = r[(r.record == name) & (r["size"] == size) & (r.colour == "blue") & (r.statistic == "midpoint")].iloc[0]
        ax.text(0.0, 1.05, name[0].upper() + name[1:], transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left")
        ax.text(1.0, 1.05, f"r = {q.r:.2f}", transform=ax.transAxes, fontsize=6, va="bottom", ha="right", color=MUTED)
        if k == 0:
            letter(ax, "C", x=-0.1, y=1.05)
        if row == 2:
            ax.set_xlabel("Time (h)", fontsize=6.5, labelpad=1)
    # D: the test summarised over its 36 readings
    order = ["top", "midpoint", "mean of pixels", "bottom"]
    specs = [("level", "Level (digitised/true)", True), ("dark_share_error", "Dark-phase share error", False),
             ("acrophase_error_h", "Absolute acrophase error (h)", False)]
    for k, (col, lab, logy) in enumerate(specs):
        ax = fig.add_axes([0.1 + k * 0.32, 0.055, 0.22, 0.15])
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
        ax.set_xticklabels([STAT_SHORT[st] for st in order], fontsize=6.5)
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylabel(lab, fontsize=7)
        style(ax)
        ax.grid(axis="x", visible=False)
        if k == 0:
            letter(ax, "D", x=-0.2)
    save(fig, out, "S5_Fig")


def s6_fig(out):
    """Digitising the CD157 figure: what each pixel column of it holds, and the result in every analysis."""
    fig = plt.figure(figsize=(7.5, 5.2))
    case = cd157.COPIES[next(iter(cd157.COPIES))]["left out"]
    cols = cd157.columns(case)
    cfg = json.loads((AT / "configs" / f"{case}.json").read_text())
    # A, B: a stretch of each panel with the values of every column
    for i, ((panel, name), (a, b), ymax) in enumerate(zip(cd157.GENOTYPES, [(26, 38), (24, 36)], [250, 150])):
        ax = fig.add_axes([0.075 + i * 0.475, 0.62, 0.42, 0.31])
        rgb, ext, p = panel_crop(panel, a, b)
        faded = (255 - 0.55 * (255 - rgb.astype(float))).astype(np.uint8)
        ax.imshow(faded, extent=ext, aspect="auto", interpolation="nearest", zorder=0)
        if i == 1:  # the gridline rows excluded from panel B, in value units
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
        ax.set_xlabel("Time from first lights-off (h)")
        if i == 0:
            ax.set_ylabel("Activity (cm/s)")
        style(ax, grid=False)
        letter(ax, "AB"[i])
        tag(ax, GENO_LABEL[name])
        if i == 0:
            ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5,
                      handletextpad=0.2, columnspacing=0.8, borderpad=0.3)
    # C: the CD157 result in every analysis
    sm = pd.read_csv(AT / "results" / "cd157_summary.csv")
    readings = list(dict.fromkeys(sm.reading))
    rcol = {"top": STAT["top"][1], "mean of pixels": STAT["mean of pixels"][1], "bottom": STAT["bottom"][1]}
    for k, (fn, lab, ylim) in enumerate([
            (lambda q: q.acrophase_ko_minus_wt_h, "Acrophase difference (h)", (-8, 0)),
            (lambda q: q.dark_share_ko - q.dark_share_wt, "Dark-phase share difference", (0, 0.25))]):
        ax = fig.add_axes([0.1 + k * 0.475, 0.2, 0.38, 0.27])
        for j, rd in enumerate(readings):
            for cp, mk, dx in zip(cd157.COPIES, ("o", "s"), (-0.15, 0.15)):
                q = sm[(sm.reading == rd) & (sm["copy"] == cp)].iloc[0]
                c = rcol.get(rd, INK)
                ax.plot(j + dx, fn(q), mk, ms=3.8, color=c, mfc=c if mk == "o" else "white", mec=c, mew=1.0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(readings)))
        ax.set_xticklabels([READING[x] for x in readings], rotation=45, ha="right", fontsize=6.5)
        ax.set_xlim(-0.6, len(readings) - 0.4)
        ax.set_ylabel(lab, fontsize=7)
        style(ax)
        ax.grid(axis="x", visible=False)
        if k == 0:
            letter(ax, "C", x=-0.12)
            ax.text(0.03, 0.04, "● PMC  □ journal", transform=ax.transAxes, fontsize=6, color=MUTED)
    save(fig, out, "S6_Fig")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="the figures of the paper")
    ap.add_argument("out", nargs="?", default=str(HERE / "out"))
    ap.add_argument("--workbench", help="a checkout of digitiser-workbench, for S1 Fig (the figures of Validation 1)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(out)
    fig3(out)
    if args.workbench:
        s1_fig(out, args.workbench)
    else:
        print("S1 Fig left out: give --workbench PATH (a checkout of digitiser-workbench)")
    s2_fig(out)
    s3_fig(out)
    s4_fig(out)
    s5_fig(out)
    s6_fig(out)
