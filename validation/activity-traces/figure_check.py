"""How closely each trace as read follows the line drawn in its figure, judged from the figure alone.

    python validation/activity-traces/figure_check.py CASE [CSV] [--out DIR] [--zoom Z] [--tol T]

For each series of each panel of a case (cases.json), the reading (digitised/CASE.csv, or CSV) is put back into
pixels with the panel's calibration (digitised/CASE.calibration.json, written by digitise_case.py) and set against the
pixels of the series' colour in that column of the figure: those within T (CIE76 delta E, default 30) of the colour
clicked, or of any colour its runs clicked. Nothing here uses the authors' data. For each series it gives

    columns  pixel columns read, from the first seen to the last
    seen     the share of them in which the reader saw the trace (the rest it filled in)
    on       the share of the seen columns read within 2 px of a pixel of the colour (on the line, a marker or an
             error bar of the series)
    off_px   the median distance, in px, from the reading to the nearest pixel of the colour, over the columns read
             more than 2 px from one
    empty    the share of the seen columns with no pixel of the colour at all

    hit      the share of the values the analysis takes from the reading (as compare.py takes them: at the
             authors' times, where the figure draws them, or at regular steps) that lie within 2 px of a pixel of the
             colour

and, with --out, draws each panel's readings over the figure (enlarged Z times) for looking at: each series in a
colour of its own, solid where seen and dotted where filled in, with a cross at each value the analysis takes.
Writes DIR/CASE__PANEL.png and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from trace_digitiser.trace_detection import parse_colour  # noqa: E402

PENS = [(255, 0, 255), (0, 190, 0), (0, 150, 255), (255, 140, 0), (0, 0, 0), (200, 0, 0)]


def lab(rgb):
    """CIE Lab of sRGB values (..., 3) in 0-255."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def to_rows(y, cal):
    """Values back to pixel rows: value = a * row + b (log10 of the value, for a log axis), times y_scale."""
    y = np.asarray(y, float) / cal.get("y_scale", 1.0)
    if cal.get("log"):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log10(y)
    return (y - cal["b"]) / cal["a"]


def colours_of(panel, name):
    out = [panel["series"][name]]
    for seg in panel.get("traces", {}).get(name, []):
        out += [seg[k] for k in ("colour",) if k in seg]
    return [tuple(parse_colour(c)) for c in dict.fromkeys(out)]


def pens(panel):
    """A pen per series, far from every series' colour."""
    cols = [np.array(parse_colour(c), float) for c in panel["series"].values()]
    order = sorted(PENS, key=lambda p: -min(np.abs(np.array(p) - c).sum() for c in cols))
    return {name: order[i % len(order)] for i, name in enumerate(panel["series"])}


def samples(case, panel, series, g, truth):
    """Where compare.py reads a series (the authors' times, moved to where the figure draws them, or regular steps
    across the trace) and what it reads there: (x, value)."""
    from compare import drawn_at, grid, sample

    tm = {**case["time"], **case.get("panel_time", {}).get(panel, {})}
    g = g.dropna(subset=["x", "y"])
    if truth is not None:
        xs = np.sort(truth[(truth.panel == panel) & (truth.series == series)].x.to_numpy(float))
        xs = drawn_at(case.get("drawn", {}).get(panel), xs)
    else:
        span = case.get("x_range", {}).get(f"{panel}|{series}") or case.get("x_range", {}).get(panel)
        xs = grid(g, tm, span) if len(g) else np.array([])
    vals = np.full(len(xs), np.nan)
    ok = np.isfinite(xs)
    if len(g) > 1 and ok.any():
        vals[ok] = sample(g, xs[ok], 0.2 * tm["bin_h"])
    return xs, vals


def check(case_id, csv=None, out=None, zoom=None, tol=30.0):
    case = next(c for c in json.loads((HERE / "cases.json").read_text()) if c["id"] == case_id)
    config = json.loads((HERE / case["config"]).read_text())
    image = np.asarray(Image.open((HERE / case["config"]).parent / config["image"]).convert("RGB"))
    csv = Path(csv) if csv else HERE / case["digitised"]
    data = pd.read_csv(csv)
    cals = json.loads(csv.with_suffix(".calibration.json").read_text())
    truth = pd.read_csv(HERE / case["truth"]) if case.get("truth") else None
    rows = []
    for panel in config["panels"]:
        left, top, right, bottom = (int(round(v)) for v in panel["plot_area"])
        cal = cals.get(panel["name"])
        crop = image[top:bottom + 1, left:right + 1]
        crop_lab = lab(crop)
        pen = pens(panel)
        z = zoom or max(1, int(np.ceil(900 / max(1, right - left))))
        picture = Image.fromarray(image[max(0, top - 8):bottom + 9, max(0, left - 8):right + 9]).resize(
            ((right - left + 17) * z, (bottom - top + 17) * z), Image.NEAREST)
        draw = ImageDraw.Draw(picture)
        ox, oy = max(0, left - 8), max(0, top - 8)
        for name in panel["series"]:
            g = data[(data.panel == panel["name"]) & (data.series == name)].sort_values("x_pixel")
            if not len(g) or not cal or "a" not in cal:
                rows.append({"case": case_id, "panel": panel["name"], "series": name, "columns": 0})
                continue
            xs = g.x_pixel.to_numpy(int)
            ys = to_rows(g.y.to_numpy(float), cal)
            seen = ~g.interpolated.to_numpy(bool)
            first, last = (np.nonzero(seen)[0][[0, -1]] if seen.any() else (0, len(g) - 1))
            span = slice(first, last + 1)
            mask = np.zeros(crop.shape[:2], bool)
            for c in colours_of(panel, name):
                mask |= np.sqrt(((crop_lab - lab(np.array(c, float))) ** 2).sum(-1)) < tol
            dist = np.full(len(g), np.nan)
            for i in np.nonzero(seen)[0]:
                cx = xs[i] - left
                if 0 <= cx < mask.shape[1] and np.isfinite(ys[i]):
                    hit = np.nonzero(mask[:, cx])[0]
                    if len(hit):
                        dist[i] = float(np.min(np.abs(hit + top - ys[i])))
            s = seen[span]
            d = dist[span][s]
            far = d[np.isfinite(d) & (d > 2)]
            rows.append({"case": case_id, "panel": panel["name"], "series": name, "columns": int(last - first + 1),
                         "seen": round(float(s.mean()), 3), "on": round(float(np.mean(np.isfinite(d) & (d <= 2))), 3),
                         "off_px": round(float(np.median(far)), 1) if len(far) else 0.0,
                         "empty": round(float(np.mean(~np.isfinite(d))), 3)})
            # the values the analysis takes, as compare.py takes them: x by the reading's own columns, y by the
            # calibration
            at, vals = samples(case, panel["name"], name, g, truth)
            okx = np.isfinite(g.x.to_numpy(float))
            order = np.argsort(g.x.to_numpy(float)[okx])
            keep = np.isfinite(vals)
            sx = np.interp(at[keep], g.x.to_numpy(float)[okx][order], xs[okx][order].astype(float))
            sy = to_rows(vals[keep], cal)
            hits = []
            for cx, cy in zip(sx, sy):
                c0 = int(round(cx)) - left
                cols = [c for c in (c0 - 1, c0, c0 + 1) if 0 <= c < mask.shape[1]]
                hit = np.concatenate([np.nonzero(mask[:, c])[0] for c in cols]) + top if cols else np.array([])
                hits.append(bool(len(hit)) and float(np.min(np.abs(hit - cy))) <= 2)
            rows[-1]["hit"] = round(float(np.mean(hits)), 3) if hits else None
            rows[-1]["samples"] = len(hits)
            # the reading over the figure
            px = (xs - ox + 0.5) * z
            py = (ys - oy + 0.5) * z
            for i in range(len(g) - 1):
                if not (np.isfinite(py[i]) and np.isfinite(py[i + 1])) or xs[i + 1] - xs[i] > 2:
                    continue
                if seen[i] and seen[i + 1]:
                    draw.line([(px[i], py[i]), (px[i + 1], py[i + 1])], fill=pen[name], width=1)
                elif i % 3 == 0:
                    draw.point((px[i], py[i]), fill=pen[name])
            r = 3 * z
            for cx, cy in zip((sx - ox + 0.5) * z, (sy - oy + 0.5) * z):
                if np.isfinite(cy):
                    draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=pen[name], width=max(1, z // 2))
                    draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=pen[name], width=max(1, z // 2))
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            picture.save(Path(out) / f"{case_id}__{panel['name'].replace(' ', '_').replace('/', '-')}.png")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="+", help="case ids, or 'all'")
    ap.add_argument("--csv", help="a reading other than digitised/CASE.csv (one case only)")
    ap.add_argument("--out", help="folder for the pictures")
    ap.add_argument("--zoom", type=int)
    ap.add_argument("--tol", type=float, default=30.0)
    a = ap.parse_args()
    ids = [c["id"] for c in json.loads((HERE / "cases.json").read_text())] if a.case == ["all"] else a.case
    t = pd.concat([check(i, a.csv, a.out, a.zoom, a.tol) for i in ids], ignore_index=True)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
