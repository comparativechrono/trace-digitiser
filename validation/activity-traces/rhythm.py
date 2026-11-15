"""The rhythm analysis applied to every series, digitised or original.

It follows the submission's own analysis (S1 Code): a periodogram peak in the circadian band (18-30 h), a
single-component cosinor at that period, dark-phase and light-phase means, and M10/L5/RA; and adds the tests that
decide whether a series is rhythmic at all: the chi-square periodogram (Sokolove and Bushell 1978), the Lomb-Scargle
periodogram (with its false-alarm probability), and the cosinor's zero-amplitude F test.

A series is (t, y): times in hours from the start of the plotted record (regularly spaced), values in any unit.
"""

import math

import numpy as np
from scipy import signal, stats

BAND = (18.0, 30.0)
ALPHA = 0.05


def regular(t, y):
    """Drop missing values; check the spacing; the bin width in hours."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t)
    t, y = t[order], y[order]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    return t, y, dt


def fft_period(t, y, dt):
    """As the submission: scipy's periodogram of the linearly detrended series, its peak within 18-30 h. (A record
    with missing values is filled in linearly onto its regular grid for this estimate only.)"""
    grid = t[0] + dt * np.arange(int(np.rint((t[-1] - t[0]) / dt)) + 1)
    if len(grid) != len(t):
        y = np.interp(grid, t, y)
    y0 = signal.detrend(y, type="linear")
    f, p = signal.periodogram(y0, fs=1.0 / dt, scaling="spectrum")
    band = (f > 0) & (f >= 1 / BAND[1]) & (f <= 1 / BAND[0])
    if not band.any():
        return float("nan"), float("nan")
    i = np.argmax(np.where(band, p, -np.inf))
    nondc = p[1:].sum()
    return float(1.0 / f[i]), float(p[i] / nondc) if nondc > 0 else float("nan")


def lomb_scargle(t, y):
    """Lomb-Scargle periodogram over 18-30 h (normalised by the variance); the peak, its power and its false-alarm
    probability (the usual approximation, with as many independent frequencies as the band holds in the record)."""
    yc = y - y.mean()
    var = yc.var()
    if var <= 0 or len(y) < 6:
        return float("nan"), float("nan"), 1.0
    periods = np.linspace(BAND[0], BAND[1], 481)
    w = 2 * np.pi / periods
    p = signal.lombscargle(t, yc, w) / var  # Scargle normalisation: z = P / sigma^2 (power in units of the variance)
    i = int(np.argmax(p))
    z = float(p[i])
    span = t[-1] - t[0]
    m = max(1.0, span * (1 / BAND[0] - 1 / BAND[1]))
    fap = 1.0 - (1.0 - math.exp(-z)) ** m
    return float(periods[i]), z, float(min(1.0, max(0.0, fap)))


def chi_square(t, y, dt):
    """Sokolove-Bushell chi-square periodogram over the whole-bin periods in 18-30 h: the period with the largest Qp
    above the critical value (alpha 0.05), and that excess; rhythmic if any Qp exceeds its critical value. Values
    are folded by their time, over the whole cycles of the record, so a missing value leaves a gap rather than
    shifting the rest; with none missing this is Qp = K N sum (M_h - M)^2 / sum (X_i - M)^2."""
    i = np.rint((t - t[0]) / dt).astype(int)
    span = int(i[-1]) + 1
    best = (float("nan"), -np.inf, float("nan"))
    for pb in range(max(2, int(math.ceil(BAND[0] / dt))), int(math.floor(BAND[1] / dt)) + 1):
        k = span // pb
        if k < 2:
            continue
        use = i < k * pb
        x, ph = y[use], i[use] % pb
        if len(x) < pb + 2:
            continue
        m = x.mean()
        denom = ((x - m) ** 2).sum()
        if denom <= 0:
            continue
        num = 0.0
        for h in range(pb):
            sel = ph == h
            if sel.any():
                num += sel.sum() * (x[sel].mean() - m) ** 2
        qp = len(x) * num / denom
        crit = stats.chi2.ppf(1 - ALPHA, pb - 1)
        if qp - crit > best[1]:
            best = (pb * dt, qp - crit, qp)
    return best[0], float(best[1]), best[2]


def cosinor(t, y, period):
    """Single-component cosinor at the given period: mesor, amplitude, acrophase (h from t = 0, mod period), R2, and
    the F test of zero amplitude."""
    w = 2 * np.pi / period
    x = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    beta, res, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = x @ beta
    ss_res = float(((y - fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(y)
    f = ((ss_tot - ss_res) / 2) / (ss_res / (n - 3)) if n > 3 and ss_res > 0 else float("inf")
    p = float(stats.f.sf(f, 2, n - 3)) if n > 3 else float("nan")
    amp = float(math.hypot(beta[1], beta[2]))
    phase = float((math.atan2(beta[2], beta[1]) / w) % period)
    return {"mesor": float(beta[0]), "amplitude": amp, "acrophase": phase, "r2": r2, "p": p}


def dark_light(t, y, dark_start, dark_hours=12.0, period=24.0):
    """Means in the dark and the light phases; dark_start is when lights go off, in hours from t = 0, or a list of
    (start, end) dark intervals in hours for an irregular schedule."""
    if isinstance(dark_start, (list, tuple)):
        dark = np.zeros(len(t), bool)
        for a, b in dark_start:
            dark |= (t >= a) & (t < b)
        d, light = y[dark], y[~dark]
        md = float(d.mean()) if len(d) else float("nan")
        ml = float(light.mean()) if len(light) else float("nan")
        share = float(d.sum() / y.sum()) if y.sum() > 0 else float("nan")
        return {"dark_mean": md, "light_mean": ml, "dark_fraction": share}
    ph = (t - dark_start) % period
    dark = ph < dark_hours
    d, light = y[dark], y[~dark]
    md = float(d.mean()) if len(d) else float("nan")
    ml = float(light.mean()) if len(light) else float("nan")
    share = md * dark_hours / (md * dark_hours + ml * (period - dark_hours)) \
        if np.isfinite(md + ml) and (md + ml) > 0 else float("nan")
    return {"dark_mean": md, "light_mean": ml, "dark_fraction": share}


def m10_l5(t, y, dt):
    """M10, L5 and the relative amplitude from the average 24-h profile (whole days; each bin of the profile is the
    mean of the values present at that time of day)."""
    nb = int(round(24 / dt))
    i = np.rint((t - t[0]) / dt).astype(int)
    k = (int(i[-1]) + 1) // nb if len(i) else 0
    if nb < 4 or k < 1:
        return {"M10": float("nan"), "L5": float("nan"), "RA": float("nan")}
    use = i < k * nb
    prof = np.full(nb, np.nan)
    for h in range(nb):
        sel = use & (i % nb == h)
        if sel.any():
            prof[h] = y[sel].mean()
    if np.isnan(prof).any():
        ok = ~np.isnan(prof)
        idx = np.arange(nb)
        prof = np.interp(idx, idx[ok], prof[ok], period=nb)
    ext = np.concatenate([prof, prof])

    def roll(hours):
        w = max(1, int(round(hours / dt)))
        return np.array([ext[j:j + w].mean() for j in range(nb)])

    m10, l5 = float(roll(10).max()), float(roll(5).min())
    return {"M10": m10, "L5": l5, "RA": (m10 - l5) / (m10 + l5) if (m10 + l5) > 0 else float("nan")}


def daily(t, y, dark_start=None, period=24.0, start=None):
    """One cosinor at 24 h per whole day of the record (consecutive 24-h windows from start, by default its first
    value): for each, the window's start, the zero-amplitude p, amplitude, acrophase and dark-phase share. Give the
    same start to series of the same record so that their days are the same windows."""
    t, y, dt = regular(t, y)
    out = []
    if len(t) < 4:
        return out
    start = t[0] if start is None else float(start)
    k = 0
    while start + (k + 1) * period <= t[-1] + dt + 1e-9:
        w = (t >= start + k * period - 1e-9) & (t < start + (k + 1) * period - 1e-9)
        expected = int(round(period / dt))
        if w.sum() >= max(6, 0.75 * expected):
            c = cosinor(t[w], y[w], period)
            row = {"day": k + 1, "start_h": float(start + k * period), "n": int(w.sum()), "cos_p": c["p"],
                   "cos_amplitude": c["amplitude"], "cos_acrophase": c["acrophase"], "cos_mesor": c["mesor"],
                   "rhythmic_cos": bool(c["p"] < ALPHA)}
            if dark_start is not None:
                row.update(dark_light(t[w], y[w], dark_start))
            out.append(row)
        k += 1
    return out


def analyse(t, y, dark_start=None, fixed_period=None):
    """Everything, for one series. fixed_period: fit the cosinor at this period (24 h under LD); otherwise at the
    Lomb-Scargle peak."""
    t, y, dt = regular(t, y)
    out = {"n": int(len(y)), "bin_h": dt, "span_h": float(t[-1] - t[0] + dt) if len(t) else 0.0}
    if len(t) > 1 and np.isfinite(dt) and dt > 0:
        out["missing"] = int(np.rint((t[-1] - t[0]) / dt)) + 1 - len(t)
    if len(y) < 8 or not np.isfinite(dt) or dt <= 0:
        out["error"] = "too short"
        return out
    out["fft_period"], out["fft_power_share"] = fft_period(t, y, dt)
    out["ls_period"], out["ls_power"], out["ls_fap"] = lomb_scargle(t, y)
    out["chi_period"], out["chi_excess"], out["chi_qp"] = chi_square(t, y, dt)
    per = fixed_period or out["ls_period"]
    c = cosinor(t, y, per)
    out.update({f"cos_{k}": v for k, v in c.items()})
    out["cos_period"] = per
    # The chi-square periodogram folds the record at each trial period, so it needs at least two whole cycles of
    # the longest (30 h): records shorter than 60 h get no chi-square call.
    out["chi_applicable"] = bool(out["span_h"] >= 2 * BAND[1])
    out["rhythmic_chi"] = bool(out["chi_excess"] > 0) if out["chi_applicable"] else None
    out["rhythmic_ls"] = bool(out["ls_fap"] < ALPHA)
    out["rhythmic_cos"] = bool(c["p"] < ALPHA)
    if dark_start is not None:
        out.update(dark_light(t, y, dark_start))
    out.update(m10_l5(t, y, dt))
    out["mean"] = float(y.mean())
    return out
