"""Shared helpers for the manuscript figures (print size, large fonts, bounds at tick multiples)."""
import math
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

S = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))          # repository root
B = S / "results"
OUT = S / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SPINE_W = 1.0
LEGEND_W = 0.6
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10.5, "axes.labelsize": 11.5, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": SPINE_W, "xtick.major.width": SPINE_W, "ytick.major.width": SPINE_W,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "lines.linewidth": 1.4,
    "savefig.dpi": 600, "savefig.bbox": "tight", "figure.dpi": 150, "pdf.fonttype": 42,
})
W_FULL = 7.2      # inches, full page width
W_HALF = 3.5

C = {"zero_shot": "#7f7f7f", "moment": "#e08a1e", "sign_moment": "#f2b65f", "diag_paired": "#c8a2c8", "coral": "#b5651d",
     "paired_linear": "#6baed6", "paired_rff": "#1b6ca8", "distill_rff": "#08306b", "pseudo_label": "#7b4fa0",
     "twin_paired_rff": "#2a9d5c", "full_refit": "#000000", "scratch": "#c0392b", "warm": "#e377c2"}
FAM_COLORS = {"LD": "#1b6ca8", "VT": "#e08a1e", "VV": "#2a9d5c", "ZH": "#c0392b", "DF": "#7b4fa0", "DN": "#8c6d3f", "TX": "#17becf"}
FAMS = ["LD", "VT", "VV", "ZH", "DF", "DN"]


def nice_step(span, n=5):
    span = max(span, 1e-12)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= raw - 1e-12:
            return mult * mag
    return 10 * mag


def decimals(step):
    d = 0
    while d < 6 and abs(round(step * 10 ** d) - step * 10 ** d) > 1e-6:
        d += 1
    return d


def set_bounds(ax, axis, lo, hi, step=None, n=5):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        hi = lo + 1.0
    step = step or nice_step(hi - lo, n)
    a = math.floor(lo / step + 1e-9) * step
    b = math.ceil(hi / step - 1e-9) * step
    if b <= a:
        b = a + step
    ticks = np.arange(a, b + step / 2, step)
    d = decimals(step)
    fmt = FuncFormatter(lambda v, _: f"{v:.{d}f}" if abs(v) > 10 ** (-d - 1) else f"{0:.{d}f}")
    if axis == "y":
        ax.set_ylim(a, b); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(fmt)
    else:
        ax.set_xlim(a, b); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(fmt)
    return a, b, step


def style(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(SPINE_W)
    ax.tick_params(direction="out")


def legend(ax, **kw):
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, **kw)
    leg.get_frame().set_linewidth(LEGEND_W)
    return leg


def letters(fig, axes, dy=0.012, size=13):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for i, ax in enumerate(axes):
        bb = ax.yaxis.label.get_window_extent(rend)
        x0 = inv.transform((bb.x0, 0))[0]
        top = inv.transform((0, ax.get_window_extent(rend).y1))[1]
        fig.text(x0, top + dy, chr(ord("a") + i), fontsize=size, fontweight="bold", ha="left", va="bottom")


def ci(ax, x, med, lo, hi, color, marker="o", label=None, ms=5.5, lw=1.3, capsize=2.5):
    med, lo, hi = map(lambda v: np.asarray(v, float), (med, lo, hi))
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt=marker, color=color, ms=ms, capsize=capsize, lw=lw, label=label,
                markeredgecolor="black", markeredgewidth=0.4)


def colorbar(fig, ax, im, label, lo, hi, step):
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    tk = np.arange(lo, hi + step / 2, step)
    d = decimals(step)
    cb.set_ticks(tk); cb.ax.set_yticklabels([f"{v:.{d}f}" for v in tk])
    cb.set_label(label); cb.outline.set_linewidth(SPINE_W)
    cb.ax.tick_params(width=SPINE_W)
    return cb


AUDIT = S / "figures" / "figure_audit.txt"


def audit(fig, name):
    """Check every axis: bounds equal the outer ticks, all plotted data and error bars inside the bounds,
    no titles, all spines visible, and report the smallest font size used anywhere in the figure."""
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.lines import Line2D
    issues, sizes = [], []
    fig.canvas.draw()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible():
            sizes.append((t.get_fontsize(), t.get_text()[:25]))
    for i, ax in enumerate(fig.axes):
        if not ax.axison or ax.get_label() == "<colorbar>":
            continue
        if ax.get_title():
            issues.append(f"ax{i} has title")
        for k, sp in ax.spines.items():
            if not sp.get_visible():
                issues.append(f"ax{i} spine {k} hidden")
        for axis, lim in (("x", ax.get_xlim()), ("y", ax.get_ylim())):
            A = ax.xaxis if axis == "x" else ax.yaxis
            ticks = [v for v in A.get_ticklocs()]
            labs = [l.get_text() for l in A.get_ticklabels()]
            if not ticks:
                continue
            cat = any(l and not l.replace("-", "").replace(".", "").replace("\u2212", "").strip().isdigit() for l in labs)
            if cat or ax.images:
                continue
            lo, hi = min(lim), max(lim)
            if abs(min(ticks) - lo) > 1e-9 * max(1, abs(lo)) or abs(max(ticks) - hi) > 1e-9 * max(1, abs(hi)):
                issues.append(f"ax{i} {axis} bounds {lo:g},{hi:g} not at ticks {min(ticks):g},{max(ticks):g}")
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        pts = []
        for c in ax.collections:
            if isinstance(c, LineCollection):
                for seg in c.get_segments():
                    pts += list(seg)
            elif isinstance(c, PathCollection):
                pts += list(c.get_offsets())
        for l in ax.lines:
            xd, yd = np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)
            if len(xd) == len(yd) and len(xd) > 0 and not (len(set(np.round(yd, 12))) == 1 and len(xd) == 2 and l.get_linestyle() in (":", "--")):
                pts += list(zip(xd, yd))
        P = np.array([p for p in pts if np.all(np.isfinite(p))]) if pts else np.zeros((0, 2))
        if len(P):
            out = (P[:, 0] < x0 - 1e-9) | (P[:, 0] > x1 + 1e-9) | (P[:, 1] < y0 - 1e-9) | (P[:, 1] > y1 + 1e-9)
            if out.any():
                issues.append(f"ax{i} {int(out.sum())} data points outside bounds")
    mn = min(sizes) if sizes else (None, "")
    with open(AUDIT, "a") as fh:
        fh.write(f"{name}: smallest font {mn[0]:.1f} pt ('{mn[1]}'); " + ("OK" if not issues else "; ".join(issues)) + "\n")


def save(fig, name):
    audit(fig, name)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def load_engine():
    """Import the v9 script (contains every device model and the TiOx calibration code) as a library."""
    os.environ["MEMRISTIVE_PROJECT_ROOT"] = str(S / "results" / "_engine_scratch")
    spec = importlib.util.spec_from_file_location("engine", str(S / "scripts" / "memristive_portability_v9_generality.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["engine"] = m
    spec.loader.exec_module(m)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 10.5, "axes.labelsize": 11.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 9, "axes.linewidth": SPINE_W, "xtick.major.width": SPINE_W,
                         "ytick.major.width": SPINE_W, "savefig.dpi": 600})
    return m


def rd(path):
    return pd.read_csv(B / path)
