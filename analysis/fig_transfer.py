from figs_common import *
from scipy.stats import spearmanr

ms = rd("portability_v4_main_evaluation/tables/method_summary.csv")
pair = rd("portability_v4_main_evaluation/tables/pair_level_results.csv")
pair = pair[pair.analysis == "main"]

# ---------------------------------------------------------------- Figure 2: transfer ladder and spread dependence
LAD = [("zero_shot", "Zero shot"), ("moment", "Moment\nmatching"), ("sign_moment", "Sign\ncorrected"), ("diag_paired", "Per-feature\npaired"),
       ("paired_linear", "Paired\nlinear"), ("paired_rff", "Paired\nnonlinear"), ("distill_rff", "Score\ndistillation"),
       ("pseudo_label", "Teacher\nlabels"), ("twin_paired_rff", "Model\ntwin")]
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.5), gridspec_kw={"width_ratios": [1.45, 1.0]})
ax = axes[0]
s = ms[ms.level == "cross"].set_index("method")
x = np.arange(len(LAD))
for i, (m, _) in enumerate(LAD):
    ci(ax, [i], [s.loc[m, "median_gap"]], [s.loc[m, "ci95_low"]], [s.loc[m, "ci95_high"]], C[m], "o")
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([l.replace("\n", " ") for _, l in LAD], fontsize=9, rotation=40, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, len(LAD) - 0.4)
set_bounds(ax, "y", min(0.0, float(s.loc[[m for m, _ in LAD], "ci95_low"].min())), float(s.loc[[m for m, _ in LAD], "ci95_high"].max()), step=0.1)
ax.set_ylabel("Gap to full refit (nBA)")
style(ax)
ax = axes[1]
levels = ["d2d_0.05", "d2d_0.20", "d2d_0.50", "cross"]
lab = ["5%", "20%", "50%", "Other\nfamily"]
allv = [0.0]
for (m, name, mk), off in zip([("zero_shot", "Zero shot", "s"), ("moment", "Moment matching", "^"), ("paired_rff", "Paired nonlinear", "o"),
                                ("pseudo_label", "Teacher labels", "D")], [-0.21, -0.07, 0.07, 0.21]):
    g = ms[ms.method == m].set_index("level").reindex(levels)
    ci(ax, np.arange(4) + off, g.median_gap, g.ci95_low, g.ci95_high, C[m], mk, name, ms=4.5)
    allv += list(g.ci95_high) + list(g.ci95_low)
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=9); ax.set_xlim(-0.5, 3.5)
ax.set_xlabel("Target device")
set_bounds(ax, "y", min(allv), max(allv) + 0.2, step=0.1)
ax.set_ylabel("Gap to full refit (nBA)")
style(ax); legend(ax, loc="upper left", fontsize=9, handletextpad=0.3, borderpad=0.4)
fig.tight_layout(w_pad=1.6); letters(fig, axes); save(fig, "fig2_transfer")

# ---------------------------------------------------------------- Figure 3: why readouts fail across families
mo = pair[pair.method == "moment"]
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.1), gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})
ax = axes[0]
d = mo.groupby("level")[["feature_corr_mean", "feature_sign_flip_fraction", "readout_cosine"]].median().reindex(levels)
xx = np.arange(4)
w = 0.26
for j, (col, name, cc) in enumerate([("feature_corr_mean", "Feature correlation", "#1b6ca8"), ("feature_sign_flip_fraction", "Sign-flip fraction", "#c0392b"),
                                     ("readout_cosine", "Readout cosine", "#7b4fa0")]):
    ax.bar(xx + (j - 1) * w, d[col], w, color=cc, edgecolor="black", lw=0.5, label=name)
ax.set_xticks(xx); ax.set_xticklabels(["5%", "20%", "50%", "Other\nfamily"], fontsize=9); ax.set_xlim(-0.5, 3.5)
ax.set_xlabel("Target device")
set_bounds(ax, "y", 0.0, 1.2, step=0.2)
ax.set_ylabel("Median value")
style(ax); legend(ax, loc="upper right", fontsize=9, handlelength=1.0, borderpad=0.3, labelspacing=0.25)
ax = axes[1]
cols = {"d2d_0.05": "#9ecae1", "d2d_0.20": "#4292c6", "d2d_0.50": "#08519c", "cross": "#7f7f7f"}
names = {"d2d_0.05": "5%", "d2d_0.20": "20%", "d2d_0.50": "50%", "cross": "Other family"}
for lv in ["cross", "d2d_0.50", "d2d_0.20", "d2d_0.05"]:
    g = mo[mo.level == lv]
    ax.scatter(g.feature_corr_mean, g.gap, s=4, alpha=0.45, color=cols[lv], label=names[lv], lw=0)
set_bounds(ax, "x", float(mo.feature_corr_mean.min()), 1.0, step=0.4)
set_bounds(ax, "y", float(mo.gap.min()), float(mo.gap.max()) + 0.4, step=0.4)
ax.set_xlabel("Mean feature correlation")
ax.set_ylabel("Moment gap (nBA)")
style(ax); legend(ax, loc="upper right", fontsize=9, markerscale=2.2, handletextpad=0.2, borderpad=0.3, labelspacing=0.25)
ax = axes[2]
a = pair[pair.method == "scratch_k5"][["dataset", "source", "target", "level", "gap"]]
b = pair[pair.method == "moment_warm_k5"][["dataset", "source", "target", "level", "gap", "readout_cosine"]]
wv = a.merge(b, on=["dataset", "source", "target", "level"], suffixes=("_s", "_w"))
wv["adv"] = wv.gap_s - wv.gap_w
for lv in ["cross", "d2d_0.50", "d2d_0.20", "d2d_0.05"]:
    g = wv[wv.level == lv]
    ax.scatter(g.readout_cosine, g.adv, s=4, alpha=0.45, color=cols[lv], lw=0)
ax.axhline(0, color="0.5", ls=":", lw=0.8)
set_bounds(ax, "x", float(wv.readout_cosine.min()), 1.0, step=0.4)
set_bounds(ax, "y", float(wv.adv.min()), float(wv.adv.max()), step=0.2)
ax.set_xlabel("Readout cosine")
ax.set_ylabel("Warm-start gain (nBA)")
style(ax)
fig.tight_layout(w_pad=1.0); letters(fig, axes); save(fig, "fig3_mechanism")
print(d.round(3))
print("warm adv medians", wv.groupby(wv.level == "cross").adv.median())
