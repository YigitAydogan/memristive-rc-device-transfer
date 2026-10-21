from figs_common import *

ms = rd("portability_v4_main_evaluation/tables/method_summary.csv")
mom_cross = float(ms[(ms.level == "cross") & (ms.method == "moment")].median_gap.iloc[0])

# ---------------------------------------------------------------- Figure 4: what label-free recovery requires
chain = rd("portability_v3_diagnostics/tables/alignment_failure_chain.csv").set_index("stage")
bud = rd("portability_v4_main_evaluation/tables/paired_budget_curve.csv")
alt = rd("portability_v4_main_evaluation/tables/pairing_input_source.csv")
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.4), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
ax = axes[0]
st = [("white_probe_r2_on_probe", "Probe on probe"), ("white_probe_r2_on_task", "Probe on task"),
      ("replay_r2_on_task", "Task replay"), ("paired_linear_feature_r2", "Linear features"),
      ("paired_rff_feature_r2", "Nonlinear features")]
for i, (k, _) in enumerate(st):
    r = chain.loc[k]
    ax.errorbar([i], [r["median"]], yerr=[[r["median"] - r.q25], [r.q75 - r["median"]]], fmt="o", color="#1b6ca8", ms=5, capsize=2.5,
                lw=1.3, markeredgecolor="black", markeredgewidth=0.4)
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(range(len(st))); ax.set_xticklabels([l.replace("\n", " ") for _, l in st], fontsize=10, rotation=35, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.5, len(st) - 0.5)
set_bounds(ax, "y", float(chain.loc[[k for k, _ in st], "q25"].min()), float(chain.loc[[k for k, _ in st], "q75"].max()), step=0.2)
ax.set_ylabel("Cross-family $R^2$")
style(ax)
ax = axes[1]
xs = {8: 3, 16: 4, 32: 5, 64: 6, 128: 7, "all": 8.5}
g = bud[bud.level == "cross"]
for kind, name, mk, col in [("linear", "Paired linear", "^", C["paired_linear"]), ("rff", "Paired nonlinear", "o", C["paired_rff"])]:
    h = g[g["map"] == kind].copy()
    h["x"] = [xs[int(b) if str(b).isdigit() else b] - (0.18 if kind == "linear" else 0.0) for b in h.budget]
    h = h.sort_values("x")
    ax.plot(h.x[:-1], h.median_gap[:-1], color=col, lw=1.0)
    ci(ax, h.x, h.median_gap, h.ci95_low, h.ci95_high, col, mk, name, ms=4.5)
tb = rd("portability_v10_additional_analyses/tables/teacher_budget_and_combined.csv")
tb = tb[tb.level == "cross"].set_index("method")
keys = [f"teacher_n{b}" for b in [8, 16, 32, 64, 128]] + ["teacher"]
tx = np.array([3, 4, 5, 6, 7, 8.5]) + 0.18
th = tb.reindex(keys)
ax.plot(tx[:-1], th.median_gap.to_numpy()[:-1], color=C["pseudo_label"], lw=1.0)
ci(ax, tx, th.median_gap, th.ci95_low, th.ci95_high, C["pseudo_label"], "D", "Teacher labels", ms=4.5)
g = pd.concat([g, th.reset_index()[["median_gap", "ci95_low", "ci95_high"]]], ignore_index=True)
ax.axhline(mom_cross, color=C["moment"], ls="--", lw=1.1, label="Moment matching")
ax.set_xticks([3, 4, 5, 6, 7, 8.5]); ax.set_xticklabels(["8", "16", "32", "64", "128", "All"])
ax.set_xlim(2.4, 9.1)
set_bounds(ax, "y", min(0.0, float(g.ci95_low.min())), max(mom_cross, float(g.ci95_high.max())) + 0.2, step=0.1)
ax.set_xlabel("Unlabeled task sequences")
ax.set_ylabel("Gap to full refit (nBA)")
style(ax); legend(ax, loc="upper right", fontsize=9, handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
ax = axes[2]
labs = ["Task inputs", "Matched probe", "White probe", "Other datasets"]
for kind, mk, col, off in [("linear", "^", C["paired_linear"], -0.12), ("rff", "o", C["paired_rff"], 0.12)]:
    h = alt[alt["map"] == kind].set_index("pairing_inputs").reindex(labs)
    ci(ax, np.arange(4) + off, h.median_gap, h.ci95_low, h.ci95_high, col, mk, ms=4.5)
ax.axhline(mom_cross, color=C["moment"], ls="--", lw=1.1)
ax.set_xticks(range(4)); ax.set_xticklabels(["Task", "Matched", "White", "Other"], fontsize=10, rotation=35, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.5, 3.5)
set_bounds(ax, "y", min(0.0, float(alt.ci95_low.min())), float(alt.ci95_high.max()), step=0.1)
ax.set_xlabel("Paired inputs")
ax.set_ylabel("Gap to full refit (nBA)")
style(ax)
fig.tight_layout(w_pad=1.0); letters(fig, axes); save(fig, "fig4_requirements")

# ---------------------------------------------------------------- Figure 5: family-pair matrices
mats = {m: rd(f"portability_v4_main_evaluation/tables/family_matrix_gap_{m}.csv").set_index("source").reindex(index=FAMS, columns=FAMS)
        for m in ["moment", "paired_rff"]}
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.3))
cmap = plt.get_cmap("magma_r").copy(); cmap.set_bad("0.88")
dcmap = plt.get_cmap("RdBu_r").copy(); dcmap.set_bad("0.88")
scales = {"moment": (cmap, 0.0, 0.8, 0.2, 0.45), "paired_rff": (dcmap, -0.06, 0.06, 0.02, 0.04)}
for ax, (m, M), lab in zip(axes, mats.items(), ["Moment-matching gap (nBA)", "Paired nonlinear gap (nBA)"]):
    A = M.to_numpy(float)
    cm_, lo, hi, step, dark = scales[m]
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cm_, vmin=lo, vmax=hi, aspect="equal")
    ax.set_xticks(range(6)); ax.set_xticklabels(FAMS); ax.set_yticks(range(6)); ax.set_yticklabels(FAMS)
    ax.set_xlabel("Target family"); ax.set_ylabel("Source family")
    for i in range(6):
        for j in range(6):
            if np.isfinite(A[i, j]):
                v = 0.0 if abs(A[i, j]) < 0.005 else A[i, j]
                white = (A[i, j] > dark) if m == "moment" else (abs(A[i, j]) > dark)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if white else "black")
    style(ax)
    colorbar(fig, ax, im, lab, lo, hi, step)
fig.tight_layout(w_pad=2.0); letters(fig, axes); save(fig, "fig5_family_matrices")
print(float(np.nanmax(mats["moment"].to_numpy())), float(np.nanmin(mats["paired_rff"].to_numpy())))
