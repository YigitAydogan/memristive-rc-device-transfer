from figs_common import *
import json

# ---------------------------------------------------------------- Figure 6: operating point
rank = rd("portability_v5_operating_point/tables/universal_point_v2_rank_map.csv")
pol = rd("portability_v5_operating_point/tables/policy_summary.csv")
uni = json.load(open(B / "portability_v5_operating_point/config/universal_operating_point.json"))
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.4), gridspec_kw={"width_ratios": [1.0, 1.1]})
ax = axes[0]
piv = rank.pivot(index="gain", columns="frac", values="mean_rank_v2").sort_index(ascending=False)
A = piv.to_numpy()
im = ax.imshow(A, cmap="viridis", vmin=0.2, vmax=0.8, aspect="auto")
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{v:g}" for v in piv.columns])
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if A[i, j] < 0.55 else "black")
ii, jj = list(piv.index).index(uni["gain"]), list(piv.columns).index(uni["frac"])
ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False, edgecolor="#e31a1c", lw=2.0))
ax.set_xlabel("Drive duration / switching time"); ax.set_ylabel("Input gain")
style(ax)
colorbar(fig, ax, im, "Mean percentile rank", 0.2, 0.8, 0.2)
ax = axes[1]
order = ["random", "source_selected", "consensus_others", "universal_v2", "family_default_v2", "target_cv_oracle", "test_ceiling"]
labs = ["Random", "Source", "Consensus", "Universal", "Family", "Target", "Ceiling"]
p = pol.set_index("policy").reindex(order)
cols = ["#7f7f7f", "#e08a1e", "#f2b65f", "#1b6ca8", "#6baed6", "#7b4fa0", "#000000"]
for i, (k, c_) in enumerate(zip(order, cols)):
    ci(ax, [i], [p.loc[k, "median"]], [p.loc[k, "ci95_low"]], [p.loc[k, "ci95_high"]], c_, "o")
ax.set_xticks(range(len(order))); ax.set_xticklabels([l.replace("\n", " ") for l in labs], fontsize=10, rotation=35, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.5, len(order) - 0.5)
set_bounds(ax, "y", 0.2, float(p.ci95_high.max()) + 0.01, step=0.1)
ax.set_ylabel("Target nBA")
style(ax)
fig.tight_layout(w_pad=1.5); letters(fig, axes); save(fig, "fig6_operating_point")

# ---------------------------------------------------------------- Figure 7: TiOx calibration and out-of-sample check
eng = load_engine()
D = eng.load_tiox(S / "data" / "measured")
cal = json.load(open(B / "portability_v6_measured_validation/config/tiox_calibration.json"))
q = eng.tx_unpack(np.array(cal["x"]), 3)
pr = eng.tx_predict(q, D, eng.tx_conditions())
pv = rd("portability_v6_measured_validation/tables/tiox_three_bit_validation.csv")
fig = plt.figure(figsize=(W_FULL, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05])
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[:, 2])]
cm = plt.get_cmap("viridis")
specs = [("int", eng.TX_INTERVALS, "Pulse number", None), ("int_relax", eng.TX_INTERVALS, "Time after train (ms)", lambda c: f"{c} µs"),
         ("amp", eng.TX_AMPS, "Pulse number", None), ("amp_relax", eng.TX_AMPS, "Time after train (ms)", lambda c: f"{c:.1f} V")]
for ax, (ser, conds, xl, lab) in zip(axes[:4], specs):
    ally = []
    for i, c_ in enumerate(conds):
        col = cm(i / max(1, len(conds) - 1) * 0.9)
        y = eng.tx_measured(D, (ser, c_)) * 1e6
        x = ((D["relax_int"][c_][0] if ser == "int_relax" else D["relax_amp"][c_][0]) * 1e3) if "relax" in ser else np.arange(len(y))
        stp = max(1, len(x) // 25)
        ax.plot(x[::stp], y[::stp], "o", ms=2.2, color=col, alpha=0.8, mew=0)
        ax.plot(x, pr[(ser, c_)] * 1e6, "-", lw=1.0, color=col, label=lab(c_) if lab else None)
        ally += [np.nanmin(y), np.nanmax(y)]
    if "relax" in ser:
        set_bounds(ax, "x", 0, 30, step=10)
    else:
        set_bounds(ax, "x", 0, 200, step=50)
    set_bounds(ax, "y", min(ally), max(ally), step=2 if ser.startswith("int") else 1)
    ax.set_xlabel(xl); ax.set_ylabel("Current (µA)")
    style(ax)
    if lab:
        legend(ax, loc="upper right", fontsize=9, ncol=2, handlelength=1.0, columnspacing=0.6, borderpad=0.3, labelspacing=0.15, handletextpad=0.3)
ax = axes[4]
for (dl, g), col in zip(pv.groupby("read_delay_us"), ["#1b6ca8", "#2a9d5c", "#e08a1e", "#c0392b"]):
    ax.errorbar(g.predicted, g.measured, yerr=1.96 * g.measured_se, fmt="o", ms=3.5, color=col, capsize=1.5, lw=0.8,
                markeredgecolor="black", markeredgewidth=0.3, label=f"{dl:.0f} µs")
lo_, hi_ = min(1.0, float((pv.measured - 1.96 * pv.measured_se).min())), max(float(pv.measured.max() + 2 * pv.measured_se.max()), float(pv.predicted.max()))
ax.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.8)
set_bounds(ax, "x", lo_, hi_, step=0.01); set_bounds(ax, "y", lo_, hi_, step=0.01)
ax.set_xlabel("Predicted, relative to input 000"); ax.set_ylabel("Measured, relative to input 000")
style(ax); legend(ax, loc="upper left", fontsize=9, title="Read delay", title_fontsize=9)
fig.tight_layout(h_pad=1.2, w_pad=1.2); letters(fig, axes); save(fig, "fig7_tiox_calibration")
