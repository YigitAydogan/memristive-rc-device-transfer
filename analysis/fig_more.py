from figs_common import *
import json
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- Figure 8: measured noise, calibrated device, measured pair
mpn = rd("portability_v6_measured_validation/tables/measured_pattern_noise.csv")
spn = rd("portability_v6_measured_validation/tables/simulated_pattern_noise.csv")
tts = rd("portability_v6_measured_validation/tables/tiox_transfer_summary.csv")
mps = rd("portability_v7_measured_pair/tables/measured_pair_summary.csv")
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.6), gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})
ax = axes[0]
labs = ["0 µs", "108 µs", "648 µs", "1728 µs"] + ["TiOx model" if f == "TX" else f for f in spn.family]
vals = list(mpn.noise_to_range) + list(spn.noise_to_range)
cols = ["#2a9d5c"] * len(mpn) + ["#1b6ca8"] * (len(spn) - 1) + ["#17becf"]
ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", lw=0.5)
ax.axhspan(float(mpn.noise_to_range.min()), float(mpn.noise_to_range.max()), color="#2a9d5c", alpha=0.12, lw=0)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=9, rotation=90)
ax.set_xlim(-0.6, len(vals) - 0.4)
set_bounds(ax, "y", 0.0, max(vals), step=0.05)
ax.set_ylabel("Repeat noise / response range")
style(ax)
from matplotlib.patches import Patch
legend(ax, loc="upper left", fontsize=9, handles=[Patch(facecolor="#2a9d5c", edgecolor="black", lw=0.5, label="Measured TiOx"),
                                                   Patch(facecolor="#1b6ca8", edgecolor="black", lw=0.5, label="Simulated")])
ax = axes[1]
order = [("tx_source", "cross", "TiOx\nas source"), ("tx_target", "cross", "TiOx\nas target"), ("tx_within", "d2d_0.05", "TiOx\n5%"), ("tx_within", "d2d_0.50", "TiOx\n50%")]
m4 = [("moment", "^"), ("paired_rff", "o"), ("pseudo_label", "D"), ("twin_paired_rff", "s")]
s = tts[tts.analysis == "main"]
allv = [0.0]
for (m, mk), off in zip(m4, [-0.21, -0.07, 0.07, 0.21]):
    g = [s[(s.direction == d) & (s.level == l) & (s.method == m)].iloc[0] for d, l, _ in order]
    ci(ax, np.arange(4) + off, [r.median_gap for r in g], [r.ci95_low for r in g], [r.ci95_high for r in g], C[m], mk, ms=4.2)
    allv += [r.ci95_high for r in g] + [r.ci95_low for r in g]
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([o[2].replace("\n", " ") for o in order], fontsize=9, rotation=35, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.5, 3.5)
set_bounds(ax, "y", min(allv), max(allv) + 0.15, step=0.1)
ax.set_ylabel("Gap to full refit (nBA)")
style(ax)
ax = axes[2]
mm = [("zero_shot", "Zero shot"), ("moment", "Moment matching"), ("coral", "CORAL"), ("paired_linear", "Paired linear"), ("paired_rff", "Paired nonlinear"),
      ("pseudo_label", "Teacher labels")]
allv = [0.0]
for pk, mk, off, name in [("measured_pair_shared_protocol", "o", -0.12, "Same protocol"), ("measured_pair_own_interface", "s", 0.12, "Own interface")]:
    g = mps[(mps.pair == pk) & (mps.noise == "nominal_noise") & (mps.op_policy == "universal")].set_index("method").reindex([m for m, _ in mm])
    ci(ax, np.arange(len(mm)) + off, g.median_gap, g.ci95_low, g.ci95_high, "#1b6ca8" if mk == "o" else "#e08a1e", mk, name, ms=4.2)
    allv += list(g.ci95_high) + list(g.ci95_low)
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(range(len(mm))); ax.set_xticklabels([l.replace("\n", " ") for _, l in mm], fontsize=9, rotation=35, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.5, len(mm) - 0.5)
set_bounds(ax, "y", min(allv), max(allv), step=0.1)
ax.set_ylabel("Gap to full refit (nBA)")
style(ax); legend(ax, loc="upper right", fontsize=9)
handles = [plt.Line2D([], [], marker=mk, ls="", color=C[m], markeredgecolor="black", markeredgewidth=0.4, ms=4.5) for m, mk in m4]
axes[1].legend(handles, ["Moment matching", "Paired nonlinear", "Teacher labels", "Model twin"], loc="upper right", fontsize=9,
               frameon=True, fancybox=False, edgecolor="black", borderpad=0.3, handletextpad=0.2, labelspacing=0.2).get_frame().set_linewidth(LEGEND_W)
fig.tight_layout(w_pad=1.0); letters(fig, axes); save(fig, "fig8_measured_device")

# ---------------------------------------------------------------- Figure 9: repeatability-corrected correspondence
cp = rd("portability_v8_corrected_correspondence/tables/correspondence_pairs.csv")
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.2))
cols = {"measured": "#2a9d5c", "nominal": "#1b6ca8", "high": "#c0392b"}
labs = {"measured": "Measured TiOx noise", "nominal": "Nominal noise", "high": "High noise"}
for ax, x, xl in [(axes[0], "corr_raw", "Feature correspondence, raw"), (axes[1], "corr_corrected", "Feature correspondence, corrected")]:
    for n in ["high", "nominal", "measured"]:
        g = cp[cp.noise == n]
        ax.scatter(g[x], g.moment_gap, s=3, alpha=0.35, color=cols[n], lw=0, label=labs[n])
    set_bounds(ax, "x", float(cp[["corr_raw", "corr_corrected"]].min().min()) - 0.05, 1.05, step=0.4)
    set_bounds(ax, "y", float(cp.moment_gap.min()), float(cp.moment_gap.max()), step=0.4)
    ax.set_xlabel(xl); ax.set_ylabel("Moment gap (nBA)")
    style(ax)
legend(axes[0], loc="lower left", fontsize=9, markerscale=3, handletextpad=0.2)
fig.tight_layout(w_pad=1.5); letters(fig, axes); save(fig, "fig9_corrected_correspondence")

# ---------------------------------------------------------------- Figure 10: generality
cs = rd("portability_v9_generality/tables/multidevice_classification_summary.csv")
fs = rd("portability_v9_generality/tables/forecasting_summary.csv")
fu = rd("portability_v9_generality/tables/forecasting_units.csv")
cu = rd("portability_v9_generality/tables/multidevice_classification_units.csv")
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 3.8), gridspec_kw={"width_ratios": [1.15, 1.1, 1.0]})
ax = axes[0]
mm = [("zero_shot", "Zero shot"), ("moment", "Moment matching"), ("paired_rff", "Paired nonlinear"), ("pseudo_label", "Teacher labels"),
      ("scratch_k5", "Scratch, 5 per class"), ("moment_warm_k5", "Warm start, 5 per class")]
allv = [0.0]
for (lv, name, col, mk), off in zip([("cross", "Other family", "#7f7f7f", "s"), ("d2d_0.05", "Same family, 5%", "#9ecae1", "o"),
                                     ("d2d_0.50", "Same family, 50%", "#08519c", "^")], [-0.2, 0.0, 0.2]):
    g = cs[cs.level == lv].set_index("method").reindex([m for m, _ in mm])
    ci(ax, np.arange(len(mm)) + off, g.median_gap, g.ci95_low, g.ci95_high, col, mk, name, ms=4.0)
    allv += list(g.ci95_high) + list(g.ci95_low)
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(range(len(mm))); ax.set_xticklabels([l for _, l in mm], fontsize=9, rotation=35, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.5, len(mm) - 0.5)
set_bounds(ax, "y", min(allv), max(allv) + 0.1, step=0.1)
ax.set_ylabel("Gap to full refit (nBA)")
style(ax); legend(ax, loc="upper right", fontsize=9, borderpad=0.3, handletextpad=0.2, labelspacing=0.2)
ax = axes[1]
groups = [("NARMA10", "single", "NARMA-10, one device"), ("NARMA10", "multi", "NARMA-10, four devices"),
          ("MackeyGlass", "single", "Mackey-Glass, one device"), ("MackeyGlass", "multi", "Mackey-Glass, four devices")]
allv = [0.0]
for (m, mk), off in zip([("moment", "^"), ("paired_rff", "o"), ("pseudo_label", "D")], [-0.16, 0.0, 0.16]):
    g = [fs[(fs.task == t) & (fs.reservoir == r) & (fs.level == "cross") & (fs.method == m)].iloc[0] for t, r, _ in groups]
    ci(ax, np.arange(4) + off, [x.median_gap for x in g], [x.ci95_low for x in g], [x.ci95_high for x in g], C[m], mk, ms=4.0)
    allv += [x.ci95_high for x in g] + [x.ci95_low for x in g]
ax.axhline(0, color="0.5", ls=":", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([g[2] for g in groups], fontsize=9, rotation=35, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.5, 3.5)
set_bounds(ax, "y", min(allv), max(allv) + 0.2, step=0.2)
ax.set_ylabel("Cross-family gap (skill)")
style(ax)
handles = [plt.Line2D([], [], marker=mk, ls="", color=C[m], markeredgecolor="black", markeredgewidth=0.4, ms=4.5) for m, mk in [("moment", "^"), ("paired_rff", "o"), ("pseudo_label", "D")]]
ax.legend(handles, ["Moment matching", "Paired nonlinear", "Teacher labels"], loc="upper left", fontsize=9, frameon=True, fancybox=False,
          edgecolor="black", borderpad=0.3, handletextpad=0.2, labelspacing=0.2).get_frame().set_linewidth(LEGEND_W)
ax = axes[2]
fm = fu[fu.method == "moment"].groupby(["task", "reservoir", "source", "target", "level"], as_index=False)[["gap", "corr_corrected"]].median()
cm_ = cu[cu.method == "moment"].groupby(["dataset", "source", "target", "level"], as_index=False)[["gap", "corr_corrected"]].median()
ax.scatter(cm_.corr_corrected, cm_.gap, s=4, alpha=0.3, color="#1b6ca8", lw=0, label="Classification")
for t, col, mk in [("NARMA10", "#7b4fa0", "D"), ("MackeyGlass", "#e08a1e", "^")]:
    g = fm[fm.task == t]
    ax.scatter(g.corr_corrected, g.gap, s=10, alpha=0.6, color=col, marker=mk, lw=0, label="NARMA-10" if t == "NARMA10" else "Mackey-Glass")
allx = pd.concat([cm_.corr_corrected, fm.corr_corrected]); ally = pd.concat([cm_.gap, fm.gap])
set_bounds(ax, "x", float(allx.min()) - 0.05, 1.05, step=0.6)
set_bounds(ax, "y", float(ally.min()), float(ally.max()) + 0.3, step=0.4)
ax.set_xlabel("Corrected correspondence"); ax.set_ylabel("Moment-matching gap")
style(ax); legend(ax, loc="upper right", fontsize=9, markerscale=2, handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
fig.tight_layout(w_pad=0.8); letters(fig, axes); save(fig, "fig10_generality")

# ---------------------------------------------------------------- Figure 1: problem, approaches and device responses
eng = load_engine()
for f in eng.FAMILY_CODES:
    eng.T_SWITCH[f] = eng.calibrate_switch_time(f)
    eng.T_RECOVER[f] = eng.recovery_duration(f)
D = eng.load_tiox(S / "data" / "measured")
cal = json.load(open(B / "portability_v6_measured_validation/config/tiox_calibration.json"))
eng.register_tx(cal["params"], eng.observed_range(D))
u = 33
rng = np.random.default_rng(3)
U = np.clip(np.cumsum(rng.normal(0, 0.35, 60)) * 0.25 + 0.6 * np.sin(np.linspace(0, 4 * np.pi, 60)), -1, 1)[None, :]
fams = eng.FAMILY_CODES + ["TX"]
names = ["Input"] + ["TiOx" if f == "TX" else f for f in fams]
fig = plt.figure(figsize=(W_FULL, 10.0))
gs = fig.add_gridspec(2, 1, height_ratios=[3.55, 5.9], hspace=0.08)
ax0 = fig.add_subplot(gs[0, 0])
sub = gs[1, 0].subgridspec(len(names), 1, hspace=0.35)
axs = [fig.add_subplot(sub[k, 0]) for k in range(len(names))]
diagram = plt.imread(str(OUT / "fig1a_diagram.png"))
ax0.imshow(diagram, interpolation="lanczos")
ax0.axis("off")
traces = [U[0]]
for f in fams:
    traces.append(eng.simulate(eng.nominal_instance(f), U, [u], "traj", ("fig1", 0))[0, :, 0].astype(float))
for k, (ax, z, nm) in enumerate(zip(axs, traces, names)):
    z = np.asarray(z, float)
    zn = (z - z.min()) / max(z.max() - z.min(), 1e-12)
    col = "black" if nm == "Input" else FAM_COLORS["TX" if nm == "TiOx" else nm]
    ax.plot(np.arange(len(zn)), zn, color=col, lw=1.3)
    set_bounds(ax, "x", 0, 60, step=10)
    ax.set_ylim(-0.15, 1.15); ax.yaxis.set_major_locator(FixedLocator([0, 1])); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylim(0, 1)
    ax.text(1.012, 0.5, nm, transform=ax.transAxes, fontsize=10, va="center", color=col, fontweight="bold")
    ax.tick_params(labelsize=9)
    if k < len(names) - 1:
        ax.tick_params(labelbottom=False)
    style(ax)
axs[-1].set_xlabel("Time step")
fig.text(0.0, 0.5, "", fontsize=1)
fig.canvas.draw()
yl = fig.supylabel("Response, scaled to its range", x=0.035, y=float(np.mean([axs[0].get_position().y1, axs[-1].get_position().y0])), fontsize=11.5)
fig.subplots_adjust(left=0.1, right=0.92, top=0.97, bottom=0.07)
fig.canvas.draw()
rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
xL = inv.transform((yl.get_window_extent(rend).x0, 0))[0]
# Align the outer box of the diagram (teacher-label block, border centres at image columns BL and BR)
# with the left and right spines of the panel b axes.
BL, BR = 134.5, 7669.5
pb = axs[0].get_position()
scale = (pb.x1 - pb.x0) / (BR - BL)                      # figure-width fraction per image pixel
w_new = diagram.shape[1] * scale
h_new = diagram.shape[0] * scale * fig.get_figwidth() / fig.get_figheight()
p0 = ax0.get_position()
ax0.set_position([pb.x0 - BL * scale, p0.y1 - h_new, w_new, h_new])
ax0.set_aspect("auto")
fig.canvas.draw()
fig.text(xL, ax0.get_position().y1 + 0.005, "a", fontsize=13, fontweight="bold", ha="left", va="bottom")
fig.text(xL, axs[0].get_position().y1 + 0.008, "b", fontsize=13, fontweight="bold", ha="left", va="bottom")
save(fig, "fig1_overview")
