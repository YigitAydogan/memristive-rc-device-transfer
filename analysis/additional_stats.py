import json, math, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from pathlib import Path
S = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
B = S / "results"; R = S / "results"; T = S / "analysis" / "tables"
T.mkdir(parents=True, exist_ok=True)
N = {}
rng = np.random.default_rng(20261021)

def ds_boot(df, fn, n=10000):
    """Statistic fn over rows; CI by resampling datasets."""
    groups = [g for _, g in df.groupby("dataset")]
    obs = fn(df)
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        bs.append(fn(pd.concat([groups[i] for i in pick])))
    bs = np.array(bs, float)
    return float(obs), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))

# ---------------- 1. Ceiling with widened bins
u10 = pd.read_csv(B / "portability_v10_additional_analyses/tables/v10_units.csv")
cr = u10[(u10.analysis == "main") & (u10.level == "cross")].copy()
cr["d"] = cr.full_refit_nba - cr.source_nba
cr["bin"] = pd.cut(cr.d, [-3, -0.05, 0.05, 3], labels=["worse", "similar", "better"])
rows = []
for m in ["teacher", "paired_rff", "teacher_fix_k5", "teacher_warm_k5"]:
    g = cr[cr.method == m]
    r = {"method": m}
    for b in ["worse", "similar", "better"]:
        gb = g[g.bin == b]
        r[b] = float(gb.gap.median()); r[f"n_{b}"] = int(len(gb))
    gb = g[g.bin == "better"]
    r["above_source_better"] = float((gb.nba - gb.source_nba).median())
    rows.append(r)
ceil = pd.DataFrame(rows); N["ceiling"] = ceil.to_dict("records")
N["ceiling_units_total"] = int(cr[cr.method == "teacher"].bin.notna().sum())
nm = {"teacher": "Teacher labels", "paired_rff": "Paired nonlinear map", "teacher_fix_k5": "Teacher labels, five per class corrected",
      "teacher_warm_k5": "Teacher labels, warm start with five per class"}
f3 = lambda v: f"{v:.3f}".replace("-", "$-$")
open(T / "ceiling.tex", "w").write("\n".join(f"{nm[r['method']]} & {f3(r['worse'])} & {f3(r['similar'])} & {f3(r['better'])} & {f3(r['above_source_better'])} \\\\" for r in rows) + "\n")

# ---------------- 2. Decision rule error analysis
lo = pd.read_csv(B / "portability_v10_additional_analyses/tables/threshold_rule_lodo.csv")
lo["grp"] = np.where(lo.tier == "held_out", "H", "D")
lo["excess"] = lo.moment_gap - lo.teacher_gap
rule = {}
for name, sub in [("all", lo), ("D", lo[lo.grp == "D"]), ("H", lo[lo.grp == "H"])]:
    st = sub[sub.use_moment]
    r = {"pairs": int(len(sub)), "routed": int(len(st)),
         "routing_rate": ds_boot(sub, lambda d: d.use_moment.mean()),
         "rule_gap_median": float(sub.rule_gap.median()), "teacher_gap_median": float(sub.teacher_gap.median())}
    if len(st):
        for tol in [0.02, 0.05]:
            r[f"false_rate_{tol}"] = ds_boot(st, lambda d, tol=tol: (d.moment_gap > tol).mean())
            r[f"excess_rate_{tol}"] = ds_boot(st, lambda d, tol=tol: (d.excess > tol).mean())
        r["excess_median"] = float(st.excess.median()); r["excess_q90"] = float(np.quantile(st.excess, 0.9)); r["excess_max"] = float(st.excess.max())
        r["moment_gap_q90"] = float(np.quantile(st.moment_gap, 0.9)); r["moment_gap_max"] = float(st.moment_gap.max())
        r["routed_by_level"] = st.level.value_counts().to_dict()
    rule[name] = r
N["rule"] = rule
# leave one family out: choose threshold on pairs not involving family f, apply to pairs involving f
pt = pd.read_csv(B / "portability_v10_additional_analyses/tables/correspondence_pairs_v10.csv")
def choose(d, col="corr_corrected"):
    d = d.dropna(subset=[col, "moment_gap", "teacher_gap"])
    best_t, best_v = np.nan, np.inf
    for t in np.round(np.arange(0.99, 0.29, -0.01), 2):
        v = float(np.where(d[col] >= t, d.moment_gap, d.teacher_gap).mean())
        if v < best_v - 1e-12:
            best_t, best_v = float(t), v
    return best_t
lofo = []
for f in sorted(set(pt.source)):
    inv = (pt.source == f) | (pt.target == f)
    t = choose(pt[~inv]); te = pt[inv].dropna(subset=["corr_corrected"])
    um = te.corr_corrected >= t
    lofo.append({"family": f, "threshold": t, "pairs": int(len(te)), "routed": int(um.sum()),
                 "false_0.05": float((te[um].moment_gap > 0.05).mean()) if um.any() else np.nan,
                 "rule_gap_median": float(np.where(um, te.moment_gap, te.teacher_gap).mean())})
lofo = pd.DataFrame(lofo); N["rule_leave_one_family_out"] = lofo.to_dict("records")
st = pd.concat([pt[(pt.source == r.family) | (pt.target == r.family)].dropna(subset=["corr_corrected"]).assign(t=r.threshold) for r in lofo.itertuples()])
st = st[st.corr_corrected >= st.t]
N["lofo_pooled"] = {"routed_units": int(len(st)), "false_0.05": float((st.moment_gap > 0.05).mean()), "false_0.02": float((st.moment_gap > 0.02).mean()),
                    "thresholds": [float(lofo.threshold.min()), float(lofo.threshold.max())]}

# ---------------- 3. Development versus held-out for H1-H3, H8, operating point
u4 = pd.read_csv(R / "portability_v4_main_evaluation/tables/all_units.csv", usecols=["dataset", "tier", "seed", "source", "target", "level", "analysis", "method", "gap"])
u4 = u4[u4.analysis == "main"]
pair = u4.groupby(["dataset", "tier", "source", "target", "level", "method"]).gap.median().reset_index()
dsg = lambda m, lvl="cross": pair[(pair.level == lvl) & (pair.method == m)].groupby(["dataset", "tier"]).gap.median()
dh = []
for h, a, b in [("H1", "paired_rff", "moment"), ("H2", "paired_rff", "paired_linear"), ("H3", "twin_paired_rff", "moment"), ("H8", "paired_rff", "pseudo_label")]:
    A, Bv = dsg(a), dsg(b)
    for grp, mask in [("D", A.index.get_level_values(1) != "held_out"), ("H", A.index.get_level_values(1) == "held_out")]:
        x, y = A[mask], Bv[mask]
        alt = "two-sided" if h == "H8" else "less"
        dh.append({"test": h, "set": grp, "n": int(len(x)), "a": float(x.median()), "b": float(y.median()),
                   "a_better": int((x < y).sum()), "p": float(wilcoxon(x - y, alternative=alt).pvalue)})
pol = pd.read_csv(B / "portability_v5_operating_point/tables/policy_rows.csv")
d = pol.groupby(["dataset", "tier"])[["universal_v2", "source_selected"]].median().reset_index()
for grp, x in [("D", d[d.tier != "held_out"]), ("H", d[d.tier == "held_out"])]:
    diff = x.universal_v2 - x.source_selected
    dh.append({"test": "Universal point", "set": grp, "n": int(len(x)), "a": float(x.universal_v2.median()), "b": float(x.source_selected.median()),
               "a_better": int((diff > 0).sum()), "p": float(wilcoxon(diff, alternative="greater").pvalue)})
N["dev_heldout"] = dh
# H8 paired difference interval
A, Bv = dsg("paired_rff"), dsg("pseudo_label"); dd = (A - Bv).to_numpy()
bs = np.median(dd[rng.integers(0, len(dd), (10000, len(dd)))], axis=1)
N["H8_diff"] = [float(np.median(dd)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

# ---------------- 4. H7 dependence sensitivity
cap = pd.read_csv(R / "portability_v4_main_evaluation/units/capacity_emulation_98fd8d185f36.csv")
cm = cap.groupby(["source", "target", "op"], as_index=False).median(numeric_only=True)
fams = sorted(set(cm.source))
h7 = {}
for key, x in [("H7a", "target_total_capacity"), ("H7b", "memory_profile_corr")]:
    g = cm.dropna(subset=[x, "emulation_r2"])
    obs = float(spearmanr(g[x], g.emulation_r2)[0])
    # family bootstrap: resample families with replacement, keep ordered pairs between sampled families
    bs = []
    for _ in range(4000):
        pick = list(rng.choice(fams, len(fams), replace=True))
        parts = [g[(g.source == s) & (g.target == t)] for s in pick for t in pick if s != t]
        dfb = pd.concat(parts) if parts else pd.DataFrame()
        if len(dfb) > 10 and dfb[x].std() > 0:
            bs.append(spearmanr(dfb[x], dfb.emulation_r2)[0])
    bs = np.array(bs)
    lofo_r = {f: float(spearmanr(g[(g.source != f) & (g.target != f)][x], g[(g.source != f) & (g.target != f)].emulation_r2)[0]) for f in fams}
    # within source family: does the association hold within each source?
    within = {f: float(spearmanr(g[g.source == f][x], g[g.source == f].emulation_r2)[0]) for f in fams}
    h7[key] = {"rho": obs, "family_boot_ci": [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))],
               "family_boot_frac_le0": float(np.mean(bs <= 0)), "leave_one_family_out_range": [min(lofo_r.values()), max(lofo_r.values())],
               "within_source_family": within, "points": int(len(g))}
N["H7"] = h7

# ---------------- 5. 3-bit quantitative check
for nmk, f in [("joint", "three_bit_run_A_joint.csv"), ("separate", "three_bit_run_B_independent.csv")]:
    t = pd.read_csv(B / "portability_v10_additional_analyses/tables" / f)
    a, b = t.measured - 1, t.predicted_unscaled - 1
    N[f"3bit_{nmk}"] = {"slope_measured_on_predicted": float((a @ b) / (b @ b)), "rmse_unscaled": float(np.sqrt(np.mean((a - b) ** 2))),
                        "mae_unscaled": float(np.mean(np.abs(a - b))), "rmse_measured_sd": float(a.std()),
                        "overshoot_by_delay": {float(k): float((g.predicted_unscaled - 1).max() / (g.measured - 1).max()) for k, g in t.groupby("read_delay_us")}}

# ---------------- 6. TiOx parameter table
cal = json.load(open(B / "portability_v6_measured_validation/config/tiox_calibration.json"))
sep = json.load(open(B / "portability_v10_additional_analyses/config/run_b_independent.json"))
N["params_joint"] = cal["params"]; N["params_sep"] = sep["params"]
N["at_bound_joint"] = cal["parameters_at_bound"]; N["at_bound_sep"] = sep["parameters_at_bound"]
json.dump(N, open(S / "analysis" / "review_numbers.json", "w"), indent=1, default=float)
print(json.dumps(N, indent=1, default=float)[:9000])
