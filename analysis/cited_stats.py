import json, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, pearsonr
from pathlib import Path
ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
B = ROOT / "results"
rd = lambda p: pd.read_csv(B / p)
N = {}
def wil(a, b):
    d = (a - b).dropna(); return dict(med_a=float(a.median()), med_b=float(b.median()), diff=float(d.median()), n_better=int((d < 0).sum()), n=len(d), p=float(wilcoxon(d).pvalue))
# v6 calibration
cal = json.load(open(B / "portability_v6_measured_validation/config/tiox_calibration.json"))
cc = rd("portability_v6_measured_validation/tables/tiox_calibration_curves.csv")
N["cal_r2_in_med"] = cc.r2_in_sample.median(); N["cal_r2_out_med"] = cc.r2_left_out.median()
N["cal_keys"] = [k for k in cal.keys()]
for k in cal:
    if k not in ("curves", "x"): N["cal_" + k] = cal[k]
tb = rd("portability_v6_measured_validation/tables/tiox_three_bit_validation.csv")
N["3bit_pearson"] = pearsonr(tb.predicted, tb.measured)[0]; N["3bit_spearman"] = spearmanr(tb.predicted, tb.measured)[0]
N["3bit_pearson_unscaled"] = pearsonr(tb.predicted_unscaled, tb.measured)[0]
mp = rd("portability_v6_measured_validation/tables/measured_pattern_noise.csv"); sp = rd("portability_v6_measured_validation/tables/simulated_pattern_noise.csv")
N["meas_ntr"] = mp.noise_to_range.tolist(); N["sim_ntr"] = dict(zip(sp.family, sp.noise_to_range))
# v6 transfer tests (dataset level)
pu = rd("portability_v6_measured_validation/tables/tiox_pair_units.csv")
m = pu[pu.analysis == "main"]
def dsmed(df, meth, col="gap"):
    return df[df.method == meth].groupby(["dataset", "seed", "source", "target"])[col].median().groupby("dataset").median()
for dr in ["tx_source", "tx_target"]:
    g = m[(m.direction == dr) & (m.level == "cross")]
    N[f"T_{dr}_paired_vs_moment"] = wil(dsmed(g, "paired_rff"), dsmed(g, "moment"))
    N[f"T_{dr}_pseudo_vs_moment"] = wil(dsmed(g, "pseudo_label"), dsmed(g, "moment"))
land = rd("portability_v6_measured_validation/tables/tiox_landscape.csv")
sel = land.loc[land.groupby(["dataset", "seed"]).cv_ba.idxmax()].set_index(["dataset", "seed"]).test_nba
uni = land[land.op == 33].set_index(["dataset", "seed"]).test_nba
d = (uni - sel).groupby("dataset").median()
N["T5_universal_minus_txcv"] = dict(diff=float(d.median()), n_neg=int((d < 0).sum()), p=float(wilcoxon(d).pvalue))
# v7
mps = rd("portability_v7_measured_pair/tables/measured_pair_summary.csv")
# v8
cp = rd("portability_v8_corrected_correspondence/tables/correspondence_pairs.csv")
for nz, g in cp.groupby("noise"):
    N[f"v8_pooled_{nz}"] = dict(raw=spearmanr(g.corr_raw, g.moment_gap, nan_policy="omit")[0], corr=spearmanr(g.corr_corrected, g.moment_gap, nan_policy="omit")[0], n=len(g))
r, c = [], []
for ds, h in cp[cp.noise == "nominal"].groupby("dataset"):
    r.append(spearmanr(h.corr_raw, h.moment_gap)[0]); c.append(spearmanr(h.corr_corrected, h.moment_gap)[0])
r, c = np.array(r), np.array(c)
N["v8_within"] = dict(raw=float(np.median(r)), corr=float(np.median(c)), n_stronger=int((c < r).sum()), n=len(r), p=float(wilcoxon(c - r, alternative="less").pvalue))
# out-of-type prediction: fit on non-TiOx pairs (nominal), predict TiOx within-type and measured-pair gaps
g = cp[cp.noise == "nominal"]
tr = g[~g.tiox_involved]
te = g[g.tiox_involved & g.kind.isin(["within", "measured_pair"])]
for col in ["corr_raw", "corr_corrected"]:
    x = tr[col].clip(-1, 1); b = np.polyfit(x, tr.moment_gap, 1)
    N[f"v8_tiox_pred_{col}"] = float(np.median(np.polyval(b, te[col].clip(-1, 1))))
N["v8_tiox_obs"] = float(te.moment_gap.median())
# LODO
pr, pc = [], []
for ds in g.dataset.unique():
    a, bdf = g[g.dataset != ds], g[g.dataset == ds]
    for col, store in [("corr_raw", pr), ("corr_corrected", pc)]:
        cf = np.polyfit(a[col].clip(-1, 1), a.moment_gap, 1); store.append(pd.Series(np.polyval(cf, bdf[col].clip(-1, 1)), index=bdf.index))
pr, pc = pd.concat(pr), pd.concat(pc)
N["v8_lodo_spearman"] = dict(raw=spearmanr(pr, g.loc[pr.index, "moment_gap"])[0], corr=spearmanr(pc, g.loc[pc.index, "moment_gap"])[0])
# v9 multi-device
cu = rd("portability_v9_generality/tables/multidevice_classification_units.csv")
m = cu[(cu.method == "full_refit") & (cu.level == "cross")]
dd = m.groupby("dataset")[["nba", "single_device_nba", "raw_nba"]].median()
N["v9_multi_vs_single"] = wil(dd.single_device_nba, dd.nba); N["v9_multi_vs_raw"] = wil(dd.nba, dd.raw_nba)
for lv in ["cross", "d2d_0.50"]:
    g = cu[cu.level == lv]
    N[f"v9_{lv}_pseudo_vs_moment"] = wil(dsmed(g, "pseudo_label"), dsmed(g, "moment"))
    N[f"v9_{lv}_paired_vs_moment"] = wil(dsmed(g, "paired_rff"), dsmed(g, "moment"))
fu = rd("portability_v9_generality/tables/forecasting_units.csv")
f1 = fu[(fu.method == "full_refit") & (fu.reservoir == "single") & (fu.level == "cross")]
N["v9_fc_single_by_family"] = f1.groupby(["task", "target"]).r2.median().round(3).to_dict()
N["v9_fc_raw"] = fu.groupby("task").raw_delay_line_r2.median().round(3).to_dict()
f2 = fu[(fu.method == "full_refit") & (fu.reservoir == "multi") & (fu.level == "cross")]
N["v9_fc_multi_by_family"] = f2.groupby(["task", "target"]).r2.median().round(3).to_dict()
N["v9_fc_seeds"] = sorted(fu.seed.unique().tolist())
N = {k: (str(v) if isinstance(v, dict) and any(isinstance(kk, tuple) for kk in v) else v) for k, v in N.items()}
s = json.dumps(N, indent=1, default=float)
open(ROOT / "analysis" / "numbers.json", "w").write(s); print(s)
