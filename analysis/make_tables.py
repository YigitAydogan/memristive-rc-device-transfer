import json, math
import numpy as np, pandas as pd
from pathlib import Path
import os
S = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
B, T = S / "results", S / "analysis" / "tables"
T.mkdir(parents=True, exist_ok=True)
rd = lambda p: pd.read_csv(B / p)
def f(v, d=3):
    s = f"{v:.{d}f}"
    return "0." + "0" * d if s in ("-0." + "0" * d,) else s.replace("-", "$-$")

# ---------------- datasets
mod = rd("portability_v4_main_evaluation/tables/dataset_moderators.csv").set_index("dataset")
ab = rd("portability_v2/tables/absolute_performance.csv").set_index("dataset")
prot = json.load(open(B / "portability_v4_main_evaluation/config/protocol_v4.json"))
prior = json.load(open(B / "portability_v2/config/protocol.json"))["prior_datasets"]
held = prot["held_out_datasets"]
TEST_HELD = {"MoteStrain": 1252, "SonyAIBORobotSurface1": 601, "Coffee": 28, "Meat": 60, "Lightning7": 73, "MedicalImages": 760,
             "ProximalPhalanxOutlineCorrect": 291, "Ham": 105, "OSULeaf": 242, "Fish": 175}
rows = []
for ds in prot["datasets"]:
    tier = "A" if ds in prior else ("C" if ds in held else "B")
    ntr = int(round(math.exp(mod.loc[ds, "log_n_train"]))); L = int(round(math.exp(mod.loc[ds, "log_length"])))
    nte = int(ab.loc[ds, "n_test"]) if ds in ab.index else TEST_HELD[ds]
    if ds in ab.index:
        assert int(ab.loc[ds, "n_train"]) == ntr and int(ab.loc[ds, "length"]) == L
    rows.append((tier, ds, ntr, nte, L, int(mod.loc[ds, "n_classes"])))
rows.sort(key=lambda r: (r[0], r[1].lower()))
name = {"ProximalPhalanxOutlineCorrect": "ProximalPhalanxOutline-\\\\ \\quad Correct", "SonyAIBORobotSurface1": "SonyAIBORobotSurface1",
        "UWaveGestureLibraryX": "UWaveGestureLibraryX"}
lines = []
half = (len(rows) + 1) // 2
for i in range(half):
    a = rows[i]; b = rows[i + half] if i + half < len(rows) else None
    ca = f"{a[0]} & {a[1]} & {a[2]} & {a[3]} & {a[4]} & {a[5]}"
    cb = f"{b[0]} & {b[1]} & {b[2]} & {b[3]} & {b[4]} & {b[5]}" if b else "& & & & & "
    lines.append(ca + " & & " + cb + r" \\")
(T / "datasets.tex").write_text("\n".join(lines) + "\n")

# ---------------- absolute performance (v2, 18 datasets)
def nba(ba, C): return (ba - 1 / C) / (1 - 1 / C)
fams = ["LD", "VT", "VV", "ZH", "DF", "DN"]
cap = rd("portability_v2/tables/capacity_at_selected_ops.csv").set_index("family")
md = rd("portability_v9_generality/tables/multidevice_classification_units.csv")
print(md.columns.tolist()[:30])
out = []
for fm in fams:
    v = nba(ab[f"oracle_{fm}_ba"], ab.classes)
    w = (ab[f"oracle_{fm}_ba"] > ab.raw_binned_ridge_ba).sum()
    out.append((fm, float(np.median(v)), int(w), float(cap.loc[fm, "C_total"])))
best = ab[[f"oracle_{fm}_ba" for fm in fams]].max(1)
base = [("Raw input, same features", nba(ab.raw_binned_ridge_ba, ab.classes)), ("One-nearest neighbour", nba(ab.nn1_euclidean_ba, ab.classes)),
        ("MiniRocket", nba(ab.minirocket_ba, ab.classes)), ("Best device per dataset", nba(best, ab.classes))]
L2 = [f"{fm} & {f(m)} & {w} of 18 & {f(c,1)} \\\\" for fm, m, w, c in out]
L2.append(r"\midrule")
for nm, v in base:
    extra = f"{int((best > ab.raw_binned_ridge_ba).sum())} of 18" if nm.startswith("Best") else ""
    L2.append(f"{nm} & {f(float(np.median(v)))} & {extra} & \\\\")
(T / "absolute.tex").write_text("\n".join(L2) + "\n")
print("\n".join(L2))

# ---------------- robustness
rb = rd("portability_v4_main_evaluation/tables/robustness_noise_features_readout.csv")
lab = {"main": "Nominal", "noise_none": "No cycle or read noise", "noise_high": "Three-fold cycle and read noise", "bins12": "12 bins",
       "bins48": "48 bins", "means24": "Bin means only", "logistic": "Logistic readout"}
L3 = []
for _, r in rb.iterrows():
    L3.append(f"{lab[r.analysis]} & {f(r.median_feature_corr)} & {f(r.median_full_refit_nba)} & {f(r.moment_gap)} & {f(r.paired_rff_gap)} & {int(r.paired_vs_moment_datasets_a_better)} of {int(r.paired_vs_moment_datasets)} \\\\")
(T / "robust.tex").write_text("\n".join(L3) + "\n")
print("\n".join(L3))
