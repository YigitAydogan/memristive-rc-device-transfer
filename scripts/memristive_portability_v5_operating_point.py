"""
Memristive reservoir portability, v5: independent test of a universal operating point.

The universal operating point (input gain and drive duration as a fraction of each device's switching time) is
chosen from the v2 runs alone (device seeds 11 to 33, 18 datasets, training cross-validation scores only). It is then
evaluated on the independent v4 runs (fresh seeds 44 to 66, 28 datasets including 10 held out) against the
source-selected point, a consensus of the other families, family-specific defaults, the target's own CV choice and
the test ceiling. The transfer comparisons are rerun at the universal point.

Paste into one Colab cell with Drive already mounted. The script does not mount Drive. Requires the finished
portability_v2 and portability_v4_main_evaluation folders. Units are checkpointed, so an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_VERSION = "5.0.0"
SMOKE = os.environ.get("PORTV2_SMOKE", "0").strip() == "1"
IDEAL = os.environ.get("PORTV2_IDEAL", "0").strip() == "1"
RUN_MINIROCKET = os.environ.get("PORTV2_MINIROCKET", "1").strip() == "1" and not SMOKE
DATASET_FILTER = [d.strip() for d in os.environ.get("PORTV2_DATASETS", "").split(",") if d.strip()]

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/memristive_device_generalization")
LOCAL_ROOT = Path("/content/memristive_device_generalization")
ENV_ROOT = os.environ.get("MEMRISTIVE_PROJECT_ROOT", "").strip()

# Datasets previously examined in the earlier project (reported as their own tier)
# and datasets never used before (fresh tier).
PRIOR_DATASETS = [
    "ECG5000", "FordA", "Wafer", "UWaveGestureLibraryX",
    "CricketX", "RefrigerationDevices", "Earthquakes", "StarLightCurves",
]
FRESH_DATASETS = [
    "GunPoint", "TwoPatterns", "SyntheticControl", "Trace", "FaceAll",
    "Strawberry", "Yoga", "ECG200", "Plane", "Symbols",
]

SEEDS = [11, 22, 33] if not SMOKE else [11]
GAINS = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 1.00]) if not SMOKE else np.array([0.35, 1.00])
FRACS = np.array([0.03, 0.06, 0.125, 0.25, 0.50, 1.00]) if not SMOKE else np.array([0.06, 0.50])
N_BINS = 24
ALPHA_REL = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
K_SHOTS = [1, 2, 5, 10] if not SMOKE else [1, 2]
LAMBDA_PRIOR = 10.0          # warm-start and scratch k-shot shrinkage, in sample units
CORAL_EPS = 0.05
FIR_TAPS = 24
QUAD_TAPS = 12
PROBE_EPISODES = 50 if not SMOKE else 12
PROBE_LENGTH = 256 if not SMOKE else 128
PROBE_TRAIN_EPISODES = int(round(0.75 * PROBE_EPISODES))
CAP_STATE_LAGS = 24
CAP_MAX_DELAY_D1 = 39
CAP_MAX_DELAY_D2 = 14
CAP_MAX_DELAY_D3 = 9
CAP_THRESHOLD = 0.01
RETENTION_PER_SYMBOL = 0.90
SIGMA_D2D = 0.0 if IDEAL else 0.05
SIGMA_C2C = 0.0 if IDEAL else 0.002
SIGMA_READ = 0.0 if IDEAL else 0.01   # decades of read conductance
N_BOOT = 10000 if not SMOKE else 300
MASTER_SEED = 20261020
EPS = 1e-12

PRETTY_METHOD = {
    "zero_shot": "Zero shot",
    "moment": "Moment matching",
    "coral": "CORAL",
    "replay_fir": "Replay FIR",
    "probe_fir": "Probe FIR",
    "probe_quad": "Probe quadratic",
    "probe_fir_mm": "Probe FIR + moment",
    "full_refit": "Full refit",
}
LABEL_FREE = ["coral", "replay_fir", "probe_fir", "probe_quad", "probe_fir_mm"]

# =============================================================================
# Paths and logging
# =============================================================================


def choose_root() -> Path:
    if ENV_ROOT:
        return Path(ENV_ROOT).expanduser().resolve()
    if DEFAULT_DRIVE_ROOT.parent.exists():
        return DEFAULT_DRIVE_ROOT
    if "google.colab" in sys.modules:
        raise RuntimeError("Google Drive is not mounted at /content/drive/MyDrive. Mount it before running.")
    return LOCAL_ROOT


PROJECT_ROOT = choose_root()
RUN_NAME = "portability_v5_operating_point" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else "")
ROOT = PROJECT_ROOT / RUN_NAME
DIRS = {k: ROOT / k for k in ["config", "units", "probe", "tables", "figures", "logs", "data"]}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data" / "tsml_ucr"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = DIRS["logs"] / "run.log"


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


# =============================================================================
# Device families
# =============================================================================


@dataclass(frozen=True)
class Family:
    code: str
    name: str
    citation: str
    voltage_mode: str      # "bipolar" or "unipolar"
    v_pos: float
    v_mid: float
    discrete: bool
    volatile: bool
    dt_hint: float
    x_lo: float
    x_hi: float
    params: Dict[str, float]


FAMILIES: Dict[str, Family] = {
    "LD": Family("LD", "Linear ion drift", "Strukov et al., Nature 2008", "bipolar", 0.5, 0.0, False, False,
                 1e-3, 0.0, 1.0, {"R_on": 100.0, "R_off": 1e5, "D": 1e-8, "mu_v": 1e-14, "x0": 0.5}),
    "VT": Family("VT", "VTEAM (fitted SDC)", "Bednarz and Garda, Energies 2021", "bipolar", 0.5, 0.0, False, False,
                 1e-4, 0.0, 1.0, {"R_on": 1357.8, "R_off": 352196.0, "k_on": -997.642, "k_off": 0.2969,
                                  "alpha_on": 2.0, "alpha_off": 1.0, "v_on": -0.340661, "v_off": 0.044669,
                                  "x0": 0.00099943}),
    "VV": Family("VV", "V-VTEAM volatile", "Patni et al., IFETC 2024", "unipolar", 3.0, 1.6583, False, True,
                 1e-5, 0.0, 1.0, {"v_th": 1.744, "v_hold": 1.5726, "R_on": 3e4, "R_off": 15e9, "k": 650.0,
                                  "alpha": 0.09999, "tau": 1.0444, "beta": 2.14262, "x0": 0.0}),
    "ZH": Family("ZH", "Dynamic memristor (Zhong)", "Zhong et al., Nat. Commun. 2021", "bipolar", 3.0, 0.0, True,
                 True, 1.0, 0.0, 1.0, {"r": 0.99, "G0": 0.5, "Kp": 9.13, "Kn": 0.32, "alpha": 0.23, "x0": 0.5}),
    "DF": Family("DF", "Diffusive compact", "Cisternas Ferri et al., Materials 2019", "bipolar", 2.5, 0.0, False,
                 False, 1e-6, 0.0, 1.0, {"alpha_plus": 15.0, "alpha_minus": 15.0, "delta_plus": 0.2,
                                         "delta_minus": 0.2, "R_on": 35.0, "R_off": 9500.0, "v0": 0.3,
                                         "tau0": 0.01, "x0": 0.5, "lambda0": 0.5}),
    "DN": Family("DN", "Volatile dynamic (Daniels)", "Daniels et al., Adv. Electron. Mater. 2026", "unipolar", 0.8,
                 0.5, True, True, 1.0, 0.1, 1.0, {"alpha_i": 1e-8, "beta_i": 0.5, "gamma_i": 1e-5, "delta_i": 4.0,
                                                  "lambda": 1e3, "eta": 8.0, "x_max": 1.0, "x_min": 0.1,
                                                  "V_th": 0.6, "t_pulse": 1e-9, "tau": 10e-9, "x0": 0.1}),
}
FAMILY_CODES = list(FAMILIES.keys())
READ_VOLTAGE = 0.1


@dataclass
class Instance:
    family: str
    role: str
    seed: int
    params: Dict[str, float]

    @property
    def iid(self) -> str:
        return f"{self.family}_{self.role}_s{self.seed}"


def _lognorm(v: float, rng: np.random.Generator) -> float:
    if SIGMA_D2D <= 0 or v == 0:
        return float(v)
    return float(math.copysign(abs(v) * math.exp(SIGMA_D2D * rng.standard_normal()), v))


def make_instance(fam: str, seed: int, role: str) -> Instance:
    p = dict(FAMILIES[fam].params)
    rng = np.random.default_rng(stable_seed("instance", fam, seed, role, MASTER_SEED))
    if fam == "LD":
        for k in ["R_on", "R_off", "mu_v"]:
            p[k] = _lognorm(p[k], rng)
    elif fam == "VT":
        for k in ["R_on", "R_off", "k_on", "k_off", "v_on", "v_off"]:
            p[k] = _lognorm(p[k], rng)
    elif fam == "VV":
        for k in ["R_on", "R_off", "k", "alpha", "tau", "beta", "v_th", "v_hold"]:
            p[k] = _lognorm(p[k], rng)
        p["v_hold"] = min(p["v_hold"], 0.97 * p["v_th"])
    elif fam == "ZH":
        leak = _lognorm(1.0 - p["r"], rng)
        p["r"] = float(np.clip(1.0 - leak, 0.8, 0.9999))
        for k in ["G0", "Kp", "Kn", "alpha"]:
            p[k] = _lognorm(p[k], rng)
        p["G0"] = float(np.clip(p["G0"], 0.05, 0.95))
    elif fam == "DF":
        for k in ["alpha_plus", "alpha_minus", "delta_plus", "delta_minus", "R_on", "R_off", "v0", "tau0"]:
            p[k] = _lognorm(p[k], rng)
    elif fam == "DN":
        for k in ["alpha_i", "beta_i", "gamma_i", "delta_i", "lambda", "eta", "V_th", "tau"]:
            p[k] = _lognorm(p[k], rng)
        p["V_th"] = float(np.clip(p["V_th"], 0.45, 0.75))
    return Instance(fam, role, seed, p)


def nominal_instance(fam: str) -> Instance:
    return Instance(fam, "nominal", -1, dict(FAMILIES[fam].params))


def u_to_voltage(fam: str, u: np.ndarray) -> np.ndarray:
    f = FAMILIES[fam]
    u = np.clip(u, -1.0, 1.0)
    if f.voltage_mode == "bipolar":
        return u * f.v_pos
    return np.where(u <= 0.0, (u + 1.0) * f.v_mid, f.v_mid + u * (f.v_pos - f.v_mid))


def rest_input(fam: str) -> float:
    return -1.0 if FAMILIES[fam].voltage_mode == "unipolar" else 0.0


def init_state(inst: Instance, shape: Tuple[int, ...]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x = np.full(shape, inst.params["x0"], dtype=np.float64)
    aux: Dict[str, np.ndarray] = {}
    if inst.family == "VV":
        aux["age"] = np.zeros(shape, dtype=np.float64)
    if inst.family == "DF":
        aux["lam"] = np.full(shape, inst.params["lambda0"], dtype=np.float64)
    return x, aux


def _vt_solve(target: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Invert G(z) = 0.5 (atanh z + atan z) via A = atanh z, where the map is well conditioned."""
    a = np.arctanh(np.clip(z0, 0.0, 1.0 - 1e-12))
    for _ in range(8):
        th = np.tanh(a)
        h = 0.5 * a + 0.5 * np.arctan(th) - target
        dh = 0.5 + 0.5 * (1.0 - th * th) / (1.0 + th * th)
        a = np.maximum(a - h / dh, 0.0)
    return np.clip(np.tanh(a), 0.0, 1.0 - 1e-12)


def advance(inst: Instance, x: np.ndarray, aux: Dict[str, np.ndarray], v: np.ndarray, dur: Any) -> np.ndarray:
    """Advance the state under a constant voltage for duration `dur` (seconds, or update ticks for discrete
    models). All updates are exact or closed-form for a constant voltage."""
    p = inst.params
    fam = inst.family
    if fam == "LD":
        ron, roff = p["R_on"], p["R_off"]
        c = p["mu_v"] * ron / (p["D"] ** 2)
        a = 0.5 * (ron - roff)
        b = roff
        f0 = b * x + a * x * x
        ft = np.clip(f0 + c * v * dur, 0.0, b + a)
        xn = (-b + np.sqrt(np.maximum(b * b + 4.0 * a * ft, 0.0))) / (2.0 * a)
        return np.clip(xn, 0.0, 1.0)
    if fam == "VT":
        pos = v >= p["v_off"]
        neg = v <= p["v_on"]
        cpos = np.where(pos, p["k_off"] * np.maximum(v / p["v_off"] - 1.0, 0.0) ** p["alpha_off"], 0.0)
        cneg = np.where(neg, abs(p["k_on"]) * np.maximum(v / p["v_on"] - 1.0, 0.0) ** p["alpha_on"], 0.0)
        z0 = np.where(neg, 1.0 - x, x)
        z0c = np.clip(z0, 0.0, 1.0 - 1e-12)
        c = cpos + cneg
        target = 0.5 * (np.arctanh(z0c) + np.arctan(z0c)) + c * dur
        z = np.where(c > 0, _vt_solve(target, z0c), z0)
        return np.clip(np.where(neg, 1.0 - z, z), 0.0, 1.0)
    if fam == "VV":
        age = aux["age"]
        write = v >= p["v_th"]
        decay = v <= p["v_hold"]
        dxdt = p["k"] * np.maximum(v / p["v_th"] - 1.0, 0.0) ** p["alpha"]
        tau, beta = max(p["tau"], EPS), p["beta"]
        t1 = age + dur
        inc = (t1 / tau) ** beta - (age / tau) ** beta
        xn = np.where(write, x + dur * dxdt, np.where(decay, x * np.exp(-inc), x))
        aux["age"] = np.where(decay, t1, 0.0)
        return np.clip(xn, 0.0, 1.0)
    if fam == "ZH":
        av = p["alpha"] * np.abs(v)
        U = av / (av + 1.0)
        B = (v > 0.0).astype(np.float64)
        a = p["r"] - U
        c = (1.0 - p["r"]) * p["G0"] + U * B
        n = np.asarray(dur, dtype=np.float64)
        an = np.power(a, n)
        xn = an * x + c * (1.0 - an) / (1.0 - a)
        return np.clip(xn, 0.0, 1.0)
    if fam == "DF":
        lam = aux["lam"]
        gp = 1.0 / (1.0 + np.exp(-p["alpha_plus"] * (v - p["delta_plus"])))
        gm = 1.0 / (1.0 + np.exp(-p["alpha_minus"] * (v + p["delta_minus"])))
        lam = np.clip(np.minimum(gm, np.maximum(lam, gp)), 0.0, 1.0)
        aux["lam"] = lam
        tau = np.maximum(p["tau0"] * np.exp(-np.abs(v) / max(p["v0"], EPS)), EPS)
        xn = lam + (x - lam) * np.exp(-dur / tau)
        return np.clip(xn, 0.0, 1.0)
    if fam == "DN":
        xmin, xmax = p["x_min"], p["x_max"]
        n = np.asarray(dur, dtype=np.float64)
        write = v > p["V_th"]
        inc = p["t_pulse"] * p["lambda"] * np.sinh(p["eta"] * np.maximum(v, 0.0))
        # Exact solution of dx/dn = inc (1 - exp(3 (x - x_max))), the continuum limit of the published pulse
        # map (per-pulse increment below 4e-4, so the step error is negligible; verified in the self-test).
        s0 = np.clip(np.exp(3.0 * (x - xmax)), 1e-300, 1.0 - 1e-15)
        K = np.log(s0) - np.log1p(-s0) + 3.0 * inc * n
        s = 1.0 / (1.0 + np.exp(-np.clip(K, -700.0, 700.0)))
        xw = xmax + np.log(np.maximum(s, 1e-300)) / 3.0
        xl = xmin + (x - xmin) * np.exp(-n * p["t_pulse"] / max(p["tau"], EPS))
        return np.clip(np.where(write, xw, xl), xmin, xmax)
    raise ValueError(fam)


def observable(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Measurable output: log10 of the read conductance at a small read voltage."""
    p = inst.params
    fam = inst.family
    if fam == "LD":
        r = p["R_on"] * x + p["R_off"] * (1.0 - x)
        return -np.log10(r)
    if fam == "VT":
        r = p["R_on"] + (p["R_off"] - p["R_on"]) * x
        return -np.log10(r)
    if fam == "VV":
        r = p["R_off"] + (p["R_on"] - p["R_off"]) * x
        return -np.log10(np.maximum(r, 1.0))
    if fam == "ZH":
        return np.log10(np.maximum(x, 1e-4))
    if fam == "DF":
        r = p["R_on"] * x + p["R_off"] * (1.0 - x)
        return -np.log10(r)
    if fam == "DN":
        vr = READ_VOLTAGE
        i = (1.0 - x) * p["alpha_i"] * (1.0 - np.exp(-p["beta_i"] * vr)) + x * p["gamma_i"] * np.sinh(p["delta_i"] * vr)
        return np.log10(np.maximum(np.abs(i) / vr, 1e-30))
    raise ValueError(fam)


def norm_state(fam: str, x: np.ndarray) -> np.ndarray:
    f = FAMILIES[fam]
    return (x - f.x_lo) / (f.x_hi - f.x_lo)


# =============================================================================
# Task-blind interface calibration (datasheet quantities from nominal parameters)
# =============================================================================


def calibrate_switch_time(fam: str, max_steps: int = 300_000) -> float:
    inst = nominal_instance(fam)
    x, aux = init_state(inst, (1, 1))
    v = u_to_voltage(fam, np.ones((1, 1)))
    x0 = float(norm_state(fam, x)[0, 0])
    dt = FAMILIES[fam].dt_hint
    for step in range(1, max_steps + 1):
        x = advance(inst, x, aux, v, dt)
        if abs(float(norm_state(fam, x)[0, 0]) - x0) >= 0.35:
            return float(step if FAMILIES[fam].discrete else step * dt)
    raise RuntimeError(f"{fam} did not reach the 0.35 calibration excursion.")


def recovery_duration(fam: str) -> float:
    p = FAMILIES[fam].params
    q = RETENTION_PER_SYMBOL
    if fam == "VV":
        return p["tau"] * (-math.log(q)) ** (1.0 / p["beta"])
    if fam == "ZH":
        return float(max(1, round(math.log(q) / math.log(p["r"]))))
    if fam == "DN":
        return float(max(1, round(math.log(q) / (-p["t_pulse"] / p["tau"]))))
    return 0.0


T_SWITCH: Dict[str, float] = {}
T_RECOVER: Dict[str, float] = {}
OP_GAIN_IDX = np.repeat(np.arange(len(GAINS)), len(FRACS))
OP_FRAC_IDX = np.tile(np.arange(len(FRACS)), len(GAINS))
N_OPS = len(OP_GAIN_IDX)


def op_duration(fam: str, op: int) -> float:
    d = FRACS[OP_FRAC_IDX[op]] * T_SWITCH[fam]
    if FAMILIES[fam].discrete:
        return float(max(1, int(round(d))))
    return float(d)


def op_key(fam: str, op: int) -> Tuple[int, float]:
    return int(OP_GAIN_IDX[op]), float(np.round(op_duration(fam, op), 15))


# =============================================================================
# Simulation
# =============================================================================


def bin_edges(L: int) -> np.ndarray:
    return np.linspace(0, L, N_BINS + 1).round().astype(int)


def features_from_traj(traj: np.ndarray) -> np.ndarray:
    """traj (n, L) -> (n, 2 * N_BINS): bin means and bin standard deviations."""
    traj = np.asarray(traj, dtype=np.float64)
    e = bin_edges(traj.shape[1])
    cnt = np.diff(e).astype(np.float64)
    s = np.add.reduceat(traj, e[:-1], axis=1)
    q = np.add.reduceat(traj * traj, e[:-1], axis=1)
    mu = s / cnt
    sd = np.sqrt(np.maximum(q / cnt - mu * mu, 0.0))
    return np.concatenate([mu, sd], axis=1).astype(np.float32)


def simulate(inst: Instance, U: np.ndarray, ops: Sequence[int], mode: str, key: Tuple[Any, ...]) -> np.ndarray:
    """Drive the device with each row of U (one independent sequence per row) at each operating point.

    mode "features": returns (len(ops), n, 2*N_BINS) float32
    mode "traj":     returns (n, L, len(ops)) float32 trajectories of the noisy observable

    Noise streams are keyed by the instance's effective physical configuration, so the same operating point
    produces identical noise regardless of which other operating points are simulated alongside it.
    """
    fam = inst.family
    U = np.asarray(U, dtype=np.float64)
    n, L = U.shape
    ops = list(ops)
    keys = [op_key(fam, o) for o in ops]
    uniq = sorted(set(keys))
    col_of = {k: i for i, k in enumerate(uniq)}
    m = len(uniq)
    gains = np.array([GAINS[k[0]] for k in uniq])[None, :]
    durs = np.array([k[1] for k in uniq])[None, :]
    rngs = [np.random.default_rng(stable_seed(*key, inst.iid, "cfg", k[0], repr(k[1]))) for k in uniq]
    x, aux = init_state(inst, (n, m))
    v_rest = float(u_to_voltage(fam, np.array(rest_input(fam))))
    rec = T_RECOVER[fam]
    lo, hi = FAMILIES[fam].x_lo, FAMILIES[fam].x_hi
    span = hi - lo
    if fam == "DN":
        lo, hi = inst.params["x_min"], inst.params["x_max"]
    nc = np.empty((n, m), dtype=np.float32)
    nr = np.empty((n, m), dtype=np.float32)
    if mode == "features":
        e = bin_edges(L)
        bin_of = np.searchsorted(e, np.arange(L), side="right") - 1
        S = np.zeros((n, m, N_BINS))
        Q = np.zeros((n, m, N_BINS))
    else:
        out = np.empty((n, L, m), dtype=np.float32)
    for t in range(L):
        v = u_to_voltage(fam, gains * U[:, t:t + 1])
        x = advance(inst, x, aux, v, durs)
        if SIGMA_C2C > 0 or SIGMA_READ > 0:
            for j, r in enumerate(rngs):
                nc[:, j] = r.standard_normal(n, dtype=np.float32)
                nr[:, j] = r.standard_normal(n, dtype=np.float32)
        if SIGMA_C2C > 0:
            x = np.clip(x + SIGMA_C2C * span * nc, lo, hi)
        o = observable(inst, x)
        if SIGMA_READ > 0:
            o = o + SIGMA_READ * nr
        if mode == "features":
            b = bin_of[t]
            S[:, :, b] += o
            Q[:, :, b] += o * o
        else:
            out[:, t, :] = o
        if rec > 0:
            x = advance(inst, x, aux, np.full_like(x, v_rest), rec)
    idx = [col_of[k] for k in keys]
    if mode == "features":
        cnt = np.diff(bin_edges(L)).astype(np.float64)
        mu = S / cnt
        sd = np.sqrt(np.maximum(Q / cnt - mu * mu, 0.0))
        feats = np.concatenate([mu, sd], axis=2).astype(np.float32)   # (n, m, F)
        return np.transpose(feats, (1, 0, 2))[idx]
    return out[:, :, idx]


def rest_observable(inst: Instance) -> float:
    x, _ = init_state(inst, (1,))
    return float(observable(inst, x)[0])


# =============================================================================
# Self-tests of the closed-form updates
# =============================================================================


def self_tests() -> pd.DataFrame:
    rows = []
    # Daniels: closed form versus the published per-pulse map.
    inst = nominal_instance("DN")
    p = inst.params
    for v in [0.62, 0.7, 0.8]:
        x_loop = p["x0"]
        inc = p["t_pulse"] * p["lambda"] * math.sinh(p["eta"] * v)
        for _ in range(2000):
            x_loop = min(max(x_loop + inc * (1 - math.exp(3 * x_loop) / math.exp(3 * p["x_max"])), p["x_min"]), p["x_max"])
        x_cf = float(advance(inst, np.array([p["x0"]]), {}, np.array([v]), 2000.0)[0])
        rows.append({"test": f"DN closed form vs pulse map, V={v}", "abs_error": abs(x_cf - x_loop)})
    # VTEAM: exact inverse versus fine RK4.
    inst = nominal_instance("VT")
    for v, dur in [(0.5, 0.05), (-0.5, 0.002), (-0.4, 0.01)]:
        x_cf = float(advance(inst, np.array([0.3]), {}, np.array([v]), dur)[0])
        x = 0.3
        pp = inst.params
        def f(z):
            if v >= pp["v_off"]:
                return pp["k_off"] * (v / pp["v_off"] - 1) ** pp["alpha_off"] * max(1 - z ** 4, 0)
            if v <= pp["v_on"]:
                return pp["k_on"] * (v / pp["v_on"] - 1) ** pp["alpha_on"] * max(1 - (z - 1) ** 4, 0)
            return 0.0
        nsub = 20000
        h = dur / nsub
        for _ in range(nsub):
            k1 = f(x); k2 = f(min(max(x + 0.5 * h * k1, 0), 1)); k3 = f(min(max(x + 0.5 * h * k2, 0), 1))
            k4 = f(min(max(x + h * k3, 0), 1))
            x = min(max(x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0), 1)
        rows.append({"test": f"VT exact vs RK4, V={v}, dt={dur}", "abs_error": abs(x_cf - x)})
    # Zhong: closed-form affine iterate versus explicit iteration.
    inst = nominal_instance("ZH")
    pp = inst.params
    x = 0.5
    for _ in range(7):
        U_ = pp["alpha"] * 2.0 / (pp["alpha"] * 2.0 + 1)
        x = pp["r"] * x + (1 - pp["r"]) * pp["G0"] + U_ * (1 - x)
    x_cf = float(advance(inst, np.array([0.5]), {}, np.array([2.0]), 7.0)[0])
    rows.append({"test": "ZH closed form vs iteration, 7 ticks", "abs_error": abs(x_cf - x)})
    return pd.DataFrame(rows)


# =============================================================================
# Data
# =============================================================================


@dataclass
class Dataset:
    name: str
    tier: str
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    U_train: np.ndarray
    U_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int


def ensure_aeon() -> None:
    try:
        importlib.import_module("aeon")
    except ImportError:
        log("Installing aeon ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "aeon>=1.0"])


def robust_scale(Xtr: np.ndarray, X: np.ndarray) -> np.ndarray:
    c = float(np.median(Xtr))
    s = float(np.quantile(np.abs(Xtr - c), 0.995))
    if not np.isfinite(s) or s <= EPS:
        s = float(np.std(Xtr)) or 1.0
    return np.clip((X - c) / s, -1.0, 1.0)


def synthetic_dataset(name: str, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    L = 96
    t = np.linspace(0, 1, L)
    def make(n):
        y = rng.integers(0, 3, n)
        X = np.empty((n, L))
        for i, c in enumerate(y):
            ph = rng.uniform(0, 2 * np.pi)
            base = [np.sin(2 * np.pi * 3 * t + ph), np.sign(np.sin(2 * np.pi * 3 * t + ph)),
                    np.sin(2 * np.pi * 6 * t + ph) * t][c]
            X[i] = base + 0.3 * rng.standard_normal(L)
        return X, y
    Xtr, ytr = make(60)
    Xte, yte = make(60)
    return Dataset(name, "smoke", Xtr, Xte, robust_scale(Xtr, Xtr), robust_scale(Xtr, Xte), ytr, yte, 3)


def load_dataset(name: str) -> Dataset:
    ensure_aeon()
    from aeon.datasets import load_classification
    Xtr, ytr = load_classification(name=name, split="train", extract_path=str(DATA_DIR))
    Xte, yte = load_classification(name=name, split="test", extract_path=str(DATA_DIR))
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    if Xtr.ndim != 3 or Xtr.shape[1] != 1:
        raise RuntimeError(f"{name}: expected univariate equal-length data, got {Xtr.shape}")
    Xtr, Xte = Xtr[:, 0, :], Xte[:, 0, :]
    if not (np.all(np.isfinite(Xtr)) and np.all(np.isfinite(Xte))):
        raise RuntimeError(f"{name}: non-finite values")
    classes = np.unique(np.asarray(ytr).astype(str))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([cmap[v] for v in np.asarray(ytr).astype(str)])
    yte_i = np.array([cmap[v] for v in np.asarray(yte).astype(str)])
    tier = "prior" if name in PRIOR_DATASETS else "fresh"
    return Dataset(name, tier, Xtr, Xte, robust_scale(Xtr, Xtr), robust_scale(Xtr, Xte), ytr_i, yte_i, len(classes))


# =============================================================================
# Readout, metrics
# =============================================================================


def balanced_accuracy(y: np.ndarray, yp: np.ndarray, C: int) -> float:
    rec = []
    for c in range(C):
        mask = y == c
        if mask.any():
            rec.append(float(np.mean(yp[mask] == c)))
    return float(np.mean(rec))


def nba(ba: float, C: int) -> float:
    return (ba - 1.0 / C) / (1.0 - 1.0 / C)


def std_stats(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = F.mean(axis=0).astype(np.float64)
    sd = F.std(axis=0).astype(np.float64)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def class_weights(y: np.ndarray, C: int) -> np.ndarray:
    counts = np.bincount(y, minlength=C).astype(np.float64)
    present = int((counts > 0).sum())
    return len(y) / (present * counts[y])


def _gram(Z: np.ndarray, y: np.ndarray, C: int):
    w = class_weights(y, C)
    Y = -np.ones((len(y), C))
    Y[np.arange(len(y)), y] = 1.0
    ws = w.sum()
    mz = (w @ Z) / ws
    my = (w @ Y) / ws
    Zc = Z - mz
    Zw = Zc * w[:, None]
    return Zw.T @ Zc, Zw.T @ (Y - my), mz, my, ws


@dataclass
class Readout:
    W: np.ndarray
    b: np.ndarray

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.argmax(Z @ self.W + self.b, axis=1)


def ridge_fit(Z: np.ndarray, y: np.ndarray, C: int, alpha_abs: float,
              prior: Optional[np.ndarray] = None) -> Readout:
    A, B, mz, my, _ = _gram(np.asarray(Z, dtype=np.float64), y, C)
    reg = alpha_abs * np.eye(A.shape[0])
    rhs = B if prior is None else B + alpha_abs * prior
    try:
        W = np.linalg.solve(A + reg, rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(A + reg, rhs, rcond=None)[0]
    return Readout(W, my - mz @ W)


def ridge_multi_alpha(Z: np.ndarray, y: np.ndarray, C: int, alphas_rel: np.ndarray) -> List[Readout]:
    A, B, mz, my, ws = _gram(np.asarray(Z, dtype=np.float64), y, C)
    e, V = np.linalg.eigh(A)
    VB = V.T @ B
    out = []
    for a in alphas_rel:
        W = V @ (VB / (e + a * ws)[:, None])
        out.append(Readout(W, my - mz @ W))
    return out


def make_folds(y: np.ndarray, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    min_count = int(np.bincount(y).min())
    k = int(min(3, max(2, min_count)))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed % (2**31))
    return list(skf.split(np.zeros(len(y)), y))


def select_and_test(Ftr: np.ndarray, Fte: np.ndarray, ytr: np.ndarray, yte: np.ndarray, C: int,
                    folds) -> Tuple[float, int, float]:
    """CV over the relative ridge grid; returns (cv_ba, alpha_index, test_ba)."""
    scores = np.zeros(len(ALPHA_REL))
    for tr, va in folds:
        mu, sd = std_stats(Ftr[tr])
        Ztr = (Ftr[tr] - mu) / sd
        Zva = (Ftr[va] - mu) / sd
        for i, ro in enumerate(ridge_multi_alpha(Ztr, ytr[tr], C, ALPHA_REL)):
            scores[i] += balanced_accuracy(ytr[va], ro.predict(Zva), C)
    scores /= len(folds)
    ai = int(np.argmax(scores))
    mu, sd = std_stats(Ftr)
    ro = ridge_multi_alpha((Ftr - mu) / sd, ytr, C, ALPHA_REL[ai:ai + 1])[0]
    return float(scores[ai]), ai, balanced_accuracy(yte, ro.predict((Fte - mu) / sd), C)


# =============================================================================
# Alignment maps
# =============================================================================


def lag_design(r: np.ndarray, taps: int, pad: float, quadratic: bool) -> np.ndarray:
    """r (n, L) -> (n*L, P) causal lag design, padded with the device rest value."""
    n, L = r.shape
    rp = np.concatenate([np.full((n, taps - 1), pad, dtype=np.float64), r.astype(np.float64)], axis=1)
    cols = [rp[:, taps - 1 - k: taps - 1 - k + L] for k in range(taps)]
    X = np.stack(cols, axis=2).reshape(n * L, taps)
    if quadratic:
        X = np.concatenate([X, X * X], axis=1)
    return X


@dataclass
class LagMap:
    taps: int
    quadratic: bool
    pad: float
    mx: np.ndarray
    sx: np.ndarray
    beta: np.ndarray
    my: float

    def apply(self, r: np.ndarray) -> np.ndarray:
        n, L = r.shape
        out = np.full((n, L), self.my, dtype=np.float64)
        rp = np.concatenate([np.full((n, self.taps - 1), self.pad), r.astype(np.float64)], axis=1)
        P = self.taps
        for k in range(self.taps):
            lagk = rp[:, P - 1 - k: P - 1 - k + L]
            out += self.beta[k] * (lagk - self.mx[k]) / self.sx[k]
            if self.quadratic:
                out += self.beta[P + k] * (lagk * lagk - self.mx[P + k]) / self.sx[P + k]
        return out.astype(np.float32)


def fit_lag_map(r_src: np.ndarray, r_tgt: np.ndarray, pad: float, taps: int, quadratic: bool) -> LagMap:
    X = lag_design(r_tgt, taps, pad, quadratic)
    y = r_src.reshape(-1).astype(np.float64)
    mx = X.mean(0)
    sx = X.std(0)
    sx[sx < 1e-12] = 1.0
    Xs = (X - mx) / sx
    my = float(y.mean())
    A = Xs.T @ Xs
    lam = 1e-3 * np.trace(A) / A.shape[0] + 1e-12
    beta = np.linalg.solve(A + lam * np.eye(A.shape[0]), Xs.T @ (y - my))
    return LagMap(taps, quadratic, pad, mx, sx, beta, my)


def r2_score_flat(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    ss = np.sum((y - y.mean()) ** 2)
    if ss < 1e-20:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / ss)


def coral_matrix(Zs: np.ndarray, Zt: np.ndarray) -> np.ndarray:
    def msqrt(C, inv):
        e, V = np.linalg.eigh(C)
        e = np.maximum(e, 1e-10)
        return (V * (e ** (-0.5 if inv else 0.5))) @ V.T
    Cs = np.cov(Zs, rowvar=False) + CORAL_EPS * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + CORAL_EPS * np.eye(Zt.shape[1])
    return msqrt(Ct, True) @ msqrt(Cs, False)


# =============================================================================
# Capacity
# =============================================================================


def legendre(k: int, u: np.ndarray) -> np.ndarray:
    if k == 1:
        return u
    if k == 2:
        return 0.5 * (3 * u * u - 1)
    if k == 3:
        return 0.5 * (5 * u ** 3 - 3 * u)
    raise ValueError(k)


def capacity_targets(Uprobe: np.ndarray, t0: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (targets (rows, n_targets), degree per target) on rows t >= t0 of every episode."""
    E, L = Uprobe.shape
    def lagged(d):
        return Uprobe[:, t0 - d: L - d].reshape(-1)
    cols, deg = [], []
    for d in range(CAP_MAX_DELAY_D1 + 1):
        cols.append(lagged(d)); deg.append(1)
    D2 = CAP_MAX_DELAY_D2
    for i in range(D2 + 1):
        cols.append(legendre(2, lagged(i))); deg.append(2)
        for j in range(i + 1, D2 + 1):
            cols.append(lagged(i) * lagged(j)); deg.append(2)
    D3 = CAP_MAX_DELAY_D3
    for i in range(D3 + 1):
        cols.append(legendre(3, lagged(i))); deg.append(3)
        for j in range(D3 + 1):
            if j != i:
                cols.append(legendre(2, lagged(i)) * lagged(j)); deg.append(3)
        for j in range(i + 1, D3 + 1):
            for k in range(j + 1, D3 + 1):
                cols.append(lagged(i) * lagged(j) * lagged(k)); deg.append(3)
    Y = np.stack(cols, axis=1)
    return Y - Y.mean(0), np.array(deg)


def capacities(state_signal: np.ndarray, Yc: np.ndarray, deg: np.ndarray, t0: int) -> Dict[str, float]:
    E, L = state_signal.shape
    X = np.stack([state_signal[:, t0 - k: L - k].reshape(-1) for k in range(CAP_STATE_LAGS)], axis=1)
    X = X - X.mean(0)
    sc = X.std(0)
    keep = sc > 1e-12
    if not keep.any():
        return {"C1": 0.0, "C2": 0.0, "C3": 0.0, "C_total": 0.0, "linear_fraction": float("nan")}
    Q, R = np.linalg.qr(X[:, keep] / sc[keep])
    diag = np.abs(np.diag(R))
    Q = Q[:, diag > 1e-8 * diag.max()]
    proj = Q.T @ Yc
    cap = np.sum(proj * proj, axis=0) / np.sum(Yc * Yc, axis=0)
    cap[cap < max(CAP_THRESHOLD, 4.0 * (CAP_STATE_LAGS + 1) / X.shape[0])] = 0.0
    c1, c2, c3 = (float(cap[deg == d].sum()) for d in (1, 2, 3))
    tot = c1 + c2 + c3
    return {"C1": c1, "C2": c2, "C3": c3, "C_total": tot,
            "linear_fraction": c1 / tot if tot > 1e-9 else float("nan")}


# =============================================================================
# Statistics helpers
# =============================================================================


def wilcoxon_less(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """One-sided p for median(x - y) < 0."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.all(np.abs(d) < 1e-12):
        return float("nan"), len(d)
    return float(wilcoxon(d, alternative="less", zero_method="wilcox").pvalue), len(d)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, v in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def boot_median_ci(x: np.ndarray, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(N_BOOT, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def hier_boot_ci(df: pd.DataFrame, col: str, seed: int) -> Tuple[float, float, float]:
    """Resample datasets, then pairs within datasets; statistic is the median of dataset medians."""
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby("dataset")]
    groups = [g for g in groups if len(g)]
    if not groups:
        return (float("nan"),) * 3
    obs = float(np.median([np.median(g) for g in groups]))
    rng = np.random.default_rng(seed)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        stats[b] = np.median([np.median(rng.choice(groups[i], len(groups[i]))) for i in pick])
    return obs, float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


# =============================================================================
# Figure helpers (all spines, labelled bounds at tick multiples, outlined legends)
# =============================================================================

SPINE_W = 1.6
LEGEND_W = 0.8
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "axes.linewidth": SPINE_W, "xtick.major.width": SPINE_W,
    "ytick.major.width": SPINE_W, "savefig.dpi": 600, "savefig.bbox": "tight", "figure.dpi": 110,
})
FAM_COLORS = {"LD": "#1b6ca8", "VT": "#e08a1e", "VV": "#2a9d5c", "ZH": "#c0392b", "DF": "#7b4fa0", "DN": "#8c6d3f"}


def _nice_step(span: float, n: int = 5) -> float:
    span = max(span, 1e-12)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= raw:
            return mult * mag
    return 10 * mag


def _decimals(step: float) -> int:
    d = 0
    while d < 6 and abs(round(step * 10 ** d) - step * 10 ** d) > 1e-6:
        d += 1
    return d


def set_bounds(ax, axis: str, lo: float, hi: float, step: Optional[float] = None, n: int = 5) -> None:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi - lo < 1e-12:
        hi = lo + 1.0
    step = step or _nice_step(hi - lo, n)
    a = math.floor(lo / step + 1e-9) * step
    b = math.ceil(hi / step - 1e-9) * step
    if b <= a:
        b = a + step
    ticks = np.arange(a, b + step / 2, step)
    d = _decimals(step)
    fmt = FuncFormatter(lambda v, _: (f"{v:.{d}f}" if abs(v) > 10 ** (-d - 1) else f"{0:.{d}f}"))
    if axis == "y":
        ax.set_ylim(a, b); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(fmt)
    else:
        ax.set_xlim(a, b); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(fmt)


def style_ax(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(SPINE_W)


def outlined_legend(ax, **kw):
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, **kw)
    leg.get_frame().set_linewidth(LEGEND_W)
    return leg


def panel_letters(fig, axes: Sequence[Any]) -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for i, ax in enumerate(axes):
        bb = ax.yaxis.label.get_window_extent(rend)
        x0 = inv.transform((bb.x0, 0))[0]
        top = inv.transform((0, ax.get_window_extent(rend).y1))[1]
        fig.text(x0, top + 0.015, chr(ord("a") + i), fontsize=20, fontweight="bold", ha="left", va="bottom")


def save_fig(fig, stem: str) -> None:
    fig.savefig(DIRS["figures"] / f"{stem}.png")
    fig.savefig(DIRS["figures"] / f"{stem}.pdf")
    plt.close(fig)





# =============================================================================
# v4: preregistered confirmatory study and reviewer-question battery
# =============================================================================

from contextlib import contextmanager
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

V4_SEEDS = [44, 55, 66] if not SMOKE else [44]
ROBUST_SEED = V4_SEEDS[0]
HELD_OUT_DATASETS = ["MoteStrain", "SonyAIBORobotSurface1", "Coffee", "Meat", "Lightning7",
                     "MedicalImages", "ProximalPhalanxOutlineCorrect", "Ham", "OSULeaf", "Fish"]
V4_D2D = [0.05, 0.20, 0.50] if not SMOKE else [0.05, 0.50]
BUDGETS = [8, 16, 32, 64, 128] if not SMOKE else [8, 16]
K_V4 = [1, 5] if not SMOKE else [1, 2]
K_MAIN = K_V4[-1]
RFF_DIM = 512 if not SMOKE else 128
N_ALT = 256 if not SMOKE else 48
MAP_ALPHAS_LIN = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MAP_ALPHAS_RFF = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LR_CS = [0.1, 1.0, 10.0]
NOMINAL_NOISE = (SIGMA_C2C, SIGMA_READ)
NOISE_LEVELS = {"noise_none": (0.0, 0.0), "noise_high": (3.0 * 0.002, 3.0 * 0.01)}
FEATURE_VARIANTS = [("bins12", 12, "ms"), ("bins48", 48, "ms"), ("means24", 24, "m")]
ROBUST_ORDER = [("noise_none", "No noise"), ("main", "Nominal"), ("noise_high", "High noise"), ("bins12", "12 bins"),
                ("bins48", "48 bins"), ("means24", "Means only"), ("logistic", "Logistic")]
CAP_T0 = max(CAP_MAX_DELAY_D1, CAP_STATE_LAGS)
V3_NAME = "portability_v3_diagnostics" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else "")

LADDER = [("zero_shot", "Zero\nshot"), ("moment", "Moment\nmatching"), ("sign_moment", "Sign\ncorrected"),
          ("diag_paired", "Diagonal\npaired"), ("paired_linear", "Paired\nlinear"), ("paired_rff", "Paired\nnonlinear"),
          ("distill_rff", "Score\ndistillation"), ("pseudo_label", "Pseudo\nlabels"), ("twin_paired_rff", "Twin\nnonlinear")]


@contextmanager
def noise_level(c2c: float, rd: float):
    global SIGMA_C2C, SIGMA_READ
    old = (SIGMA_C2C, SIGMA_READ)
    SIGMA_C2C, SIGMA_READ = c2c, rd
    try:
        yield
    finally:
        SIGMA_C2C, SIGMA_READ = old


def probe_inputs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("probe_input", seed, MASTER_SEED))
    return rng.uniform(-1.0, 1.0, size=(PROBE_EPISODES, PROBE_LENGTH))


def white_inputs(n: int, L: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, L))


def matched_inputs(U: np.ndarray, n: int, L: int, seed: int) -> np.ndarray:
    """Gaussian AR(1) sequences matched to the lag-1 autocorrelation of U, quantile-mapped to the marginal of U."""
    a, b = U[:, :-1].reshape(-1), U[:, 1:].reshape(-1)
    rho = float(np.clip(np.corrcoef(a, b)[0, 1], -0.999, 0.999)) if np.std(a) > 0 else 0.0
    rng = np.random.default_rng(seed)
    g = np.empty((n, L))
    g[:, 0] = rng.standard_normal(n)
    s = math.sqrt(max(1.0 - rho * rho, 1e-6))
    for t in range(1, L):
        g[:, t] = rho * g[:, t - 1] + s * rng.standard_normal(n)
    q = np.sort(U.reshape(-1))
    ranks = np.argsort(np.argsort(g.reshape(-1)))
    pos = (ranks + 0.5) / len(ranks) * (len(q) - 1)
    return q[pos.round().astype(int)].reshape(n, L)


def generic_inputs(pool: Dict[str, np.ndarray], exclude: str, n: int, L: int, seed: int) -> np.ndarray:
    """Unlabeled sequences from other datasets, linearly resampled to length L."""
    names = [k for k in pool if k != exclude]
    rng = np.random.default_rng(seed)
    out = np.empty((n, L))
    grid = np.linspace(0, 1, L)
    for i in range(n):
        U = pool[names[rng.integers(len(names))]]
        row = U[rng.integers(len(U))]
        out[i] = np.interp(grid, np.linspace(0, 1, len(row)), row)
    return out


def features_variant(traj: np.ndarray, nb: int, stats: str) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float64)
    e = np.linspace(0, traj.shape[1], nb + 1).round().astype(int)
    cnt = np.diff(e).astype(np.float64)
    mu = np.add.reduceat(traj, e[:-1], axis=1) / cnt
    if stats == "m":
        return mu.astype(np.float32)
    q = np.add.reduceat(traj * traj, e[:-1], axis=1)
    sd = np.sqrt(np.maximum(q / cnt - mu * mu, 0.0))
    return np.concatenate([mu, sd], axis=1).astype(np.float32)


def column_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    sa, sb = A.std(0), B.std(0)
    r = ((A - A.mean(0)) * (B - B.mean(0))).mean(0) / np.maximum(sa * sb, 1e-24)
    r[(sa < 1e-9) | (sb < 1e-9)] = np.nan
    return r


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 1e-20 else float("nan")


def feature_r2(Y: np.ndarray, Yhat: np.ndarray) -> float:
    ss = np.sum((Y - Y.mean(0)) ** 2)
    return float(1.0 - np.sum((Y - Yhat) ** 2) / ss) if ss > 1e-20 else float("nan")


def make_instance_sigma(fam: str, seed: int, role: str, sigma: float) -> Instance:
    global SIGMA_D2D
    old = SIGMA_D2D
    SIGMA_D2D = sigma
    try:
        return make_instance(fam, seed, role)
    finally:
        SIGMA_D2D = old


def parameter_distance(a: Instance, b: Instance) -> float:
    if a.family != b.family:
        return float("nan")
    vals = []
    for k, va in a.params.items():
        vb = b.params.get(k)
        if vb is None or va == 0 or vb == 0 or np.sign(va) != np.sign(vb) or k in ("x0", "lambda0"):
            continue
        vals.append(math.log(abs(vb) / abs(va)))
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0


def _ridge_path(X: np.ndarray, Y: np.ndarray, lams: Sequence[float]) -> List[np.ndarray]:
    """Ridge solutions for several penalties from one eigendecomposition (primal or dual, whichever is smaller)."""
    n, p = X.shape
    if n >= p:
        e, V = np.linalg.eigh(X.T @ X)
        VB = V.T @ (X.T @ Y)
        return [V @ (VB / (e + lam)[:, None]) for lam in lams]
    e, V = np.linalg.eigh(X @ X.T)
    VY = V.T @ Y
    return [X.T @ (V @ (VY / (e + lam)[:, None])) for lam in lams]


def fit_feature_map(Zt: np.ndarray, Zs: np.ndarray, kind: str, seed: int):
    """Ridge map from target features (or their random Fourier features) to source features or scores.
    Penalty chosen by 3-fold CV mean squared error. Returns a callable."""
    rng = np.random.default_rng(seed)
    Zt = np.asarray(Zt, dtype=np.float64)
    if kind == "rff":
        d = Zt.shape[1]
        Wr = rng.standard_normal((d, RFF_DIM)) / math.sqrt(d)
        br = rng.uniform(0, 2 * math.pi, RFF_DIM)
        phi = lambda Z: math.sqrt(2.0 / RFF_DIM) * np.cos(np.asarray(Z, dtype=np.float64) @ Wr + br)
        alphas = MAP_ALPHAS_RFF
    else:
        phi = lambda Z: np.asarray(Z, dtype=np.float64)
        alphas = MAP_ALPHAS_LIN
    X = phi(Zt)
    Y = np.asarray(Zs, dtype=np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    n = len(X)
    if n >= 6:
        fold = rng.permutation(n) % 3
        err = np.zeros(len(alphas))
        for f in range(3):
            tr, va = fold != f, fold == f
            for i, W in enumerate(_ridge_path(Xc[tr], Yc[tr], [a * tr.sum() for a in alphas])):
                err[i] += np.sum((Yc[va] - Xc[va] @ W) ** 2)
        a = alphas[int(np.argmin(err))]
    else:
        a = alphas[len(alphas) // 2]
    W = _ridge_path(Xc, Yc, [a * n])[0]
    return lambda Z: (phi(Z) - mx) @ W + my


def pseudo_label_ba(Ztr: np.ndarray, Zte: np.ndarray, yp: np.ndarray, yte: np.ndarray, C: int, seed: int) -> float:
    """Target readout trained on the source device's predicted labels; penalty chosen by agreement with them."""
    present = np.unique(yp)
    if len(present) < 2:
        return balanced_accuracy(yte, np.full(len(yte), present[0]), C)
    scores = np.zeros(len(ALPHA_REL))
    for tr, va in KFold(3, shuffle=True, random_state=seed % (2**31)).split(Ztr):
        if len(np.unique(yp[tr])) < 2:
            continue
        for i, ro in enumerate(ridge_multi_alpha(Ztr[tr], yp[tr], C, ALPHA_REL)):
            scores[i] += balanced_accuracy(yp[va], ro.predict(Ztr[va]), C)
    ai = int(np.argmax(scores))
    ro = ridge_multi_alpha(Ztr, yp, C, ALPHA_REL[ai:ai + 1])[0]
    return balanced_accuracy(yte, ro.predict(Zte), C)


def fit_logistic(Z: np.ndarray, y: np.ndarray, C: int, folds) -> LogisticRegression:
    best, bc = -1.0, LR_CS[0]
    for c in LR_CS:
        sc = 0.0
        for tr, va in folds:
            m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(Z[tr], y[tr])
            sc += balanced_accuracy(y[va], m.predict(Z[va]), C)
        if sc > best:
            best, bc = sc, c
    return LogisticRegression(C=bc, max_iter=3000, class_weight="balanced").fit(Z, y)


def capacity_profile(state_signal: np.ndarray, Yc: np.ndarray, t0: int) -> np.ndarray:
    E, L = state_signal.shape
    X = np.stack([state_signal[:, t0 - k: L - k].reshape(-1) for k in range(CAP_STATE_LAGS)], axis=1)
    X = X - X.mean(0)
    sc = X.std(0)
    keep = sc > 1e-12
    if not keep.any():
        return np.zeros(Yc.shape[1])
    Q, R = np.linalg.qr(X[:, keep] / sc[keep])
    diag = np.abs(np.diag(R))
    Q = Q[:, diag > 1e-8 * diag.max()]
    proj = Q.T @ Yc
    cap = np.sum(proj * proj, axis=0) / np.sum(Yc * Yc, axis=0)
    cap[cap < max(CAP_THRESHOLD, 4.0 * (CAP_STATE_LAGS + 1) / X.shape[0])] = 0.0
    return cap


def capacity_emulation_analysis(seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        Up = probe_inputs(seed)
        Yc, deg = capacity_targets(Up, CAP_T0)
        d1 = np.flatnonzero(deg == 1)
        insts = {f: make_instance(f, seed, "A") for f in FAMILY_CODES}
        trajs = {f: simulate(insts[f], Up, range(N_OPS), "traj", ("probe", seed)) for f in FAMILY_CODES}
        prof = {f: np.stack([capacity_profile(trajs[f][:, :, op], Yc, CAP_T0) for op in range(N_OPS)]) for f in FAMILY_CODES}
        for s in FAMILY_CODES:
            for t in FAMILY_CODES:
                if s == t:
                    continue
                pad = rest_observable(insts[t])
                for op in range(N_OPS):
                    ps, pt = trajs[s][:, :, op], trajs[t][:, :, op]
                    mp = fit_lag_map(ps[:PROBE_TRAIN_EPISODES], pt[:PROBE_TRAIN_EPISODES], pad, FIR_TAPS, False)
                    ms, mt = prof[s][op][d1], prof[t][op][d1]
                    tot_s, tot_t = prof[s][op].sum(), prof[t][op].sum()
                    rows.append({"seed": seed, "source": s, "target": t, "op": op,
                                 "emulation_r2": r2_score_flat(ps[PROBE_TRAIN_EPISODES:], mp.apply(pt[PROBE_TRAIN_EPISODES:])),
                                 "target_total_capacity": tot_t, "source_total_capacity": tot_s,
                                 "memory_profile_corr": float(np.corrcoef(ms, mt)[0, 1]) if ms.std() > 0 and mt.std() > 0 else float("nan"),
                                 "lf_product": (ms.sum() / tot_s if tot_s > 0 else np.nan) * (mt.sum() / tot_t if tot_t > 0 else np.nan)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------


def hier_boot_fast(df: pd.DataFrame, col: str, seed: int) -> Tuple[float, float, float]:
    """Median of dataset medians; CI by resampling datasets, then rows within datasets."""
    groups = [g[col].dropna().to_numpy(np.float64) for _, g in df.groupby("dataset")]
    groups = [g for g in groups if len(g)]
    if not groups:
        return (float("nan"),) * 3
    obs = float(np.median([np.median(g) for g in groups]))
    rng = np.random.default_rng(seed)
    M = np.stack([np.median(g[rng.integers(0, len(g), (N_BOOT, len(g)))], axis=1) for g in groups])
    pick = rng.integers(0, len(groups), (N_BOOT, len(groups)))
    stats = np.median(M[pick, np.arange(N_BOOT)[:, None]], axis=1)
    return obs, float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def wilcoxon_two(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.all(np.abs(d) < 1e-12):
        return float("nan"), len(d)
    return float(wilcoxon(d, zero_method="wilcox").pvalue), len(d)


def blocked_rhos(df: pd.DataFrame, x: str, y: str) -> np.ndarray:
    out = []
    for _, g in df.groupby("dataset"):
        g = g[[x, y]].dropna()
        if len(g) >= 5 and g[x].std() > 0 and g[y].std() > 0:
            out.append(float(spearmanr(g[x], g[y])[0]))
    return np.array(out)


def rho_test(rhos: np.ndarray, alternative: str) -> Dict[str, float]:
    if len(rhos) < 3:
        return {"mean_rho": float("nan"), "datasets": int(len(rhos)), "p": float("nan")}
    p = float(wilcoxon(rhos, alternative=alternative).pvalue) if np.any(rhos != 0) else 1.0
    return {"mean_rho": float(rhos.mean()), "median_rho": float(np.median(rhos)), "datasets": int(len(rhos)),
            "datasets_in_direction": int(np.sum(rhos < 0) if alternative == "less" else np.sum(rhos > 0)), "p": p}


def cluster_boot_spearman(df: pd.DataFrame, x: str, y: str, cluster: List[str], seed: int) -> Dict[str, float]:
    df = df.dropna(subset=[x, y])
    groups = [g for _, g in df.groupby(cluster)]
    obs = float(spearmanr(df[x], df[y])[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(min(N_BOOT, 2000)):
        g = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        bs.append(spearmanr(g[x], g[y])[0])
    bs = np.array(bs)
    return {"rho": obs, "ci95_low": float(np.nanquantile(bs, 0.025)), "ci95_high": float(np.nanquantile(bs, 0.975)),
            "p_one_sided_positive": float(np.mean(bs <= 0)), "n_points": int(len(df)), "n_clusters": len(groups)}


def lodo_prediction(df: pd.DataFrame, x: str, y: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = []
    for d in df.dataset.unique():
        tr = df[df.dataset != d].dropna(subset=[x, y])
        te = df[df.dataset == d].dropna(subset=[x, y])
        if len(tr) < 10 or len(te) == 0:
            continue
        coef = np.polyfit(tr[x], tr[y], 2)
        out.append(te.assign(predicted=np.polyval(coef, te[x])))
    if not out:
        return pd.DataFrame(), {}
    o = pd.concat(out)
    rh = blocked_rhos(o, "predicted", y)
    return o, {"pooled_spearman": float(spearmanr(o.predicted, o[y])[0]), "mae": float(np.mean(np.abs(o.predicted - o[y]))),
               "within_dataset": rho_test(rh, "greater")}





# =============================================================================
# v5: independent test of a universal operating point
# =============================================================================

V2_DIR = PROJECT_ROOT / ("portability_v2" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else ""))
V4_DIR = PROJECT_ROOT / ("portability_v4_main_evaluation" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else ""))
V5_METHODS = ["full_refit", "zero_shot", "moment", "paired_linear", "paired_rff", "distill_rff", "pseudo_label",
              "twin_paired_rff", f"scratch_k{K_MAIN}", f"moment_warm_k{K_MAIN}"]
V5_D2D = [0.05, 0.50]
POLICY_ORDER = [("random", "Random"), ("source_selected", "Source\nselected"), ("consensus_others", "Consensus of\nother families"),
                ("family_default_v2", "Family\ndefault"), ("universal_v2", "Universal\ndefault"),
                ("target_cv_oracle", "Target CV\noracle"), ("test_ceiling", "Test\nceiling")]


def _ci_plot(ax, x, med, lo, hi, color, marker, label=None, ms=9):
    med, lo, hi = map(np.asarray, (med, lo, hi))
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt=marker, color=color, ms=ms, capsize=5, lw=2, label=label)


def compare_ds(a: pd.Series, b: pd.Series, two_sided: bool = False, higher: bool = False, seed: int = 0) -> Dict[str, float]:
    """Dataset-level paired comparison with a bootstrap CI on the median difference."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if two_sided:
        p = wilcoxon_two(j.a, j.b)[0]
    else:
        p = wilcoxon_less(j.b.to_numpy(), j.a.to_numpy())[0] if higher else wilcoxon_less(j.a.to_numpy(), j.b.to_numpy())[0]
    d = (j.a - j.b).to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.median(d[rng.integers(0, len(d), (N_BOOT, len(d)))], axis=1) if len(d) else np.array([np.nan])
    better = (j.a > j.b) if higher else (j.a < j.b)
    return {"median_a": float(j.a.median()), "median_b": float(j.b.median()), "median_difference": float(np.median(d)) if len(d) else np.nan,
            "diff_ci95_low": float(np.quantile(bs, 0.025)), "diff_ci95_high": float(np.quantile(bs, 0.975)),
            "datasets_a_better": int(better.sum()), "datasets": int(len(j)), "p": p}


def mean_rank_map(land: pd.DataFrame, unit: List[str]) -> pd.Series:
    land = land.copy()
    land["rank"] = land.groupby(unit).cv_ba.rank(pct=True)
    return land.groupby("op")["rank"].mean()


def choose_universal(v2land: pd.DataFrame) -> Tuple[int, Dict[str, int], pd.Series, pd.DataFrame]:
    unit = ["dataset", "seed", "instance"]
    mr = mean_rank_map(v2land, unit)
    u = int(mr.idxmax())
    fam = {f: int(mean_rank_map(g, unit).idxmax()) for f, g in v2land.groupby("family")}
    # Stability of the choice when the v2 datasets are resampled.
    names = v2land.dataset.unique()
    rng = np.random.default_rng(stable_seed("ustab"))
    v2land = v2land.copy()
    v2land["rank"] = v2land.groupby(unit).cv_ba.rank(pct=True)
    per_ds = v2land.groupby(["dataset", "op"])["rank"].mean().unstack()
    wins = np.zeros(N_OPS)
    for _ in range(2000):
        pick = rng.choice(names, len(names), replace=True)
        wins[int(per_ds.loc[pick].mean(axis=0).to_numpy().argmax())] += 1
    stab = pd.DataFrame({"op": range(N_OPS), "gain": GAINS[OP_GAIN_IDX], "frac": FRACS[OP_FRAC_IDX],
                         "mean_rank_v2": mr.reindex(range(N_OPS)).to_numpy(), "bootstrap_win_fraction": wins / wins.sum()})
    return u, fam, mr, stab


def policy_table(land: pd.DataFrame, u: int, fam_u: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for (d, seed), g in land.groupby(["dataset", "seed"]):
        tab = {f: gf.sort_values("op") for f, gf in g.groupby("family")}
        rk = {f: t.cv_ba.rank(pct=True).to_numpy() for f, t in tab.items()}
        for t, tt in tab.items():
            tn = tt.test_nba.to_numpy()
            sel = [tn[int(np.argmax(tab[s].cv_ba.to_numpy()))] for s in tab if s != t]
            cons = int(np.argmax(np.mean([rk[s] for s in tab if s != t], axis=0)))
            rows.append({"dataset": d, "tier": tt.tier.iloc[0], "seed": seed, "target": t, "random": float(tn.mean()),
                         "source_selected": float(np.median(sel)), "consensus_others": float(tn[cons]),
                         "family_default_v2": float(tn[fam_u[t]]), "universal_v2": float(tn[u]),
                         "target_cv_oracle": float(tn[int(np.argmax(tt.cv_ba.to_numpy()))]), "test_ceiling": float(tn.max()),
                         "cv_test_spearman": float(spearmanr(tt.cv_ba, tt.test_ba)[0]) if tt.cv_ba.std() > 0 and tt.test_ba.std() > 0 else np.nan,
                         "selection_optimism": float(tt.cv_ba.max() - tt.test_ba.to_numpy()[int(np.argmax(tt.cv_ba.to_numpy()))]),
                         "universal_optimism": float(tt.cv_ba.to_numpy()[u] - tt.test_ba.to_numpy()[u])})
    return pd.DataFrame(rows)


def run_unit_v5(ds: "Dataset", seed: int, u: int) -> pd.DataFrame:
    C = ds.n_classes
    ytr, yte = ds.y_train, ds.y_test
    folds = make_folds(ytr, stable_seed("folds", ds.name, seed))
    ws = class_weights(ytr, C).sum()
    key_tr, key_te = ("task", ds.name, seed, "train"), ("task", ds.name, seed, "test")
    rng_k = np.random.default_rng(stable_seed("kshot4", ds.name, seed))
    kidx = np.concatenate([rng_k.permutation(np.flatnonzero(ytr == c))[:K_MAIN] for c in range(C)])
    feats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def F(inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
        if inst.iid not in feats:
            feats[inst.iid] = (simulate(inst, ds.U_train, [u], "features", key_tr)[0], simulate(inst, ds.U_test, [u], "features", key_te)[0])
        return feats[inst.iid]

    rows = []
    for s in FAMILY_CODES:
        src = make_instance(s, seed, "A")
        Fs_tr, Fs_te = F(src)
        _, ai_s, src_ba = select_and_test(Fs_tr, Fs_te, ytr, yte, C, folds)
        mu_s, sd_s = std_stats(Fs_tr)
        Zs_tr = (Fs_tr - mu_s) / sd_s
        ro_s = ridge_fit(Zs_tr, ytr, C, ALPHA_REL[ai_s] * ws)
        S_tr = Zs_tr @ ro_s.W + ro_s.b
        yp = ro_s.predict(Zs_tr)
        twin = nominal_instance(s)
        Ftw_tr = simulate(twin, ds.U_train, [u], "features", key_tr)[0]
        mtw, stw = std_stats(Ftw_tr)
        Ztw_tr = (Ftw_tr - mtw) / stw
        targets = [(make_instance(t, seed, "A"), "cross", float("nan")) for t in FAMILY_CODES if t != s]
        targets += [(make_instance_sigma(s, seed, "B" if abs(g - 0.05) < 1e-12 else f"B_sig{g:.2f}", g), "within", g) for g in V5_D2D]
        for tinst, ptype, sig in targets:
            Ft_tr, Ft_te = F(tinst) if ptype == "cross" else (simulate(tinst, ds.U_train, [u], "features", key_tr)[0],
                                                              simulate(tinst, ds.U_test, [u], "features", key_te)[0])
            mu_t, sd_t = std_stats(Ft_tr)
            Zt_tr, Zt_te = (Ft_tr - mu_t) / sd_t, (Ft_te - mu_t) / sd_t
            _, ai_t, full_ba = select_and_test(Ft_tr, Ft_te, ytr, yte, C, folds)
            pred = lambda Z: balanced_accuracy(yte, ro_s.predict(Z), C)
            sd_seed = stable_seed("map5", ds.name, seed, s, tinst.iid)
            res = {"full_refit": full_ba, "zero_shot": pred((Ft_te - mu_s) / sd_s), "moment": pred(Zt_te)}
            res["paired_linear"] = pred(fit_feature_map(Zt_tr, Zs_tr, "lin", sd_seed)(Zt_te))
            res["paired_rff"] = pred(fit_feature_map(Zt_tr, Zs_tr, "rff", sd_seed + 1)(Zt_te))
            res["distill_rff"] = balanced_accuracy(yte, np.argmax(fit_feature_map(Zt_tr, S_tr, "rff", sd_seed + 2)(Zt_te), 1), C)
            res["pseudo_label"] = pseudo_label_ba(Zt_tr, Zt_te, yp, yte, C, sd_seed)
            res["twin_paired_rff"] = pred(fit_feature_map(Zt_tr, Ztw_tr, "rff", sd_seed + 3)(Zt_te))
            yk = ytr[kidx]
            res[f"scratch_k{K_MAIN}"] = balanced_accuracy(yte, ridge_fit(Zt_tr[kidx], yk, C, LAMBDA_PRIOR).predict(Zt_te), C)
            res[f"moment_warm_k{K_MAIN}"] = balanced_accuracy(yte, ridge_fit(Zt_tr[kidx], yk, C, LAMBDA_PRIOR, prior=ro_s.W).predict(Zt_te), C)
            cc = column_corr(Fs_tr, Ft_tr)
            level = "cross" if ptype == "cross" else f"d2d_{sig:.2f}"
            fn = nba(full_ba, C)
            for m, ba in res.items():
                rows.append({"dataset": ds.name, "tier": ds.tier, "seed": seed, "source": s, "target": tinst.family, "level": level,
                             "op": u, "method": m, "ba": ba, "nba": nba(ba, C), "full_refit_nba": fn, "gap": fn - nba(ba, C),
                             "source_nba": nba(src_ba, C), "feature_corr_mean": float(np.nanmean(cc))})
    return pd.DataFrame(rows)


def make_v5_figures(R: Dict[str, Any]) -> None:
    # Figure 13: universal map and operating-point policies.
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.0), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    st = R["stab"]
    piv = st.pivot(index="gain", columns="frac", values="mean_rank_v2").sort_index(ascending=False)
    A = piv.to_numpy()
    lo = math.floor(np.nanmin(A) * 10) / 10
    hi = math.ceil(np.nanmax(A) * 10) / 10
    im = ax.imshow(A, cmap="viridis", vmin=lo, vmax=hi, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=11, color="white" if (A[i, j] - lo) / max(hi - lo, 1e-9) < 0.6 else "black")
    gu, fu = R["universal"]["gain"], R["universal"]["frac"]
    ii, jj = list(piv.index).index(gu), list(piv.columns).index(fu)
    ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False, edgecolor="red", lw=3))
    ax.set_xlabel("Drive duration / switching time"); ax.set_ylabel("Input gain")
    style_ax(ax)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    cb = fig.colorbar(im, cax=cax)
    tk = np.round(np.arange(lo, hi + 0.05, 0.1), 2)
    cb.set_ticks(tk); cb.ax.set_yticklabels([f"{v:.1f}" for v in tk])
    cb.set_label("Mean percentile rank, v2 runs"); cb.outline.set_linewidth(SPINE_W)
    ax = axes[1]
    p = R["policies"]
    x = np.arange(len(p))
    _ci_plot(ax, x, p["median"], p.ci95_low, p.ci95_high, "#1b6ca8", "o")
    ax.set_xticks(x); ax.set_xticklabels(p.label, fontsize=13); ax.set_xlim(-0.5, len(p) - 0.5)
    set_bounds(ax, "y", min(0.0, float(p.ci95_low.min())), float(p.ci95_high.max()))
    ax.set_ylabel("Target normalized BA")
    style_ax(ax)
    fig.tight_layout(w_pad=4); panel_letters(fig, axes); save_fig(fig, "fig13_universal_operating_point")

    # Figure 14: transfer at the universal point versus the source-selected point, and selection noise.
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    t = R["transfer_compare"]
    ms = [m for m in ["full_refit", "moment", "paired_rff", "pseudo_label", "twin_paired_rff"] if m in set(t.method)]
    lab = {"full_refit": "Full\nrefit", "moment": "Moment\nmatching", "paired_rff": "Paired\nnonlinear", "pseudo_label": "Pseudo\nlabels",
           "twin_paired_rff": "Twin\nnonlinear"}
    allv = [0.0]
    for where, col, mk, off in [("source_selected", "#c0392b", "s", -0.12), ("universal", "#1b6ca8", "o", 0.12)]:
        g = t[t["where"] == where].set_index("method").reindex(ms)
        _ci_plot(ax, np.arange(len(ms)) + off, g.median_nba, g.ci95_low, g.ci95_high, col, mk,
                 "Source-selected point" if where == "source_selected" else "Universal point")
        allv += list(g.ci95_low.dropna()) + list(g.ci95_high.dropna())
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([lab[m] for m in ms], fontsize=13); ax.set_xlim(-0.5, len(ms) - 0.5)
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_ylabel("Cross-family target normalized BA")
    style_ax(ax); outlined_legend(ax, loc="best")
    ax = axes[1]
    dd = R["deficit"]
    ax.scatter(dd.log_n_train, dd.oracle_minus_universal, s=60, color="#1b6ca8", edgecolor="black", lw=0.6)
    ax.axhline(0, color="0.4", ls=":", lw=1)
    set_bounds(ax, "x", float(dd.log_n_train.min()), float(dd.log_n_train.max()))
    set_bounds(ax, "y", min(0.0, float(dd.oracle_minus_universal.min())), max(0.0, float(dd.oracle_minus_universal.max())))
    ax.set_xlabel("Training set size (natural log)"); ax.set_ylabel("Target CV oracle minus universal")
    style_ax(ax)
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig14_transfer_at_universal_point")


def main() -> None:
    t_start = time.time()
    log("=" * 100)
    log(f"MEMRISTIVE RESERVOIR PORTABILITY v5 UNIVERSAL OPERATING POINT | root={ROOT} | smoke={SMOKE}")
    log("Drive mounting is intentionally not performed by this script.")
    for fam in FAMILY_CODES:
        T_SWITCH[fam] = calibrate_switch_time(fam)
        T_RECOVER[fam] = recovery_duration(fam)
    p2 = V2_DIR / "tables" / "operating_point_landscape_full.csv"
    p4 = V4_DIR / "tables" / "operating_point_landscape.csv"
    p4u = V4_DIR / "tables" / "pair_level_results.csv"
    p4q = V4_DIR / "tables" / "raw_baseline_noise_decomposition_rows.csv"
    for p in (p2, p4, p4u):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run v2 and v4 first.")
    v2land, v4land = pd.read_csv(p2), pd.read_csv(p4)
    v4pair = pd.read_csv(p4u)
    v4q = pd.read_csv(p4q) if p4q.exists() else None

    # The universal point is fixed from v2 alone (seeds 11 to 33, 18 datasets) before v4 data are read for evaluation.
    u, fam_u, mr, stab = choose_universal(v2land)
    uni = {"op": u, "gain": float(GAINS[OP_GAIN_IDX[u]]), "frac": float(FRACS[OP_FRAC_IDX[u]]),
           "bootstrap_win_fraction": float(stab.bootstrap_win_fraction[u]),
           "per_family": {f: {"duration": op_duration(f, u), "drive_voltage_at_u_plus1": float(u_to_voltage(f, np.array(GAINS[OP_GAIN_IDX[u]]))),
                              "drive_voltage_at_u_minus1": float(u_to_voltage(f, np.array(-GAINS[OP_GAIN_IDX[u]]))),
                              "duration_unit": "ticks" if FAMILIES[f].discrete else "s"} for f in FAMILY_CODES},
           "family_defaults": {f: {"op": o, "gain": float(GAINS[OP_GAIN_IDX[o]]), "frac": float(FRACS[OP_FRAC_IDX[o]])} for f, o in fam_u.items()},
           "near_optimal_ops": stab[stab.mean_rank_v2 >= stab.mean_rank_v2.max() - 0.02][["op", "gain", "frac", "mean_rank_v2"]].to_dict("records"),
           "chosen_from": str(p2), "chosen_utc": datetime.now(timezone.utc).isoformat()}
    save_json(uni, DIRS["config"] / "universal_operating_point.json")
    stab.to_csv(DIRS["tables"] / "universal_point_v2_rank_map.csv", index=False)
    log(f"Universal point from v2: op {u} (gain {uni['gain']}, duration fraction {uni['frac']}), bootstrap win fraction {uni['bootstrap_win_fraction']:.2f}")

    # Policy evaluation on the independent v4 landscapes.
    pol = policy_table(v4land, u, fam_u)
    pol.to_csv(DIRS["tables"] / "policy_rows.csv", index=False)
    dsl = pol.groupby(["dataset", "tier"], as_index=False).median(numeric_only=True)
    prow = []
    for c, lab in POLICY_ORDER:
        med, lo, hi = hier_boot_fast(pol, c, stable_seed("pol", c))
        prow.append({"policy": c, "label": lab, "median": med, "ci95_low": lo, "ci95_high": hi})
    policies = pd.DataFrame(prow)
    policies.to_csv(DIRS["tables"] / "policy_summary.csv", index=False)
    D = dsl.set_index("dataset")
    tests = {"P1 universal > source selected": compare_ds(D.universal_v2, D.source_selected, higher=True, seed=1),
             "P2 universal vs target CV oracle (two-sided)": compare_ds(D.universal_v2, D.target_cv_oracle, two_sided=True, higher=True, seed=2),
             "P3 family default vs universal (two-sided)": compare_ds(D.family_default_v2, D.universal_v2, two_sided=True, higher=True, seed=3),
             "P4 consensus of other families > source selected": compare_ds(D.consensus_others, D.source_selected, higher=True, seed=4),
             "P5 universal vs consensus of other families (two-sided)": compare_ds(D.universal_v2, D.consensus_others, two_sided=True, higher=True, seed=5)}
    ph = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = ph[k]
    by_tier = {tier: {"universal_vs_source_selected": compare_ds(g.set_index("dataset").universal_v2, g.set_index("dataset").source_selected, higher=True),
                      "universal_vs_target_cv_oracle": compare_ds(g.set_index("dataset").universal_v2, g.set_index("dataset").target_cv_oracle, True, True)}
               for tier, g in dsl.groupby("tier")}
    by_family = pd.DataFrame([{"target": t, **compare_ds(g.groupby("dataset").universal_v2.median(), g.groupby("dataset").target_cv_oracle.median(), True, True)}
                              for t, g in pol.groupby("target")])
    by_family.to_csv(DIRS["tables"] / "universal_vs_oracle_by_family.csv", index=False)

    # Why per-dataset selection is noisy.
    props = []
    names = sorted(v4land.dataset.unique())
    loaded: List[Dataset] = []
    for name in names:
        try:
            ds = synthetic_dataset(name, stable_seed(name)) if SMOKE else load_dataset(name)
        except Exception as exc:
            log(f"SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        ds.tier = v4land[v4land.dataset == name].tier.iloc[0]
        loaded.append(ds)
        props.append({"dataset": name, "log_n_train": math.log(len(ds.y_train))})
    props = pd.DataFrame(props)
    deficit = dsl.merge(props, on="dataset")
    deficit["oracle_minus_universal"] = deficit.target_cv_oracle - deficit.universal_v2
    why = {"median_cv_test_spearman_over_grid": float(pol.cv_test_spearman.median()),
           "median_selection_optimism": float(pol.selection_optimism.median()),
           "median_universal_optimism": float(pol.universal_optimism.median()),
           "optimism_selected_minus_universal": compare_ds(dsl.set_index("dataset").universal_optimism, dsl.set_index("dataset").selection_optimism, seed=6),
           "deficit_vs_log_n_train": dict(zip(["rho", "p"], map(float, spearmanr(deficit.log_n_train, deficit.oracle_minus_universal)))),
           "test_ceiling_minus_universal_median": float((dsl.test_ceiling - dsl.universal_v2).median())}

    # Absolute device performance at the universal point against the raw binned baseline.
    raw_cmp = None
    if v4q is not None:
        rq = v4q.groupby("dataset").raw_nba.median()
        at_u = v4land[v4land.op == u].groupby(["dataset", "family"]).test_nba.median().reset_index()
        at_sel = v4land[v4land.selected.astype(bool)].groupby(["dataset", "family"]).test_nba.median().reset_index()
        rows = []
        for f in FAMILY_CODES:
            a = at_u[at_u.family == f].set_index("dataset").test_nba
            b = at_sel[at_sel.family == f].set_index("dataset").test_nba
            rows.append({"family": f, "gain_over_raw_universal": float((a - rq).median()), "gain_over_raw_selected": float((b - rq).median()),
                         "datasets_universal_beats_raw": int(((a - rq) > 0).sum()), "datasets": int(a.notna().sum())})
        raw_cmp = pd.DataFrame(rows)
        raw_cmp.to_csv(DIRS["tables"] / "raw_baseline_at_universal_point.csv", index=False)

    # Transfer ladder rerun at the universal point.
    U = []
    loaded.sort(key=lambda d: (len(d.y_train) + len(d.y_test)) * d.U_train.shape[1])
    for ds in loaded:
        for seed in V4_SEEDS:
            pth = DIRS["units"] / f"v5_{ds.name}_s{seed}_op{u}.csv"
            if pth.exists():
                U.append(pd.read_csv(pth)); continue
            t0 = time.time()
            try:
                df = run_unit_v5(ds, seed, u)
            except Exception:
                log(f"FAILED {ds.name} seed {seed}:\n{traceback.format_exc()}")
                continue
            df.to_csv(pth, index=False); U.append(df)
            m = df[df.level == "cross"].groupby("method").nba.median()
            log(f"{ds.name} seed {seed}: {(time.time() - t0) / 60:.1f} min | cross nBA full={m.get('full_refit', np.nan):.3f} "
                f"moment={m.get('moment', np.nan):.3f} paired={m.get('paired_rff', np.nan):.3f}")
    units = pd.concat(U, ignore_index=True)
    units.to_csv(DIRS["tables"] / "all_units_universal_point.csv", index=False)
    keys = ["dataset", "tier", "source", "target", "level", "method"]
    pu = units.groupby(keys, as_index=False)[["gap", "nba", "full_refit_nba", "feature_corr_mean"]].median()
    p4m = v4pair[(v4pair.analysis == "main") & v4pair.level.isin(pu.level.unique()) & v4pair.method.isin(V5_METHODS)]
    trows, tc = [], {}
    for lvl in sorted(pu.level.unique()):
        for m in V5_METHODS:
            for where, df in [("universal", pu), ("source_selected", p4m)]:
                g = df[(df.level == lvl) & (df.method == m)]
                if not len(g):
                    continue
                mn, ln, hn = hier_boot_fast(g, "nba", stable_seed("tc", lvl, m, where, "n"))
                mg, lg, hg = hier_boot_fast(g, "gap", stable_seed("tc", lvl, m, where, "g"))
                trows.append({"level": lvl, "method": m, "where": where, "median_nba": mn, "ci95_low": ln, "ci95_high": hn,
                              "median_gap": mg, "gap_ci95_low": lg, "gap_ci95_high": hg})
            a = pu[(pu.level == lvl) & (pu.method == m)].groupby("dataset").nba.median()
            b = p4m[(p4m.level == lvl) & (p4m.method == m)].groupby("dataset").nba.median()
            if len(a) and len(b):
                tc[f"{lvl} {m}: nBA universal vs source selected"] = compare_ds(a, b, two_sided=True, higher=True, seed=stable_seed(lvl, m))
    tcomp = pd.DataFrame(trows)
    tcomp.to_csv(DIRS["tables"] / "transfer_universal_vs_selected.csv", index=False)
    cr = pu[pu.level == "cross"]
    dsv = lambda m, col="gap": cr[cr.method == m].groupby("dataset")[col].median()
    replic = {"paired_rff < moment (gap)": compare_ds(dsv("paired_rff"), dsv("moment"), seed=11),
              "twin_rff < moment (gap)": compare_ds(dsv("twin_paired_rff"), dsv("moment"), seed=12),
              "paired_rff vs pseudo_label (gap, two-sided)": compare_ds(dsv("paired_rff"), dsv("pseudo_label"), True, seed=13),
              "feature correlation predicts moment gap": rho_test(blocked_rhos(pu[pu.method == "moment"], "feature_corr_mean", "gap"), "less")}
    R = {"stab": stab, "universal": uni, "policies": policies, "deficit": deficit,
         "transfer_compare": tcomp[tcomp.level == "cross"]}
    try:
        make_v5_figures(R)
    except Exception:
        log("Figure generation failed:\n" + traceback.format_exc())

    elapsed = (time.time() - t_start) / 60
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    J = lambda o: json.dumps(o, indent=1, default=str)
    print("\n" + "=" * 110)
    print("COPY-PASTE REPORT START - MEMRISTIVE RESERVOIR PORTABILITY v5 UNIVERSAL OPERATING POINT")
    print("=" * 110)
    print(f"Runtime: {elapsed:.1f} min | universal point chosen from {V2_DIR.name} only | evaluated on {V4_DIR.name}, seeds {V4_SEEDS}")
    print("\nUNIVERSAL POINT:", J({k: v for k, v in uni.items() if k not in ("chosen_from", "chosen_utc")}))
    print("\nTOP OF THE v2 RANK MAP"); print(stab.sort_values("mean_rank_v2", ascending=False).head(8).to_string(index=False))
    print("\nOPERATING-POINT POLICIES ON v4 (target normalized BA)"); print(policies.drop(columns="label").to_string(index=False))
    print("\nPOLICY TESTS (dataset level, Holm over P1 to P5)")
    print(pd.DataFrame([{"test": k, **v} for k, v in tests.items()]).to_string(index=False))
    print("\nBY TIER:", J(by_tier))
    print("\nUNIVERSAL VS TARGET CV ORACLE BY TARGET FAMILY"); print(by_family.to_string(index=False))
    print("\nWHY PER-DATASET SELECTION IS NOISY:", J(why))
    if raw_cmp is not None:
        print("\nGAIN OVER RAW BASELINE, UNIVERSAL VS SELECTED POINT"); print(raw_cmp.to_string(index=False))
    print("\nTRANSFER AT THE UNIVERSAL POINT VS THE SOURCE-SELECTED POINT")
    print(tcomp.to_string(index=False))
    print(pd.DataFrame([{"comparison": k, **v} for k, v in tc.items()]).to_string(index=False))
    print("\nREPLICATION OF THE MAIN TRANSFER RESULTS AT THE UNIVERSAL POINT:", J(replic))
    print("=" * 110)
    print("COPY-PASTE REPORT END")
    print("=" * 110)
    log(f"Done in {elapsed:.1f} min. Tables: {DIRS['tables']} | Figures: {DIRS['figures']}")


if __name__ == "__main__":
    main()
