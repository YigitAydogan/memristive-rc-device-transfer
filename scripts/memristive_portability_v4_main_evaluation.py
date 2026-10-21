"""
Memristive reservoir portability, v4: preregistered confirmatory study and reviewer-question battery.

Fresh device seeds (44, 55, 66) never used in v2 or v3, the 18 original datasets plus 10 held-out datasets never
examined in this project. The hypotheses are written to config/protocol_v4.json before any unit runs.

Confirmatory (Holm corrected): paired feature-level alignment beats moment matching and linear maps; a simulated
nominal twin can replace the physical source device; feature correspondence predicts portability loss; readout
cosine predicts warm-start benefit; the operating point transfers; capacity predictors of emulation; comparisons
against pseudo-label refits and a universal default operating point.

Exploratory (reported as such): device-to-device sweep, pairing budget, pairing inputs that are not task data,
transfer ladder, k-shot after pairing, family-pair matrices, robustness to noise, feature design and readout
family, out-of-sample gap prediction, operating-point decomposition, noise versus distortion behind the raw
baseline gap, dataset moderators.

Paste into one Colab cell with Drive already mounted. The script does not mount Drive. Every dataset and seed unit
is checkpointed, so an interrupted run resumes. Optional environment variables: PORTV2_SMOKE=1,
PORTV2_DATASETS=a,b, MEMRISTIVE_PROJECT_ROOT=path.
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

SCRIPT_VERSION = "4.0.0"
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
RUN_NAME = "portability_v4_main_evaluation" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else "")
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
# Per dataset and seed unit
# -----------------------------------------------------------------------------


def run_unit_v4(ds: "Dataset", seed: int, pool: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    C = ds.n_classes
    ytr, yte = ds.y_train, ds.y_test
    n, L = ds.U_train.shape
    folds = make_folds(ytr, stable_seed("folds", ds.name, seed))
    ws = class_weights(ytr, C).sum()
    robust = seed == ROBUST_SEED
    key_tr, key_te = ("task", ds.name, seed, "train"), ("task", ds.name, seed, "test")

    # Operating-point landscape for every family (selection by training CV only).
    land_rows, table = [], {}
    for f in FAMILY_CODES:
        inst = make_instance(f, seed, "A")
        Ftr = simulate(inst, ds.U_train, range(N_OPS), "features", key_tr)
        Fte = simulate(inst, ds.U_test, range(N_OPS), "features", key_te)
        cv = np.zeros(N_OPS); ai = np.zeros(N_OPS, int); te = np.zeros(N_OPS)
        cache: Dict[Tuple, Tuple] = {}
        for op in range(N_OPS):
            k = op_key(f, op)
            if k not in cache:
                cache[k] = select_and_test(Ftr[op], Fte[op], ytr, yte, C, folds)
            cv[op], ai[op], te[op] = cache[k]
        best = int(np.argmax(cv))
        table[f] = {"cv": cv, "ai": ai, "test": te, "best": best}
        for op in range(N_OPS):
            land_rows.append({"dataset": ds.name, "tier": ds.tier, "seed": seed, "family": f, "op": op,
                              "gain_idx": int(OP_GAIN_IDX[op]), "frac_idx": int(OP_FRAC_IDX[op]),
                              "gain": GAINS[OP_GAIN_IDX[op]], "frac": FRACS[OP_FRAC_IDX[op]], "cv_ba": cv[op],
                              "test_ba": te[op], "test_nba": nba(te[op], C), "selected": op == best})
        del Ftr, Fte

    Fr_tr, Fr_te = features_from_traj(ds.U_train), features_from_traj(ds.U_test)
    _, _, raw_ba = select_and_test(Fr_tr, Fr_te, ytr, yte, C, folds)
    mr, sr = std_stats(Fr_tr)
    Zr_tr, Zr_te = (Fr_tr - mr) / sr, (Fr_te - mr) / sr
    alt_inputs = {"generic": generic_inputs(pool, ds.name, N_ALT, L, stable_seed("gen", ds.name, seed)),
                  "white": white_inputs(N_ALT, L, stable_seed("white", ds.name, seed)),
                  "matched": matched_inputs(ds.U_train, N_ALT, L, stable_seed("match", ds.name, seed))}
    perm = np.random.default_rng(stable_seed("budget", ds.name, seed)).permutation(n)
    budgets = [b for b in BUDGETS if b < n]
    rng_k = np.random.default_rng(stable_seed("kshot4", ds.name, seed))
    kidx = {k: np.concatenate([rng_k.permutation(np.flatnonzero(ytr == c))[:k] for c in range(C)]) for k in K_V4}

    rows, q6 = [], []
    for s in FAMILY_CODES:
        op_s = table[s]["best"]
        alpha_s = ALPHA_REL[table[s]["ai"][op_s]]
        src = make_instance(s, seed, "A")
        rs_tr = simulate(src, ds.U_train, [op_s], "traj", key_tr)[:, :, 0]
        rs_te = simulate(src, ds.U_test, [op_s], "traj", key_te)[:, :, 0]
        Fs_tr, Fs_te = features_from_traj(rs_tr), features_from_traj(rs_te)
        mu_s, sd_s = std_stats(Fs_tr)
        Zs_tr, Zs_te = (Fs_tr - mu_s) / sd_s, (Fs_te - mu_s) / sd_s
        ro_s = ridge_fit(Zs_tr, ytr, C, alpha_s * ws)
        S_tr = Zs_tr @ ro_s.W + ro_s.b
        yp_src = ro_s.predict(Zs_tr)

        # Why devices trail the raw baseline: read and cycle noise versus deterministic distortion.
        with noise_level(0.0, 0.0):
            F0_tr = simulate(src, ds.U_train, [op_s], "features", key_tr)[0]
            F0_te = simulate(src, ds.U_test, [op_s], "features", key_te)[0]
        _, _, full0 = select_and_test(F0_tr, F0_te, ytr, yte, C, folds)
        _, _, full_noisy = select_and_test(Fs_tr, Fs_te, ytr, yte, C, folds)
        Fre = simulate(src, ds.U_train, [op_s], "features", ("retest", ds.name, seed, "train"))[0]
        m0, s0 = std_stats(F0_tr)
        mpn = fit_feature_map(Zr_tr, Zs_tr, "lin", stable_seed("q6n", ds.name, seed, s))
        mp0 = fit_feature_map(Zr_tr, (F0_tr - m0) / s0, "lin", stable_seed("q60", ds.name, seed, s))
        q6.append({"dataset": ds.name, "tier": ds.tier, "seed": seed, "family": s, "op": op_s, "n_classes": C,
                   "raw_nba": nba(raw_ba, C), "full_noisy_nba": nba(full_noisy, C), "full_noiseless_nba": nba(full0, C),
                   "feature_test_retest_r": float(np.nanmean(column_corr(Fs_tr, Fre))),
                   "noise_vs_noiseless_feature_r": float(np.nanmean(column_corr(Fs_tr, F0_tr))),
                   "raw_to_device_r2_noisy": feature_r2(Zs_te, mpn(Zr_te)),
                   "raw_to_device_r2_noiseless": feature_r2((F0_te - m0) / s0, mp0(Zr_te))})

        twin = nominal_instance(s)
        Ftw_tr = simulate(twin, ds.U_train, [op_s], "features", key_tr)[0]
        mtw, stw = std_stats(Ftw_tr)
        Ztw_tr = (Ftw_tr - mtw) / stw
        Zs_alt = {k: (simulate(src, X, [op_s], "features", ("alt", k, ds.name, seed))[0] - mu_s) / sd_s for k, X in alt_inputs.items()}

        if robust:
            var_src = {}
            for vn, nb, st in FEATURE_VARIANTS:
                Fv_tr, Fv_te = features_variant(rs_tr, nb, st), features_variant(rs_te, nb, st)
                _, ai_v, _ = select_and_test(Fv_tr, Fv_te, ytr, yte, C, folds)
                mv, sv = std_stats(Fv_tr)
                var_src[vn] = (Fv_tr, mv, sv, ridge_fit((Fv_tr - mv) / sv, ytr, C, ALPHA_REL[ai_v] * ws))
            lr_s = fit_logistic(Zs_tr, ytr, C, folds)
            noise_src = {}
            for lvl, (c2c, rd) in NOISE_LEVELS.items():
                with noise_level(c2c, rd):
                    Fn_tr = simulate(src, ds.U_train, [op_s], "features", key_tr)[0]
                mn, sn = std_stats(Fn_tr)
                noise_src[lvl] = (Fn_tr, mn, sn, ridge_fit((Fn_tr - mn) / sn, ytr, C, alpha_s * ws))

        targets = [(make_instance(t, seed, "A"), "cross", float("nan")) for t in FAMILY_CODES if t != s]
        targets += [(make_instance_sigma(s, seed, "B" if abs(sig - 0.05) < 1e-12 else f"B_sig{sig:.2f}", sig), "within", sig)
                     for sig in V4_D2D]
        for tinst, ptype, sig in targets:
            level = "cross" if ptype == "cross" else f"d2d_{sig:.2f}"
            rt_tr = simulate(tinst, ds.U_train, [op_s], "traj", key_tr)[:, :, 0]
            rt_te = simulate(tinst, ds.U_test, [op_s], "traj", key_te)[:, :, 0]
            Ft_tr, Ft_te = features_from_traj(rt_tr), features_from_traj(rt_te)
            mu_t, sd_t = std_stats(Ft_tr)
            Zt_tr, Zt_te = (Ft_tr - mu_t) / sd_t, (Ft_te - mu_t) / sd_t
            _, ai_t, full_ba = select_and_test(Ft_tr, Ft_te, ytr, yte, C, folds)
            ro_t = ridge_fit(Zt_tr, ytr, C, ALPHA_REL[ai_t] * ws)
            pred = lambda Z: balanced_accuracy(yte, ro_s.predict(Z), C)
            sd_seed = stable_seed("map", ds.name, seed, s, tinst.iid)
            res: Dict[str, float] = {"full_refit": full_ba}
            res["zero_shot"] = pred((Ft_te - mu_s) / sd_s)
            res["moment"] = pred(Zt_te)
            cc = column_corr(Fs_tr, Ft_tr)
            res["sign_moment"] = pred(Zt_te * np.where(np.nan_to_num(cc) < 0, -1.0, 1.0))
            dz_t, dz_s = Zt_tr - Zt_tr.mean(0), Zs_tr - Zs_tr.mean(0)
            a_j = np.sum(dz_t * dz_s, 0) / np.maximum(np.sum(dz_t * dz_t, 0), 1e-12)
            res["diag_paired"] = pred(Zt_te * a_j + (Zs_tr.mean(0) - a_j * Zt_tr.mean(0)))
            mlin = fit_feature_map(Zt_tr, Zs_tr, "lin", sd_seed)
            mrff = fit_feature_map(Zt_tr, Zs_tr, "rff", sd_seed + 1)
            Zh_tr, Zh_te = mrff(Zt_tr), mrff(Zt_te)
            res["paired_linear"] = pred(mlin(Zt_te))
            res["paired_rff"] = pred(Zh_te)
            res["distill_linear"] = balanced_accuracy(yte, np.argmax(fit_feature_map(Zt_tr, S_tr, "lin", sd_seed + 2)(Zt_te), 1), C)
            res["distill_rff"] = balanced_accuracy(yte, np.argmax(fit_feature_map(Zt_tr, S_tr, "rff", sd_seed + 3)(Zt_te), 1), C)
            res["pseudo_label"] = pseudo_label_ba(Zt_tr, Zt_te, yp_src, yte, C, sd_seed)
            res["twin_paired_linear"] = pred(fit_feature_map(Zt_tr, Ztw_tr, "lin", sd_seed + 4)(Zt_te))
            res["twin_paired_rff"] = pred(fit_feature_map(Zt_tr, Ztw_tr, "rff", sd_seed + 5)(Zt_te))
            for b in budgets:
                idx = perm[:b]
                res[f"budget_linear_{b}"] = pred(fit_feature_map(Zt_tr[idx], Zs_tr[idx], "lin", sd_seed + 10 + b)(Zt_te))
                res[f"budget_rff_{b}"] = pred(fit_feature_map(Zt_tr[idx], Zs_tr[idx], "rff", sd_seed + 11 + b)(Zt_te))
            for kin, X in alt_inputs.items():
                Zt_a = (simulate(tinst, X, [op_s], "features", ("alt", kin, ds.name, seed))[0] - mu_t) / sd_t
                res[f"alt_{kin}_linear"] = pred(fit_feature_map(Zt_a, Zs_alt[kin], "lin", sd_seed + 20)(Zt_te))
                res[f"alt_{kin}_rff"] = pred(fit_feature_map(Zt_a, Zs_alt[kin], "rff", sd_seed + 21)(Zt_te))
            for k, idx in kidx.items():
                yk = ytr[idx]
                res[f"scratch_k{k}"] = balanced_accuracy(yte, ridge_fit(Zt_tr[idx], yk, C, LAMBDA_PRIOR).predict(Zt_te), C)
                res[f"moment_warm_k{k}"] = balanced_accuracy(yte, ridge_fit(Zt_tr[idx], yk, C, LAMBDA_PRIOR, prior=ro_s.W).predict(Zt_te), C)
                res[f"paired_warm_k{k}"] = balanced_accuracy(yte, ridge_fit(Zh_tr[idx], yk, C, LAMBDA_PRIOR, prior=ro_s.W).predict(Zh_te), C)
            ws_v, wt_v = ro_s.W.reshape(-1), ro_t.W.reshape(-1)
            base = {"dataset": ds.name, "tier": ds.tier, "seed": seed, "source": s, "target": tinst.family,
                    "pair_type": ptype, "d2d_sigma": sig, "level": level, "op": op_s, "n_classes": C,
                    "feature_corr_mean": float(np.nanmean(cc)), "feature_sign_flip_fraction": float(np.nanmean(cc < 0)),
                    "cka": linear_cka(Zs_tr, Zt_tr),
                    "readout_cosine": float(ws_v @ wt_v / max(np.linalg.norm(ws_v) * np.linalg.norm(wt_v), 1e-20)),
                    "twin_feature_corr": float(np.nanmean(column_corr(Ftw_tr, Ft_tr))),
                    "paired_rff_feature_r2": feature_r2(Zs_te, Zh_te), "param_distance": parameter_distance(src, tinst)}
            full_n = nba(full_ba, C)
            for mth, ba in res.items():
                rows.append({**base, "analysis": "main", "method": mth, "ba": ba, "nba": nba(ba, C),
                             "full_refit_nba": full_n, "gap": full_n - nba(ba, C)})

            if robust and (ptype == "cross" or abs(sig - 0.05) < 1e-12):
                rb = {k: base[k] for k in ["dataset", "tier", "seed", "source", "target", "pair_type", "d2d_sigma", "level", "op", "n_classes"]}

                def add(analysis: str, full_v: float, out: Dict[str, float], corr: float) -> None:
                    fn = nba(full_v, C)
                    for mth, ba in out.items():
                        rows.append({**rb, "analysis": analysis, "method": mth, "ba": ba, "nba": nba(ba, C),
                                     "full_refit_nba": fn, "gap": fn - nba(ba, C), "feature_corr_mean": corr})

                for vn, nb, st in FEATURE_VARIANTS:
                    Fvs_tr, mv, sv, ro_v = var_src[vn]
                    Fvt_tr, Fvt_te = features_variant(rt_tr, nb, st), features_variant(rt_te, nb, st)
                    _, _, fv = select_and_test(Fvt_tr, Fvt_te, ytr, yte, C, folds)
                    mt_, st_ = std_stats(Fvt_tr)
                    Zv_tr, Zv_te = (Fvt_tr - mt_) / st_, (Fvt_te - mt_) / st_
                    mp = fit_feature_map(Zv_tr, (Fvs_tr - mv) / sv, "rff", sd_seed + 30)
                    add(vn, fv, {"full_refit": fv, "moment": balanced_accuracy(yte, ro_v.predict(Zv_te), C),
                                 "paired_rff": balanced_accuracy(yte, ro_v.predict(mp(Zv_te)), C)},
                        float(np.nanmean(column_corr(Fvs_tr, Fvt_tr))))
                lr_t = fit_logistic(Zt_tr, ytr, C, folds)
                flr = balanced_accuracy(yte, lr_t.predict(Zt_te), C)
                add("logistic", flr, {"full_refit": flr, "moment": balanced_accuracy(yte, lr_s.predict(Zt_te), C),
                                      "paired_rff": balanced_accuracy(yte, lr_s.predict(Zh_te), C)}, float(np.nanmean(cc)))
                for lvl, (c2c, rd) in NOISE_LEVELS.items():
                    Fns_tr, mn, sn, ro_n = noise_src[lvl]
                    with noise_level(c2c, rd):
                        Fnt_tr = simulate(tinst, ds.U_train, [op_s], "features", key_tr)[0]
                        Fnt_te = simulate(tinst, ds.U_test, [op_s], "features", key_te)[0]
                    _, _, fn_ba = select_and_test(Fnt_tr, Fnt_te, ytr, yte, C, folds)
                    mt_, st_ = std_stats(Fnt_tr)
                    Zn_tr, Zn_te = (Fnt_tr - mt_) / st_, (Fnt_te - mt_) / st_
                    mp = fit_feature_map(Zn_tr, (Fns_tr - mn) / sn, "rff", sd_seed + 40)
                    add(lvl, fn_ba, {"full_refit": fn_ba, "moment": balanced_accuracy(yte, ro_n.predict(Zn_te), C),
                                     "paired_rff": balanced_accuracy(yte, ro_n.predict(mp(Zn_te)), C)},
                        float(np.nanmean(column_corr(Fns_tr, Fnt_tr))))
    return pd.DataFrame(rows), pd.DataFrame(land_rows), pd.DataFrame(q6)


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


def op_decomposition(land: pd.DataFrame) -> pd.DataFrame:
    land = land.copy()
    land["rank"] = land.groupby(["dataset", "seed", "family"]).cv_ba.rank(pct=True)
    mean_rank = land.groupby(["dataset", "op"])["rank"].mean().unstack()
    univ = {}
    for d in mean_rank.index:
        others = mean_rank.drop(d) if len(mean_rank) > 1 else mean_rank
        univ[d] = int(np.argmax(others.mean(axis=0).to_numpy()))
    rows = []
    for (d, seed), g in land.groupby(["dataset", "seed"]):
        tab = {f: gf.sort_values("op") for f, gf in g.groupby("family")}
        for s, ts in tab.items():
            sel = int(np.argmax(ts.cv_ba.to_numpy()))
            for t, tt in tab.items():
                if t == s:
                    continue
                tn = tt.test_nba.to_numpy(); gi = tt.gain_idx.to_numpy(); fi = tt.frac_idx.to_numpy()
                rows.append({"dataset": d, "tier": tt.tier.iloc[0], "seed": seed, "source": s, "target": t,
                             "random": float(tn.mean()), "duration_only": float(tn[fi == fi[sel]].mean()),
                             "gain_only": float(tn[gi == gi[sel]].mean()), "universal_default": float(tn[univ[d]]),
                             "source_selected": float(tn[sel]), "target_oracle": float(tn[int(np.argmax(tt.cv_ba.to_numpy()))])})
    return pd.DataFrame(rows)


def analyse_v4(units: pd.DataFrame, land: pd.DataFrame, q6: pd.DataFrame, capem: pd.DataFrame,
               dsprops: pd.DataFrame) -> Dict[str, Any]:
    R: Dict[str, Any] = {}
    T = DIRS["tables"]
    diag = ["feature_corr_mean", "feature_sign_flip_fraction", "cka", "readout_cosine", "twin_feature_corr",
            "paired_rff_feature_r2", "param_distance"]
    keys = ["dataset", "tier", "source", "target", "level", "analysis", "method"]
    pair = units.groupby(keys, as_index=False)[["gap", "nba", "full_refit_nba"] + diag].median()
    pair.to_csv(T / "pair_level_results.csv", index=False)
    pr = units[units.seed == ROBUST_SEED].groupby(keys, as_index=False)[["gap", "nba", "full_refit_nba", "feature_corr_mean"]].median()
    main = pair[pair.analysis == "main"]
    levels = [f"d2d_{v:.2f}" for v in V4_D2D] + ["cross"]
    R["levels"] = levels

    def dsv(level: str, method: str, col: str = "gap", df: Optional[pd.DataFrame] = None, analysis: str = "main") -> pd.Series:
        d = (main if df is None else df)
        d = d[(d.level == level) & (d.method == method) & (d.analysis == analysis)]
        return d.groupby("dataset")[col].median()

    def compare(a: pd.Series, b: pd.Series, two_sided: bool = False, higher: bool = False) -> Dict[str, float]:
        """Dataset-level paired Wilcoxon. Default: a is better when lower (gaps); higher=True for accuracies."""
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if two_sided:
            p = wilcoxon_two(j.a, j.b)[0]
        else:
            p = wilcoxon_less(j.b.to_numpy(), j.a.to_numpy())[0] if higher else wilcoxon_less(j.a.to_numpy(), j.b.to_numpy())[0]
        better = (j.a > j.b) if higher else (j.a < j.b)
        return {"median_a": float(j.a.median()), "median_b": float(j.b.median()), "median_difference": float((j.a - j.b).median()),
                "datasets_a_better": int(better.sum()), "datasets": int(len(j)), "p": p}

    # Method summary.
    srows = []
    for (lvl, mth), g in main.groupby(["level", "method"]):
        med, lo, hi = hier_boot_fast(g, "gap", stable_seed("v4b", lvl, mth))
        srows.append({"level": lvl, "method": mth, "median_gap": med, "ci95_low": lo, "ci95_high": hi,
                      "median_nba": float(g.groupby("dataset").nba.median().median())})
    summ = pd.DataFrame(srows)
    summ.to_csv(T / "method_summary.csv", index=False)
    R["summary"] = summ
    absl = main[main.method == "full_refit"].groupby("level").agg(full_refit_nba=("full_refit_nba", "median")).reindex(levels)
    absl["moment_nba"] = main[main.method == "moment"].groupby("level").nba.median().reindex(levels)
    absl["paired_rff_nba"] = main[main.method == "paired_rff"].groupby("level").nba.median().reindex(levels)
    absl.to_csv(T / "absolute_performance_by_level.csv")
    R["absolute"] = absl
    R["diag_by_level"] = main[main.method == "moment"].groupby("level")[diag].median().reindex(levels)

    # Preregistered confirmatory tests.
    w = main[main.method == f"scratch_k{K_MAIN}"][["dataset", "source", "target", "level", "gap"]].merge(
        main[main.method == f"moment_warm_k{K_MAIN}"][["dataset", "source", "target", "level", "gap", "readout_cosine"]],
        on=["dataset", "source", "target", "level"], suffixes=("_scratch", "_warm"))
    w["warm_advantage"] = w.gap_scratch - w.gap_warm
    capm = capem.groupby(["source", "target", "op"], as_index=False).median(numeric_only=True)
    od = op_decomposition(land)
    od.to_csv(T / "operating_point_decomposition_rows.csv", index=False)
    odp = od.groupby(["dataset", "tier", "source", "target"], as_index=False).median(numeric_only=True)
    odd = odp.groupby("dataset").median(numeric_only=True)
    conf = {}
    conf["C1 paired nonlinear map < moment matching (cross family)"] = compare(dsv("cross", "paired_rff"), dsv("cross", "moment"))
    conf["C2 paired nonlinear map < paired linear map (cross family)"] = compare(dsv("cross", "paired_rff"), dsv("cross", "paired_linear"))
    conf["C3 twin-paired nonlinear map < moment matching (cross family)"] = compare(dsv("cross", "twin_paired_rff"), dsv("cross", "moment"))
    r4 = rho_test(blocked_rhos(main[main.method == "moment"], "feature_corr_mean", "gap"), "less")
    conf["C4 feature correlation predicts moment gap (rho < 0)"] = r4
    r5 = rho_test(blocked_rhos(w, "readout_cosine", "warm_advantage"), "greater")
    conf["C5 readout cosine predicts warm-start advantage (rho > 0)"] = r5
    conf["C6 source-selected operating point > random operating point"] = compare(odd.source_selected, odd.random, higher=True)
    c8a = cluster_boot_spearman(capm, "target_total_capacity", "emulation_r2", ["source", "target"], stable_seed("c8a"))
    c8b = cluster_boot_spearman(capm, "memory_profile_corr", "emulation_r2", ["source", "target"], stable_seed("c8b"))
    conf["C7a target total capacity predicts probe emulation R2 (rho > 0)"] = {**c8a, "p": c8a["p_one_sided_positive"]}
    conf["C7b memory-profile similarity predicts probe emulation R2 (rho > 0)"] = {**c8b, "p": c8b["p_one_sided_positive"]}
    conf["C8 paired nonlinear map vs pseudo labels (two-sided)"] = compare(dsv("cross", "paired_rff"), dsv("cross", "pseudo_label"), True)
    conf["C9 source-selected vs universal default operating point (two-sided)"] = compare(odd.source_selected, odd.universal_default, True, higher=True)
    ph = holm({k: v["p"] for k, v in conf.items()})
    for k in conf:
        conf[k]["p_holm"] = ph[k]
    R["confirmatory"] = conf
    save_json(conf, T / "confirmatory_tests.json")
    tiers = {}
    for tier in sorted(main.tier.unique()):
        mt = main[main.tier == tier]
        tiers[tier] = {"paired_rff_vs_moment": compare(dsv("cross", "paired_rff", df=mt), dsv("cross", "moment", df=mt)),
                       "twin_vs_moment": compare(dsv("cross", "twin_paired_rff", df=mt), dsv("cross", "moment", df=mt))}
    R["by_tier"] = tiers

    # Exploratory batteries.
    ex = {}
    ex["pseudo_label_vs_moment"] = compare(dsv("cross", "pseudo_label"), dsv("cross", "moment"))
    ex["distill_rff_vs_paired_rff"] = compare(dsv("cross", "distill_rff"), dsv("cross", "paired_rff"), True)
    ex["twin_rff_vs_paired_rff"] = compare(dsv("cross", "twin_paired_rff"), dsv("cross", "paired_rff"), True)
    ex["sign_moment_vs_moment"] = compare(dsv("cross", "sign_moment"), dsv("cross", "moment"))
    ex["diag_paired_vs_sign_moment"] = compare(dsv("cross", "diag_paired"), dsv("cross", "sign_moment"))
    ex["paired_linear_vs_diag_paired"] = compare(dsv("cross", "paired_linear"), dsv("cross", "diag_paired"))
    for k in K_V4:
        ex[f"paired_warm_k{k}_vs_paired_rff"] = compare(dsv("cross", f"paired_warm_k{k}"), dsv("cross", "paired_rff"))
        ex[f"paired_warm_k{k}_vs_scratch_k{k}"] = compare(dsv("cross", f"paired_warm_k{k}"), dsv("cross", f"scratch_k{k}"))
    for kin in ["generic", "white", "matched"]:
        ex[f"alt_{kin}_rff_vs_moment"] = compare(dsv("cross", f"alt_{kin}_rff"), dsv("cross", "moment"))
        ex[f"alt_{kin}_rff_vs_task_paired_rff"] = compare(dsv("cross", f"alt_{kin}_rff"), dsv("cross", "paired_rff"), True)
    R["exploratory"] = ex
    save_json(ex, T / "exploratory_comparisons.json")
    ex_within = {lvl: {"paired_rff_vs_moment": compare(dsv(lvl, "paired_rff"), dsv(lvl, "moment")),
                       "twin_rff_vs_moment": compare(dsv(lvl, "twin_paired_rff"), dsv(lvl, "moment"), True)} for lvl in levels[:-1]}
    R["within_family"] = ex_within

    # Budget curve and pairing inputs.
    brows = []
    for lvl in ["cross", levels[0]]:
        for kind in ["linear", "rff"]:
            for b in BUDGETS + ["all"]:
                mth = f"paired_{kind}" if b == "all" else f"budget_{kind}_{b}"
                g = main[(main.level == lvl) & (main.method == mth)]
                if len(g):
                    med, lo, hi = hier_boot_fast(g, "gap", stable_seed("bud", lvl, kind, b))
                    brows.append({"level": lvl, "map": kind, "budget": b, "median_gap": med, "ci95_low": lo, "ci95_high": hi})
    R["budget"] = pd.DataFrame(brows)
    R["budget"].to_csv(T / "paired_budget_curve.csv", index=False)
    arows = []
    for lab, mth in [("Task inputs", "paired"), ("Matched probe", "alt_matched"), ("White probe", "alt_white"), ("Other datasets", "alt_generic")]:
        for kind in ["linear", "rff"]:
            g = main[(main.level == "cross") & (main.method == f"{mth}_{kind}")]
            med, lo, hi = hier_boot_fast(g, "gap", stable_seed("alt", mth, kind))
            arows.append({"pairing_inputs": lab, "map": kind, "median_gap": med, "ci95_low": lo, "ci95_high": hi})
    R["alt"] = pd.DataFrame(arows)
    R["alt"].to_csv(T / "pairing_input_source.csv", index=False)

    # Family-pair matrices.
    cr = main[main.level == "cross"]
    R["matrix"] = {m: cr[cr.method == m].groupby(["source", "target"]).gap.median().unstack().reindex(index=FAMILY_CODES, columns=FAMILY_CODES)
                   for m in ["moment", "paired_rff"]}
    for m, M in R["matrix"].items():
        M.to_csv(T / f"family_matrix_gap_{m}.csv")
    fam_rows = []
    for s in FAMILY_CODES:
        for role, col in [("source", "source"), ("target", "target")]:
            g = cr[cr[col] == s]
            fam_rows.append({"family": s, "role": role, **compare(g[g.method == "paired_rff"].groupby("dataset").gap.median(),
                                                                  g[g.method == "moment"].groupby("dataset").gap.median())})
    R["family_effects"] = pd.DataFrame(fam_rows)
    R["family_effects"].to_csv(T / "paired_vs_moment_by_family.csv", index=False)

    # Robustness to noise, feature design and readout family (robust seed only).
    rrows = []
    for an, lab in ROBUST_ORDER:
        g = pr[(pr.analysis == an) & (pr.level == "cross")]
        if not len(g):
            continue
        cmpd = compare(g[g.method == "paired_rff"].groupby("dataset").gap.median(), g[g.method == "moment"].groupby("dataset").gap.median())
        rec = {"analysis": an, "label": lab, "median_feature_corr": float(g[g.method == "moment"].feature_corr_mean.median()),
               "median_full_refit_nba": float(g[g.method == "moment"].full_refit_nba.median()), **{f"paired_vs_moment_{k}": v for k, v in cmpd.items()}}
        for m in ["moment", "paired_rff"]:
            med, lo, hi = hier_boot_fast(g[g.method == m], "gap", stable_seed("rob", an, m))
            rec.update({f"{m}_gap": med, f"{m}_lo": lo, f"{m}_hi": hi})
        rrows.append(rec)
    R["robust"] = pd.DataFrame(rrows)
    R["robust"].to_csv(T / "robustness_noise_features_readout.csv", index=False)

    # Out-of-sample prediction of moment-matching loss from label-free feature correspondence.
    mm = main[main.method == "moment"]
    pred_src, R["lodo_source"] = lodo_prediction(mm, "feature_corr_mean", "gap")
    _, R["lodo_twin"] = lodo_prediction(mm, "twin_feature_corr", "gap")
    pred_src.to_csv(T / "lodo_gap_prediction.csv", index=False)
    R["lodo_rows"] = pred_src
    v3p = PROJECT_ROOT / V3_NAME / "tables" / "pair_level_diagnostics.csv"
    if v3p.exists():
        v3 = pd.read_csv(v3p)
        v3 = v3[v3.method == "moment"].dropna(subset=["feature_corr_mean", "gap"])
        coef = np.polyfit(v3.feature_corr_mean, v3.gap, 2)
        o = mm.dropna(subset=["feature_corr_mean", "gap"]).assign(predicted=lambda d: np.polyval(coef, d.feature_corr_mean))
        R["v3_to_v4"] = {"pooled_spearman": float(spearmanr(o.predicted, o.gap)[0]), "mae": float(np.mean(np.abs(o.predicted - o.gap))),
                         "within_dataset": rho_test(blocked_rhos(o, "predicted", "gap"), "greater"), "coefficients": coef.tolist()}
    else:
        R["v3_to_v4"] = {"note": f"{v3p} not found"}
    R["corr_scope"] = {"within_family_only": rho_test(blocked_rhos(mm[mm.level != "cross"], "feature_corr_mean", "gap"), "less"),
                       "cross_family_only": rho_test(blocked_rhos(mm[mm.level == "cross"], "feature_corr_mean", "gap"), "less"),
                       "cka_all": rho_test(blocked_rhos(mm, "cka", "gap"), "less"),
                       "param_distance_within": rho_test(blocked_rhos(mm[mm.level != "cross"], "param_distance", "gap"), "greater")}

    # Operating point decomposition.
    orows = []
    for col, lab in [("random", "Random"), ("duration_only", "Duration only"), ("gain_only", "Gain only"),
                     ("universal_default", "Universal default"), ("source_selected", "Source selected"), ("target_oracle", "Target oracle")]:
        med, lo, hi = hier_boot_fast(odp, col, stable_seed("od", col))
        orows.append({"policy": col, "label": lab, "median_target_nba": med, "ci95_low": lo, "ci95_high": hi})
    R["op"] = pd.DataFrame(orows)
    R["op"].to_csv(T / "operating_point_decomposition.csv", index=False)
    R["op_tests"] = {f"source_selected_vs_{c}": compare(odd.source_selected, odd[c], higher=True) for c in ["random", "duration_only", "gain_only"]}

    # Raw-baseline gap: noise versus deterministic distortion.
    q6 = q6.copy()
    q6["gain_noisy"] = q6.full_noisy_nba - q6.raw_nba
    q6["gain_noiseless"] = q6.full_noiseless_nba - q6.raw_nba
    q6.to_csv(T / "raw_baseline_noise_decomposition_rows.csv", index=False)
    qf = q6.groupby(["dataset", "family"], as_index=False).median(numeric_only=True)
    frows = []
    for f in FAMILY_CODES + ["ALL"]:
        g = qf if f == "ALL" else qf[qf.family == f]
        rec = {"family": f}
        for c in ["gain_noisy", "gain_noiseless", "feature_test_retest_r", "noise_vs_noiseless_feature_r", "raw_to_device_r2_noisy", "raw_to_device_r2_noiseless"]:
            med, lo, hi = hier_boot_fast(g, c, stable_seed("q6", f, c))
            rec.update({c: med, f"{c}_lo": lo, f"{c}_hi": hi})
        rec["fraction_noiseless_beats_raw"] = float(np.mean(g.gain_noiseless > 0))
        rec["fraction_noisy_beats_raw"] = float(np.mean(g.gain_noisy > 0))
        frows.append(rec)
    R["q6"] = pd.DataFrame(frows)
    R["q6"].to_csv(T / "raw_baseline_noise_decomposition.csv", index=False)
    qd = qf.groupby("dataset").median(numeric_only=True)
    R["q6_test"] = compare(qd.gain_noiseless, qd.gain_noisy, higher=True)

    # Dataset moderators.
    dm = pd.concat([dsv("cross", "moment").rename("moment_gap"), dsv("cross", "paired_rff").rename("paired_rff_gap")], axis=1)
    dm = dm.join(dsprops.set_index("dataset"))
    dm["paired_benefit"] = dm.moment_gap - dm.paired_rff_gap
    mod = {}
    for y in ["moment_gap", "paired_rff_gap", "paired_benefit"]:
        for x in ["log_length", "n_classes", "log_n_train"]:
            g = dm[[x, y]].dropna()
            r, p = spearmanr(g[x], g[y]) if len(g) > 4 else (np.nan, np.nan)
            mod[f"{y}_vs_{x}"] = {"rho": float(r), "p": float(p), "n": int(len(g))}
    R["moderators"] = mod
    dm.to_csv(T / "dataset_moderators.csv")
    R["capacity_extra"] = {"lf_product": cluster_boot_spearman(capm, "lf_product", "emulation_r2", ["source", "target"], stable_seed("lf")),
                           "source_total_capacity": cluster_boot_spearman(capm, "source_total_capacity", "emulation_r2", ["source", "target"], stable_seed("stc"))}
    R["pair"] = pair
    R["w"] = w
    return R


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def _ci_plot(ax, x, med, lo, hi, color, marker, label=None, ms=9):
    med, lo, hi = map(np.asarray, (med, lo, hi))
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt=marker, color=color, ms=ms, capsize=5, lw=2, label=label)


def make_v4_figures(R: Dict[str, Any]) -> None:
    summ = R["summary"]
    levels = R["levels"]
    lvl_lab = [f"{float(l.split('_')[1]) * 100:.0f}%\nwithin" for l in levels[:-1]] + ["Cross\nfamily"]

    # Figure 8: transfer ladder and level dependence.
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.8), gridspec_kw={"width_ratios": [1.7, 1.0]})
    ax = axes[0]
    s = summ[summ.level == "cross"].set_index("method")
    ms = [m for m, _ in LADDER if m in s.index]
    x = np.arange(len(ms))
    _ci_plot(ax, x, s.loc[ms, "median_gap"], s.loc[ms, "ci95_low"], s.loc[ms, "ci95_high"], "#1b6ca8", "o")
    ax.set_xticks(x); ax.set_xticklabels([dict(LADDER)[m] for m in ms], fontsize=13)
    ax.set_xlim(-0.6, len(ms) - 0.4)
    ax.axhline(0, color="0.4", ls=":", lw=1)
    set_bounds(ax, "y", min(0.0, float(s.loc[ms, "ci95_low"].min())), float(s.loc[ms, "ci95_high"].max()))
    ax.set_ylabel("Gap to full refit (normalized BA)")
    style_ax(ax)
    ax = axes[1]
    allv = [0.0]
    for (m, lab, col, mk), off in zip([("moment", "Moment matching", "#e08a1e", "^"), ("paired_rff", "Paired nonlinear", "#1b6ca8", "o"),
                                       ("twin_paired_rff", "Twin nonlinear", "#2a9d5c", "s"), ("pseudo_label", "Pseudo labels", "#7b4fa0", "D")],
                                      [-0.18, -0.06, 0.06, 0.18]):
        g = summ[summ.method == m].set_index("level").reindex(levels)
        _ci_plot(ax, np.arange(len(levels)) + off, g.median_gap, g.ci95_low, g.ci95_high, col, mk, lab, ms=8)
        allv += list(g.ci95_low.dropna()) + list(g.ci95_high.dropna())
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(lvl_lab, fontsize=13); ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.axhline(0, color="0.4", ls=":", lw=1)
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_ylabel("Gap to full refit (normalized BA)")
    style_ax(ax); outlined_legend(ax, loc="upper left", fontsize=12)
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig8_transfer_ladder")

    # Figure 9: pairing requirements.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.6))
    ax = axes[0]
    b = R["budget"]
    bc = b[b.level == "cross"]
    xs_map = {bb: math.log2(bb) for bb in BUDGETS}
    xs_map["all"] = math.log2(BUDGETS[-1]) + 1.5
    allv = [0.0]
    for kind, lab, col, mk in [("linear", "Paired linear", "#e08a1e", "^"), ("rff", "Paired nonlinear", "#1b6ca8", "o")]:
        g = bc[bc["map"] == kind]
        xx = [xs_map[v] for v in g.budget]
        ax.plot(xx, g.median_gap, color=col, lw=2)
        _ci_plot(ax, xx, g.median_gap, g.ci95_low, g.ci95_high, col, mk, lab)
        allv += list(g.ci95_low) + list(g.ci95_high)
    mref = summ[(summ.level == "cross") & (summ.method == "moment")]
    if len(mref):
        ax.axhline(float(mref.median_gap.iloc[0]), color="0.3", ls="--", lw=1.5, label="Moment matching")
        allv.append(float(mref.median_gap.iloc[0]))
    ticks = [xs_map[v] for v in BUDGETS] + [xs_map["all"]]
    ax.set_xticks(ticks); ax.set_xticklabels([str(v) for v in BUDGETS] + ["All"])
    ax.set_xlim(ticks[0] - 0.5, ticks[-1] + 0.5)
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_xlabel("Unlabeled paired sequences"); ax.set_ylabel("Cross-family gap to full refit")
    style_ax(ax); outlined_legend(ax, loc="best")
    ax = axes[1]
    a = R["alt"]
    labs = ["Task inputs", "Matched probe", "White probe", "Other datasets"]
    allv = [0.0]
    for kind, lab, col, mk, off in [("linear", "Paired linear", "#e08a1e", "^", -0.12), ("rff", "Paired nonlinear", "#1b6ca8", "o", 0.12)]:
        g = a[a["map"] == kind].set_index("pairing_inputs").reindex(labs)
        _ci_plot(ax, np.arange(len(labs)) + off, g.median_gap, g.ci95_low, g.ci95_high, col, mk, lab)
        allv += list(g.ci95_low.dropna()) + list(g.ci95_high.dropna())
    if len(mref):
        ax.axhline(float(mref.median_gap.iloc[0]), color="0.3", ls="--", lw=1.5, label="Moment matching")
        allv.append(float(mref.median_gap.iloc[0]))
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(["Task\ninputs", "Matched\nprobe", "White\nprobe", "Other\ndatasets"])
    ax.set_xlim(-0.5, len(labs) - 0.5)
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_ylabel("Cross-family gap to full refit")
    style_ax(ax); outlined_legend(ax, loc="upper left")
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig9_pairing_requirements")

    # Figure 10: family-pair matrices on a shared colour scale.
    mats = R["matrix"]
    vmax_raw = max(float(np.nanmax(M.to_numpy())) for M in mats.values())
    step = _nice_step(vmax_raw, 4)
    vmax = math.ceil(vmax_raw / step - 1e-9) * step
    vmin = min(0.0, math.floor(min(float(np.nanmin(M.to_numpy())) for M in mats.values()) / step) * step)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.4))
    for ax, (m, M) in zip(axes, mats.items()):
        A = M.to_numpy(dtype=float)
        cmap = plt.get_cmap("magma_r").copy(); cmap.set_bad("0.85")
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(FAMILY_CODES))); ax.set_xticklabels(FAMILY_CODES)
        ax.set_yticks(range(len(FAMILY_CODES))); ax.set_yticklabels(FAMILY_CODES)
        ax.set_xlabel("Target family"); ax.set_ylabel("Source family")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=12,
                            color="white" if (A[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55 else "black")
        style_ax(ax)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
        cb = fig.colorbar(im, cax=cax)
        tk = np.arange(vmin, vmax + step / 2, step)
        cb.set_ticks(tk); cb.ax.set_yticklabels([f"{v:.{_decimals(step)}f}" for v in tk])
        cb.set_label("Moment-matching gap" if m == "moment" else "Paired nonlinear gap")
        cb.outline.set_linewidth(SPINE_W)
    fig.tight_layout(w_pad=4); panel_letters(fig, axes); save_fig(fig, "fig10_family_pair_matrices")

    # Figure 11: robustness and out-of-sample prediction.
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.8), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    rb = R["robust"]
    allv = [0.0]
    if len(rb):
        x = np.arange(len(rb))
        for m, lab, col, mk, off in [("moment", "Moment matching", "#e08a1e", "^", -0.12), ("paired_rff", "Paired nonlinear", "#1b6ca8", "o", 0.12)]:
            _ci_plot(ax, x + off, rb[f"{m}_gap"], rb[f"{m}_lo"], rb[f"{m}_hi"], col, mk, lab)
            allv += list(rb[f"{m}_lo"]) + list(rb[f"{m}_hi"])
        ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in rb.label], fontsize=13); ax.set_xlim(-0.5, len(rb) - 0.5)
    ax.axhline(0, color="0.4", ls=":", lw=1)
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_ylabel("Cross-family gap to full refit")
    style_ax(ax); outlined_legend(ax, loc="upper left")
    ax = axes[1]
    o = R["lodo_rows"]
    if len(o):
        col = np.where(o.level == "cross", "0.45", "#1b6ca8")
        ax.scatter(o.predicted, o.gap, s=12, alpha=0.5, c=col)
        ax.scatter([], [], s=30, c="#1b6ca8", label="Within family"); ax.scatter([], [], s=30, c="0.45", label="Cross family")
        lo_ = min(float(o.predicted.min()), float(o.gap.min()), 0.0); hi_ = max(float(o.predicted.max()), float(o.gap.max()))
        ax.plot([lo_, hi_], [lo_, hi_], color="black", lw=1.2, ls="--")
        set_bounds(ax, "x", lo_, hi_); set_bounds(ax, "y", lo_, hi_)
    ax.set_xlabel("Predicted gap, held-out dataset"); ax.set_ylabel("Observed moment-matching gap")
    style_ax(ax); outlined_legend(ax, loc="upper left")
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig11_robustness_and_prediction")

    # Figure 12: operating point policies and noise decomposition of the raw-baseline gap.
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.6))
    ax = axes[0]
    op = R["op"]
    x = np.arange(len(op))
    _ci_plot(ax, x, op.median_target_nba, op.ci95_low, op.ci95_high, "#1b6ca8", "o")
    ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in op.label], fontsize=13); ax.set_xlim(-0.5, len(op) - 0.5)
    set_bounds(ax, "y", min(0.0, float(op.ci95_low.min())), float(op.ci95_high.max()))
    ax.set_ylabel("Target normalized BA")
    style_ax(ax)
    ax = axes[1]
    q = R["q6"]
    x = np.arange(len(q))
    allv = [0.0]
    for c, lab, col, mk, off in [("gain_noisy", "Nominal noise", "#c0392b", "s", -0.12), ("gain_noiseless", "Noise free", "#2a9d5c", "o", 0.12)]:
        _ci_plot(ax, x + off, q[c], q[f"{c}_lo"], q[f"{c}_hi"], col, mk, lab)
        allv += list(q[f"{c}_lo"].dropna()) + list(q[f"{c}_hi"].dropna())
    ax.axhline(0, color="0.4", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["All" if f == "ALL" else f for f in q.family]); ax.set_xlim(-0.5, len(q) - 0.5)
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_xlabel("Device family"); ax.set_ylabel("Gain over raw baseline (normalized BA)")
    style_ax(ax); outlined_legend(ax, loc="best")
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig12_operating_point_and_noise")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    log("=" * 100)
    log(f"MEMRISTIVE RESERVOIR PORTABILITY v4 CONFIRMATORY | root={ROOT} | smoke={SMOKE}")
    log("Drive mounting is intentionally not performed by this script.")
    for fam in FAMILY_CODES:
        T_SWITCH[fam] = calibrate_switch_time(fam)
        T_RECOVER[fam] = recovery_duration(fam)
    st = self_tests()
    st.to_csv(DIRS["tables"] / "solver_self_tests.csv", index=False)
    if st.abs_error.max() > 1e-3:
        raise RuntimeError("Closed-form solver self-test failed.")

    names = (["SmokeA", "SmokeB", "SmokeC"] if SMOKE else PRIOR_DATASETS + FRESH_DATASETS + HELD_OUT_DATASETS)
    if DATASET_FILTER:
        names = [d for d in names if d in DATASET_FILTER]
    protocol = {
        "script_version": SCRIPT_VERSION, "smoke": SMOKE, "datasets": names, "held_out_datasets": HELD_OUT_DATASETS,
        "device_seeds": V4_SEEDS, "seeds_used_in_earlier_runs": [11, 22, 33], "robustness_seed": ROBUST_SEED,
        "d2d_levels": V4_D2D, "budgets": BUDGETS, "k": K_V4, "rff_dim": RFF_DIM, "n_alt": N_ALT,
        "map_alphas": [MAP_ALPHAS_LIN, MAP_ALPHAS_RFF], "noise_levels": NOISE_LEVELS, "feature_variants": FEATURE_VARIANTS,
        "unit_of_replication": "dataset (pairs are seed medians, datasets are pair medians)",
        "confirmatory": {
            "C1": "cross family: paired nonlinear feature map gap < moment matching gap (one-sided Wilcoxon over datasets)",
            "C2": "cross family: paired nonlinear < paired linear (one-sided)",
            "C3": "cross family: map to a simulated nominal twin of the source family < moment matching (one-sided)",
            "C4": "within datasets, Spearman(feature correlation, moment gap) < 0 (one-sided Wilcoxon on rhos)",
            "C5": f"within datasets, Spearman(readout cosine, warm-start advantage at k={K_MAIN}) > 0 (one-sided)",
            "C6": "source-selected operating point gives higher target nBA than a random operating point (one-sided)",
            "C7a": "probe emulation R2 rises with target total capacity (Spearman, bootstrap over ordered family pairs)",
            "C7b": "probe emulation R2 rises with memory-profile similarity (Spearman, bootstrap over ordered family pairs)",
            "C8": "paired nonlinear map vs pseudo-label refit (two-sided)",
            "C9": "source-selected vs leave-one-dataset-out universal default operating point (two-sided)",
            "correction": "Holm over C1 to C9",
        },
        "exploratory": ["d2d sweep replication", "pairing budget curve", "pairing with matched probe, white probe or other datasets",
                        "transfer ladder (sign correction, diagonal, linear, nonlinear, score distillation)", "k-shot after pairing",
                        "family-pair matrices", "noise, feature-design and logistic-readout robustness", "leave-one-dataset-out gap prediction",
                        "operating-point decomposition", "noise versus distortion behind the raw-baseline gap", "dataset moderators"],
    }
    phash = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()[:12]
    ppath = DIRS["config"] / "protocol_v4.json"
    if ppath.exists() and json.loads(ppath.read_text()).get("protocol_hash") == phash:
        log(f"Protocol {phash} already registered at {json.loads(ppath.read_text()).get('registered_utc')}")
    else:
        protocol["protocol_hash"] = phash
        protocol["registered_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(protocol, ppath)
        log(f"Protocol {phash} registered before any confirmatory result was computed.")

    loaded = []
    for name in names:
        try:
            ds = synthetic_dataset(name, stable_seed(name)) if SMOKE else load_dataset(name)
        except Exception as exc:
            log(f"SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        if name in HELD_OUT_DATASETS or (SMOKE and name == "SmokeC"):
            ds.tier = "held_out"
        loaded.append(ds)
        log(f"{name}: tier={ds.tier} train={len(ds.y_train)} test={len(ds.y_test)} length={ds.U_train.shape[1]} classes={ds.n_classes}")
    loaded.sort(key=lambda d: (len(d.y_train) + len(d.y_test)) * d.U_train.shape[1])
    pool = {d.name: d.U_train for d in loaded}
    dsprops = pd.DataFrame([{"dataset": d.name, "log_length": math.log(d.U_train.shape[1]), "n_classes": d.n_classes,
                             "log_n_train": math.log(len(d.y_train))} for d in loaded])

    p_cap = DIRS["units"] / f"capacity_emulation_{phash}.csv"
    if p_cap.exists():
        capem = pd.read_csv(p_cap)
    else:
        log("Capacity and probe emulation on fresh device seeds ...")
        capem = capacity_emulation_analysis(V4_SEEDS)
        capem.to_csv(p_cap, index=False)

    U, Ld, Q = [], [], []
    for ds in loaded:
        for seed in V4_SEEDS:
            paths = [DIRS["units"] / f"{k}_{ds.name}_s{seed}_{phash}.csv" for k in ("units", "land", "q6")]
            if all(p.exists() for p in paths):
                U.append(pd.read_csv(paths[0])); Ld.append(pd.read_csv(paths[1])); Q.append(pd.read_csv(paths[2]))
                continue
            t0 = time.time()
            try:
                u, l, q = run_unit_v4(ds, seed, pool)
            except Exception:
                log(f"FAILED {ds.name} seed {seed}:\n{traceback.format_exc()}")
                continue
            for df, p in zip((u, l, q), paths):
                df.to_csv(p, index=False)
            U.append(u); Ld.append(l); Q.append(q)
            m = u[(u.analysis == "main") & (u.level == "cross")].groupby("method").gap.median()
            log(f"{ds.name} seed {seed}: {(time.time() - t0) / 60:.1f} min | cross gap moment={m.get('moment', np.nan):.3f} "
                f"paired_rff={m.get('paired_rff', np.nan):.3f} twin={m.get('twin_paired_rff', np.nan):.3f} pseudo={m.get('pseudo_label', np.nan):.3f}")
    if not U:
        raise RuntimeError("No units produced.")
    units, land, q6 = pd.concat(U, ignore_index=True), pd.concat(Ld, ignore_index=True), pd.concat(Q, ignore_index=True)
    units.to_csv(DIRS["tables"] / "all_units.csv", index=False)
    land.to_csv(DIRS["tables"] / "operating_point_landscape.csv", index=False)

    R = analyse_v4(units, land, q6, capem, dsprops)
    try:
        make_v4_figures(R)
    except Exception:
        log("Figure generation failed:\n" + traceback.format_exc())
    elapsed = (time.time() - t_start) / 60
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    J = lambda o: json.dumps(o, indent=1, default=str)
    print("\n" + "=" * 110)
    print("COPY-PASTE REPORT START - MEMRISTIVE RESERVOIR PORTABILITY v4 CONFIRMATORY")
    print("=" * 110)
    print(f"Runtime: {elapsed:.1f} min | protocol hash {phash} | datasets {len(loaded)} | seeds {V4_SEEDS} | robustness seed {ROBUST_SEED}")
    print("Tiers:", units.groupby("tier").dataset.nunique().to_dict())
    print("\nCONFIRMATORY TESTS (dataset level, Holm over C1 to C9)")
    conf = pd.DataFrame([{"test": k, **{c: v.get(c) for c in ["median_a", "median_b", "median_difference", "datasets_a_better", "datasets",
                                                                "mean_rho", "datasets_in_direction", "rho", "ci95_low", "ci95_high", "p", "p_holm"]}}
                         for k, v in R["confirmatory"].items()])
    print(conf.to_string(index=False))
    print("\nBY TIER:", J(R["by_tier"]))
    print("\nMETHOD SUMMARY (gap to full refit, normalized BA)")
    sm = R["summary"]
    keep = [m for m, _ in LADDER] + ["full_refit", "distill_linear", "twin_paired_linear"] + [f"{a}_k{k}" for k in K_V4 for a in ("scratch", "moment_warm", "paired_warm")]
    print(sm[sm.method.isin(keep)].sort_values(["level", "median_gap"]).to_string(index=False))
    print("\nABSOLUTE nBA BY LEVEL"); print(R["absolute"].to_string())
    print("\nDIAGNOSTICS BY LEVEL"); print(R["diag_by_level"].T.to_string())
    print("\nWITHIN FAMILY:", J(R["within_family"]))
    print("\nEXPLORATORY COMPARISONS:", J(R["exploratory"]))
    print("\nPAIRING BUDGET"); print(R["budget"].to_string(index=False))
    print("\nPAIRING INPUT SOURCE"); print(R["alt"].to_string(index=False))
    print("\nFAMILY MATRIX, MOMENT GAP"); print(R["matrix"]["moment"].to_string())
    print("\nFAMILY MATRIX, PAIRED NONLINEAR GAP"); print(R["matrix"]["paired_rff"].to_string())
    print("\nPAIRED VS MOMENT BY FAMILY"); print(R["family_effects"].to_string(index=False))
    print("\nROBUSTNESS (robustness seed, cross family)"); print(R["robust"].to_string(index=False))
    print("\nGAP PREDICTION FROM FEATURE CORRESPONDENCE")
    print("LODO, source features:", J(R["lodo_source"])); print("LODO, twin features:", J(R["lodo_twin"]))
    print("v3 fit applied to v4:", J(R["v3_to_v4"])); print("Scope:", J(R["corr_scope"]))
    print("\nOPERATING POINT POLICIES"); print(R["op"].to_string(index=False)); print(J(R["op_tests"]))
    print("\nRAW-BASELINE GAP: NOISE VS DISTORTION"); print(R["q6"][["family", "gain_noisy", "gain_noiseless", "feature_test_retest_r",
          "noise_vs_noiseless_feature_r", "raw_to_device_r2_noisy", "raw_to_device_r2_noiseless", "fraction_noisy_beats_raw",
          "fraction_noiseless_beats_raw"]].to_string(index=False))
    print("Noise-free gain > noisy gain:", J(R["q6_test"]))
    print("\nCAPACITY PREDICTORS (extra):", J(R["capacity_extra"]))
    print("\nDATASET MODERATORS:", J(R["moderators"]))
    print("=" * 110)
    print("COPY-PASTE REPORT END")
    print("=" * 110)
    log(f"Done in {elapsed:.1f} min. Tables: {DIRS['tables']} | Figures: {DIRS['figures']}")


if __name__ == "__main__":
    main()
