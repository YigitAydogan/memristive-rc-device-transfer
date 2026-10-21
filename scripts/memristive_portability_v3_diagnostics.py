"""
Memristive reservoir portability, v3: device-to-device sweep and mechanistic diagnostics.

Reads the finished v2 run (MyDrive/memristive_device_generalization/portability_v2) and explains its results:
  Q1  readout portability vs feature correspondence, sweeping within-family spread 5% to 50%;
  Q2  why label-free FIR alignment fails (white vs task-matched probe, probe-to-task R2 transfer,
      paired linear and paired nonlinear feature maps as upper bounds);
  Q3  why warm start loses to scratch (cosine between source and target readouts);
  Q4  why capacity did not predict emulation (nonlinear capacity vs read noise, memory-function deficit);
  Q5  why the operating point transfers (correlated, broad utility landscapes, shared good region);
  Q6  why devices do not beat the raw baseline (linear redundancy with raw binned features).

Paste into one Colab cell with Drive already mounted. The script does not mount Drive.
Environment variables (optional): PORTV2_SMOKE=1, PORTV2_DATASETS=a,b, MEMRISTIVE_PROJECT_ROOT=path.
Every dataset and seed unit is checkpointed, so an interrupted run resumes where it stopped.
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

SCRIPT_VERSION = "3.0.0"
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
RUN_NAME = "portability_v3_diagnostics" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else "")
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
# v3: device-to-device sweep and mechanistic diagnostics
# =============================================================================

V2_NAME = "portability_v2" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else "")
V2_ROOT = PROJECT_ROOT / V2_NAME
D2D_LEVELS = [0.05, 0.10, 0.20, 0.35, 0.50] if not SMOKE else [0.05, 0.35]
READ_NOISE_LEVELS = [0.0, 0.001, 0.003, 0.01, 0.03]
K_DIAG = 5 if not SMOKE else 2
RFF_DIM = 512 if not SMOKE else 128
CAP_T0 = max(CAP_MAX_DELAY_D1, CAP_STATE_LAGS)


def probe_inputs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("probe_input", seed, MASTER_SEED))
    return rng.uniform(-1.0, 1.0, size=(PROBE_EPISODES, PROBE_LENGTH))


def matched_probe_inputs(ds: "Dataset", seed: int) -> np.ndarray:
    """Task-agnostic probe whose lag-1 autocorrelation and marginal distribution match the unlabeled
    training inputs. No labels and no task sequences are used, only two input statistics."""
    U = ds.U_train
    a, b = U[:, :-1].reshape(-1), U[:, 1:].reshape(-1)
    rho = float(np.clip(np.corrcoef(a, b)[0, 1], -0.999, 0.999)) if np.std(a) > 0 else 0.0
    rng = np.random.default_rng(stable_seed("matched_probe", ds.name, seed, MASTER_SEED))
    E, L = PROBE_EPISODES, PROBE_LENGTH
    g = np.empty((E, L))
    g[:, 0] = rng.standard_normal(E)
    s = math.sqrt(max(1.0 - rho * rho, 1e-6))
    for t in range(1, L):
        g[:, t] = rho * g[:, t - 1] + s * rng.standard_normal(E)
    q = np.sort(U.reshape(-1))
    ranks = np.argsort(np.argsort(g.reshape(-1)))
    pos = (ranks + 0.5) / len(ranks) * (len(q) - 1)
    return q[pos.round().astype(int)].reshape(E, L)


def column_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-12)
    r = (A * B).mean(0)
    dead = (np.asarray(A).std(0) < 1e-9) | (np.asarray(B).std(0) < 1e-9)
    r[dead] = np.nan
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


def ridge_map(X: np.ndarray, Y: np.ndarray, alphas: Sequence[float], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-output ridge X -> Y with intercept; penalty chosen by 3-fold CV MSE."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    n = len(X)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 3, n)
    best, best_a = np.inf, alphas[0]
    for a in alphas:
        err = 0.0
        for f in range(3):
            tr, va = fold != f, fold == f
            if tr.sum() < 5 or va.sum() < 1:
                continue
            A = Xc[tr].T @ Xc[tr] + a * tr.sum() * np.eye(X.shape[1])
            W = np.linalg.solve(A, Xc[tr].T @ Yc[tr])
            err += np.sum((Yc[va] - Xc[va] @ W) ** 2)
        if err < best:
            best, best_a = err, a
    W = np.linalg.solve(Xc.T @ Xc + best_a * n * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, mx, my


def rff(Z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return math.sqrt(2.0 / W.shape[1]) * np.cos(Z @ W + b)


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


def make_instance_sigma(fam: str, seed: int, role: str, sigma: float) -> Instance:
    global SIGMA_D2D
    old = SIGMA_D2D
    SIGMA_D2D = sigma
    try:
        inst = make_instance(fam, seed, role)
    finally:
        SIGMA_D2D = old
    return inst


def parameter_distance(a: Instance, b: Instance) -> float:
    """Root-mean-square log-ratio between instance parameters (zero for identical devices)."""
    if a.family != b.family:
        return float("nan")
    vals = []
    for k, va in a.params.items():
        vb = b.params.get(k)
        if vb is None or va == 0 or vb == 0 or np.sign(va) != np.sign(vb) or k in ("x0", "lambda0"):
            continue
        vals.append(math.log(abs(vb) / abs(va)))
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0


# -----------------------------------------------------------------------------
# Per dataset and seed diagnostics
# -----------------------------------------------------------------------------


def run_diag_unit(ds: "Dataset", seed: int, sel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    C = ds.n_classes
    folds = make_folds(ds.y_train, stable_seed("folds", ds.name, seed))
    rng_k = np.random.default_rng(stable_seed("diag_k", ds.name, seed))
    Up_white = probe_inputs(seed)
    Up_match = matched_probe_inputs(ds, seed)
    E_tr = PROBE_TRAIN_EPISODES
    Fraw_tr, Fraw_te = features_from_traj(ds.U_train), features_from_traj(ds.U_test)
    mr, sr = std_stats(Fraw_tr)
    Zr_tr, Zr_te = (Fraw_tr - mr) / sr, (Fraw_te - mr) / sr
    rows, red_rows = [], []
    task_sub = np.random.default_rng(stable_seed("tsub", ds.name, seed)).permutation(len(ds.y_train))[
        :min(len(ds.y_train), max(32, int(200_000 / ds.U_train.shape[1])))]
    for s in FAMILY_CODES:
        r = sel[(sel.dataset == ds.name) & (sel.seed == seed) & (sel.instance == f"{s}_A")]
        if len(r) != 1:
            continue
        op_s = int(r.op.iloc[0])
        alpha_s = float(r.alpha_rel.iloc[0])
        src = make_instance(s, seed, "A")
        key_tr, key_te = ("task", ds.name, seed, "train"), ("task", ds.name, seed, "test")
        rs_tr = simulate(src, ds.U_train, [op_s], "traj", key_tr)[:, :, 0]
        rs_te = simulate(src, ds.U_test, [op_s], "traj", key_te)[:, :, 0]
        Fs_tr, Fs_te = features_from_traj(rs_tr), features_from_traj(rs_te)
        mu_s, sd_s = std_stats(Fs_tr)
        Zs_tr, Zs_te = (Fs_tr - mu_s) / sd_s, (Fs_te - mu_s) / sd_s
        ws = class_weights(ds.y_train, C).sum()
        ro_s = ridge_fit(Zs_tr, ds.y_train, C, alpha_s * ws)
        # Is the device representation linearly redundant with the raw-input representation?
        Wd, mxd, myd = ridge_map(Zr_tr, Zs_tr, [1e-4, 1e-3, 1e-2, 1e-1], stable_seed("red", ds.name, seed, s))
        red_rows.append({"dataset": ds.name, "tier": ds.tier, "seed": seed, "family": s, "op": op_s,
                         "raw_to_device_feature_r2": feature_r2(Zs_te, (Zr_te - mxd) @ Wd + myd)})
        pw_s = simulate(src, Up_white, [op_s], "traj", ("probe", seed))[:, :, 0]
        pm_s = simulate(src, Up_match, [op_s], "traj", ("mprobe", ds.name, seed))[:, :, 0]
        targets = [(make_instance(t, seed, "A"), "cross", float("nan")) for t in FAMILY_CODES if t != s]
        # The 5% level reuses the v2 "B" instance, so it reproduces the v2 within-family result exactly.
        targets += [(make_instance_sigma(s, seed, "B" if abs(sig - 0.05) < 1e-12 else f"B_sig{sig:.2f}", sig), "within", sig)
                    for sig in D2D_LEVELS]
        for tinst, ptype, sig in targets:
            rt_tr = simulate(tinst, ds.U_train, [op_s], "traj", key_tr)[:, :, 0]
            rt_te = simulate(tinst, ds.U_test, [op_s], "traj", key_te)[:, :, 0]
            Ft_tr, Ft_te = features_from_traj(rt_tr), features_from_traj(rt_te)
            mu_t, sd_t = std_stats(Ft_tr)
            Zt_tr, Zt_te = (Ft_tr - mu_t) / sd_t, (Ft_te - mu_t) / sd_t
            pad = rest_observable(tinst)
            res: Dict[str, float] = {}
            # Reference: full readout refit at the source operating point with the target's own CV penalty.
            _, ai_t, full_ba = select_and_test(Ft_tr, Ft_te, ds.y_train, ds.y_test, C, folds)
            res["full_refit"] = full_ba
            ro_t = ridge_fit(Zt_tr, ds.y_train, C, ALPHA_REL[ai_t] * ws)
            res["zero_shot"] = balanced_accuracy(ds.y_test, ro_s.predict((Ft_te - mu_s) / sd_s), C)
            res["moment"] = balanced_accuracy(ds.y_test, ro_s.predict(Zt_te), C)
            # Feature correspondence diagnostics on paired unlabeled training inputs.
            cc = column_corr(Fs_tr, Ft_tr)
            diag = {"feature_corr_mean": float(np.nanmean(cc)), "feature_corr_median_abs": float(np.nanmedian(np.abs(cc))),
                    "feature_sign_flip_fraction": float(np.nanmean(cc < 0)), "feature_dead_fraction": float(np.mean(np.isnan(cc))),
                    "cka": linear_cka(Zs_tr, Zt_tr)}
            ws_vec, wt_vec = ro_s.W.reshape(-1), ro_t.W.reshape(-1)
            diag["readout_cosine"] = float(ws_vec @ wt_vec / max(np.linalg.norm(ws_vec) * np.linalg.norm(wt_vec), 1e-20))
            # Upper bounds for label-free alignment: paired linear and paired nonlinear feature maps.
            Wl, mxl, myl = ridge_map(Zt_tr, Zs_tr, [1e-4, 1e-3, 1e-2, 1e-1, 1.0], stable_seed("lin", ds.name, seed, s, tinst.iid))
            Zhat = (Zt_te - mxl) @ Wl + myl
            diag["paired_linear_feature_r2"] = feature_r2(Zs_te, Zhat)
            res["paired_linear_map"] = balanced_accuracy(ds.y_test, ro_s.predict(Zhat), C)
            rr = np.random.default_rng(stable_seed("rff", ds.name, seed, s, tinst.iid))
            Wr = rr.standard_normal((Zt_tr.shape[1], RFF_DIM)) / math.sqrt(Zt_tr.shape[1])
            br = rr.uniform(0, 2 * math.pi, RFF_DIM)
            Wn, mxn, myn = ridge_map(rff(Zt_tr, Wr, br), Zs_tr, [1e-4, 1e-3, 1e-2, 1e-1], stable_seed("rffm", ds.name, seed, s, tinst.iid))
            Zhat_n = (rff(Zt_te, Wr, br) - mxn) @ Wn + myn
            diag["paired_rff_feature_r2"] = feature_r2(Zs_te, Zhat_n)
            res["paired_rff_map"] = balanced_accuracy(ds.y_test, ro_s.predict(Zhat_n), C)
            # Alignment failure chain for trajectory-level FIR maps.
            for pname, Ps, Up, pkey in [("white", pw_s, Up_white, ("probe", seed)), ("matched", pm_s, Up_match, ("mprobe", ds.name, seed))]:
                Pt = simulate(tinst, Up, [op_s], "traj", pkey)[:, :, 0]
                mp = fit_lag_map(Ps[:E_tr], Pt[:E_tr], pad, FIR_TAPS, False)
                diag[f"{pname}_probe_r2_on_probe"] = r2_score_flat(Ps[E_tr:], mp.apply(Pt[E_tr:]))
                diag[f"{pname}_probe_r2_on_task"] = r2_score_flat(rs_tr[task_sub], mp.apply(rt_tr[task_sub]))
                fa_tr, fa_te = features_from_traj(mp.apply(rt_tr)), features_from_traj(mp.apply(rt_te))
                ma, sa = std_stats(fa_tr)
                res[f"{pname}_probe_fir"] = balanced_accuracy(ds.y_test, ro_s.predict((fa_te - mu_s) / sd_s), C)
                res[f"{pname}_probe_fir_mm"] = balanced_accuracy(ds.y_test, ro_s.predict((fa_te - ma) / sa), C)
            cut = max(1, int(0.75 * len(task_sub)))
            mp = fit_lag_map(rs_tr[task_sub[:cut]], rt_tr[task_sub[:cut]], pad, FIR_TAPS, False)
            diag["replay_r2_on_task"] = r2_score_flat(rs_tr[task_sub[cut:]], mp.apply(rt_tr[task_sub[cut:]])) if cut < len(task_sub) else float("nan")
            # Warm start versus scratch at k labels per class, in the moment-matched space.
            idx = np.concatenate([rng_k.permutation(np.flatnonzero(ds.y_train == c))[:K_DIAG] for c in range(C)])
            res["scratch_k"] = balanced_accuracy(ds.y_test, ridge_fit(Zt_tr[idx], ds.y_train[idx], C, LAMBDA_PRIOR).predict(Zt_te), C)
            res["moment_warm_k"] = balanced_accuracy(ds.y_test, ridge_fit(Zt_tr[idx], ds.y_train[idx], C, LAMBDA_PRIOR, prior=ro_s.W).predict(Zt_te), C)
            base = {"dataset": ds.name, "tier": ds.tier, "seed": seed, "source": s, "target": tinst.family,
                    "pair_type": ptype, "d2d_sigma": sig, "param_distance": parameter_distance(src, tinst),
                    "op": op_s, "full_refit_nba": nba(full_ba, C), **diag}
            for mth, ba in res.items():
                rows.append({**base, "method": mth, "ba": ba, "nba": nba(ba, C), "gap": nba(full_ba, C) - nba(ba, C)})
    return pd.DataFrame(rows), pd.DataFrame(red_rows)


# -----------------------------------------------------------------------------
# Dataset-independent diagnostics
# -----------------------------------------------------------------------------


def capacity_noise_ablation() -> pd.DataFrame:
    """Nonlinear capacity of nominal devices as a function of read noise (cycle-to-cycle noise off)."""
    global SIGMA_READ, SIGMA_C2C
    old_r, old_c = SIGMA_READ, SIGMA_C2C
    rows = []
    Up = probe_inputs(SEEDS[0])
    Yc, deg = capacity_targets(Up, CAP_T0)
    try:
        SIGMA_C2C = 0.0
        for lvl in READ_NOISE_LEVELS:
            SIGMA_READ = lvl
            for fam in FAMILY_CODES:
                tr = simulate(nominal_instance(fam), Up, range(N_OPS), "traj", ("capnoise", lvl))
                for op in range(N_OPS):
                    c = capacities(tr[:, :, op], Yc, deg, CAP_T0)
                    rows.append({"read_noise_decades": lvl, "family": fam, "op": op, **c,
                                 "C_nonlinear": c["C2"] + c["C3"],
                                 "observable_std": float(np.std(tr[:, CAP_T0:, op]))})
    finally:
        SIGMA_READ, SIGMA_C2C = old_r, old_c
    return pd.DataFrame(rows)


def memory_deficit_analysis() -> pd.DataFrame:
    """Per-delay linear memory functions and emulation quality for every ordered cross-family pair."""
    rows = []
    for seed in SEEDS:
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
                    r2 = r2_score_flat(ps[PROBE_TRAIN_EPISODES:], mp.apply(pt[PROBE_TRAIN_EPISODES:]))
                    ms, mt = prof[s][op][d1], prof[t][op][d1]
                    nls, nlt = prof[s][op][deg > 1].sum(), prof[t][op][deg > 1].sum()
                    tot_s, tot_t = prof[s][op].sum(), prof[t][op].sum()
                    rows.append({"seed": seed, "source": s, "target": t, "op": op, "emulation_r2": r2,
                                 "linear_memory_deficit": float(np.sum(np.maximum(ms - mt, 0.0))),
                                 "nonlinear_deficit": float(max(nls - nlt, 0.0)),
                                 "memory_profile_corr": float(np.corrcoef(ms, mt)[0, 1]) if ms.std() > 0 and mt.std() > 0 else float("nan"),
                                 "lf_product": (ms.sum() / tot_s if tot_s > 0 else np.nan) * (mt.sum() / tot_t if tot_t > 0 else np.nan),
                                 "source_total_capacity": tot_s, "target_total_capacity": tot_t})
    return pd.DataFrame(rows)


def landscape_analysis(land: pd.DataFrame, v2units: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    la = land[land.role == "A"].copy()
    corr_rows = []
    for (dname, seed), g in la.groupby(["dataset", "seed"]):
        piv = g.pivot(index="op", columns="family", values="test_nba")
        for s in FAMILY_CODES:
            for t in FAMILY_CODES:
                if s != t and s in piv and t in piv and piv[s].std() > 0 and piv[t].std() > 0:
                    corr_rows.append({"dataset": dname, "seed": seed, "source": s, "target": t,
                                      "landscape_spearman": float(spearmanr(piv[s], piv[t])[0])})
    lc = pd.DataFrame(corr_rows)
    u = v2units[(v2units.method == "full_refit") & (v2units.pair_type == "cross")].copy()
    u["op_regret"] = u.oracle_nba - u.target_test_nba_at_source_op_phase1
    lc = lc.merge(u[["dataset", "seed", "source", "target", "op_regret"]], on=["dataset", "seed", "source", "target"], how="left")
    la["rank_norm"] = la.groupby(["dataset", "seed", "family"]).test_nba.rank(pct=True)
    region = la.groupby(["gain", "frac"], as_index=False).rank_norm.mean()
    near = la.groupby(["dataset", "seed", "family"]).apply(
        lambda g: float(np.mean(g.test_nba >= g.test_nba.max() - 0.02))).reset_index(name="fraction_within_0p02_of_best")
    return lc, region, near


# -----------------------------------------------------------------------------
# Analysis and figures
# -----------------------------------------------------------------------------


def dataset_blocked_spearman(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    rhos = []
    for _, g in df.groupby("dataset"):
        g = g[[x, y]].dropna()
        if len(g) >= 5 and g[x].std() > 0 and g[y].std() > 0:
            rhos.append(float(spearmanr(g[x], g[y])[0]))
    rhos = np.array(rhos)
    if len(rhos) < 3:
        return {"mean_rho": float("nan"), "datasets": int(len(rhos)), "p_two_sided": float("nan")}
    p = float(wilcoxon(rhos).pvalue) if np.any(rhos != 0) else 1.0
    return {"mean_rho": float(rhos.mean()), "median_rho": float(np.median(rhos)), "datasets": int(len(rhos)), "p_two_sided": p}


def analyse_v3(units: pd.DataFrame, red: pd.DataFrame, capn: pd.DataFrame, mem: pd.DataFrame,
               lc: pd.DataFrame, region: pd.DataFrame, near: pd.DataFrame, absperf: Optional[pd.DataFrame],
               land: pd.DataFrame) -> Dict[str, Any]:
    rep: Dict[str, Any] = {}
    diag_cols = ["feature_corr_mean", "feature_corr_median_abs", "feature_sign_flip_fraction", "feature_dead_fraction", "cka",
                 "readout_cosine", "paired_linear_feature_r2", "paired_rff_feature_r2", "white_probe_r2_on_probe",
                 "white_probe_r2_on_task", "matched_probe_r2_on_probe", "matched_probe_r2_on_task", "replay_r2_on_task",
                 "param_distance"]
    units["level"] = np.where(units.pair_type == "cross", "cross", "d2d_" + units.d2d_sigma.map(lambda v: f"{v:.2f}"))
    pair = units.groupby(["dataset", "tier", "source", "target", "level", "method"], as_index=False)[["gap", "nba"] + diag_cols].median()
    pair.to_csv(DIRS["tables"] / "pair_level_diagnostics.csv", index=False)
    dsl = pair.groupby(["dataset", "level", "method"], as_index=False)[["gap"] + diag_cols].median()
    dsl.to_csv(DIRS["tables"] / "dataset_level_diagnostics.csv", index=False)
    levels = [f"d2d_{v:.2f}" for v in D2D_LEVELS] + ["cross"]

    # Sweep summary: gap to full refit by method and level.
    srows = []
    for lvl in levels:
        for mth in ["zero_shot", "moment", "white_probe_fir_mm", "matched_probe_fir_mm", "paired_linear_map", "paired_rff_map", "scratch_k", "moment_warm_k"]:
            sub = pair[(pair.level == lvl) & (pair.method == mth)]
            if len(sub) == 0:
                continue
            med, lo, hi = hier_boot_ci(sub, "gap", stable_seed("v3boot", lvl, mth))
            srows.append({"level": lvl, "method": mth, "median_gap": med, "ci95_low": lo, "ci95_high": hi})
    sweep = pd.DataFrame(srows)
    sweep.to_csv(DIRS["tables"] / "d2d_sweep_summary.csv", index=False)
    rep["sweep"] = sweep

    # Diagnostic medians by level.
    dl = pair[pair.method == "moment"].groupby("level")[diag_cols].median().reindex(levels)
    dl.to_csv(DIRS["tables"] / "diagnostics_by_level.csv")
    rep["diag_by_level"] = dl

    # Why moment matching fails across families: feature correspondence predicts the gap.
    mom = pair[pair.method == "moment"].copy()
    rep["why_readout"] = {
        "corr_vs_moment_gap_all_levels": dataset_blocked_spearman(mom, "feature_corr_mean", "gap"),
        "cka_vs_moment_gap_all_levels": dataset_blocked_spearman(mom, "cka", "gap"),
        "corr_vs_moment_gap_cross_only": dataset_blocked_spearman(mom[mom.level == "cross"], "feature_corr_mean", "gap"),
        "param_distance_vs_moment_gap_within": dataset_blocked_spearman(mom[mom.level != "cross"], "param_distance", "gap"),
    }
    try:
        import statsmodels.formula.api as smf
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("gap ~ feature_corr_mean + cka", mom.dropna(subset=["gap", "feature_corr_mean", "cka"]), groups=mom.dropna(subset=["gap", "feature_corr_mean", "cka"])["dataset"]).fit()
        rep["why_readout"]["mixed_model"] = {k: {"coef": float(fit.params[k]), "p": float(fit.pvalues[k])} for k in ["feature_corr_mean", "cka"]}
    except Exception as exc:
        rep["why_readout"]["mixed_model"] = {"error": str(exc)}

    # Why warm start fails across families: readout cosine predicts the warm-start advantage.
    w = pair[pair.method == "moment_warm_k"][["dataset", "source", "target", "level", "gap", "readout_cosine"]].merge(
        pair[pair.method == "scratch_k"][["dataset", "source", "target", "level", "gap"]], on=["dataset", "source", "target", "level"],
        suffixes=("_warm", "_scratch"))
    w["warm_advantage"] = w.gap_scratch - w.gap_warm
    w.to_csv(DIRS["tables"] / "warm_start_vs_readout_cosine.csv", index=False)
    rep["why_warm"] = {"cosine_vs_warm_advantage": dataset_blocked_spearman(w, "readout_cosine", "warm_advantage"),
                       "median_cosine_cross": float(w[w.level == "cross"].readout_cosine.median()),
                       "median_cosine_within": float(w[w.level != "cross"].readout_cosine.median()),
                       "median_warm_advantage_cross": float(w[w.level == "cross"].warm_advantage.median()),
                       "median_warm_advantage_within": float(w[w.level != "cross"].warm_advantage.median())}

    # Why label-free alignment fails: the failure chain on cross-family pairs.
    cr = pair[(pair.level == "cross") & (pair.method == "moment")]
    chain = pd.DataFrame([{"stage": k, "median": float(cr[k].median()), "q25": float(cr[k].quantile(0.25)), "q75": float(cr[k].quantile(0.75))}
                          for k in ["white_probe_r2_on_probe", "white_probe_r2_on_task", "matched_probe_r2_on_probe", "matched_probe_r2_on_task",
                                    "replay_r2_on_task", "paired_linear_feature_r2", "paired_rff_feature_r2"]])
    chain.to_csv(DIRS["tables"] / "alignment_failure_chain.csv", index=False)
    rep["chain"] = chain
    crg = pair[pair.level == "cross"].pivot_table(index=["dataset", "source", "target"], columns="method", values="gap").reset_index()
    cmp_rows = []
    for a, b in [("matched_probe_fir_mm", "white_probe_fir_mm"), ("paired_linear_map", "moment"), ("paired_rff_map", "paired_linear_map"),
                 ("paired_rff_map", "moment")]:
        d = crg.groupby("dataset")[[a, b]].median()
        cmp_rows.append({"better": a, "reference": b, "median_gap_better": float(d[a].median()), "median_gap_reference": float(d[b].median()),
                         "datasets_improved": int((d[b] - d[a] > 0).sum()), "datasets": int(len(d)),
                         "p_one_sided": wilcoxon_less(d[a].to_numpy(), d[b].to_numpy())[0]})
    rep["alignment_comparisons"] = pd.DataFrame(cmp_rows)
    rep["alignment_comparisons"].to_csv(DIRS["tables"] / "alignment_comparisons.csv", index=False)

    # Why capacity did not predict emulation: range restriction under read noise, and a better predictor.
    capsum = capn.groupby(["read_noise_decades", "family"], as_index=False)[["C1", "C_nonlinear", "linear_fraction"]].median()
    capsum.to_csv(DIRS["tables"] / "capacity_vs_read_noise.csv", index=False)
    rep["capacity_noise"] = capsum.pivot(index="family", columns="read_noise_decades", values="C_nonlinear")
    mem.to_csv(DIRS["tables"] / "memory_deficit_vs_emulation.csv", index=False)
    mp = mem.groupby(["source", "target", "op"], as_index=False).median(numeric_only=True)
    preds = {}
    for p in ["lf_product", "linear_memory_deficit", "nonlinear_deficit", "memory_profile_corr", "target_total_capacity"]:
        g = mp[[p, "emulation_r2"]].dropna()
        preds[p] = {"rho": float(spearmanr(g[p], g.emulation_r2)[0]) if len(g) > 5 else float("nan"), "n": int(len(g))}
    rep["emulation_predictors"] = preds

    # Why the operating point transfers.
    lc.to_csv(DIRS["tables"] / "landscape_correlation.csv", index=False)
    region.to_csv(DIRS["tables"] / "good_region_map.csv", index=False)
    near.to_csv(DIRS["tables"] / "landscape_breadth.csv", index=False)
    lcp = lc.groupby(["dataset", "source", "target"], as_index=False)[["landscape_spearman", "op_regret"]].median()
    rep["why_oppoint"] = {"median_landscape_spearman": float(lcp.landscape_spearman.median()),
                          "landscape_corr_vs_op_regret": dataset_blocked_spearman(lcp, "landscape_spearman", "op_regret"),
                          "median_fraction_of_grid_within_0p02_of_best": float(near.fraction_within_0p02_of_best.median()),
                          "breadth_by_family": near.groupby("family").fraction_within_0p02_of_best.median().to_dict()}

    # Why devices do not beat the raw baseline: linear redundancy with raw features.
    red.to_csv(DIRS["tables"] / "device_raw_redundancy.csv", index=False)
    rr = red.groupby(["dataset", "family"], as_index=False).raw_to_device_feature_r2.median()
    out = {"median_raw_to_device_r2": float(rr.raw_to_device_feature_r2.median()),
           "by_family": rr.groupby("family").raw_to_device_feature_r2.median().to_dict()}
    if absperf is not None:
        long = []
        for _, r in absperf.iterrows():
            for f in FAMILY_CODES:
                col = f"oracle_{f}_ba"
                if col in absperf:
                    long.append({"dataset": r.dataset, "family": f, "gain_over_raw": nba(r[col], int(r.classes)) - nba(r.raw_binned_ridge_ba, int(r.classes))})
        g = rr.merge(pd.DataFrame(long), on=["dataset", "family"])
        g.to_csv(DIRS["tables"] / "redundancy_vs_gain_over_raw.csv", index=False)
        out["redundancy_vs_gain"] = dataset_blocked_spearman(g, "raw_to_device_feature_r2", "gain_over_raw")
        out["median_gain_over_raw"] = float(g.gain_over_raw.median())
    rep["why_raw"] = out
    return {"rep": rep, "pair": pair, "w": w, "capn": capsum, "mem": mp, "region": region, "lcp": lcp, "rr": rr}


def make_v3_figures(res: Dict[str, Any]) -> None:
    pair, w, capn, mem, region = res["pair"], res["w"], res["capn"], res["mem"], res["region"]
    sweep = res["rep"]["sweep"]
    levels = [f"d2d_{v:.2f}" for v in D2D_LEVELS]
    xs = np.array(D2D_LEVELS) * 100
    # Figure 4: sweep and the unifying feature-correspondence relation.
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))
    ax = axes[0]
    allv = [0.0]
    styles = [("zero_shot", "Zero shot", "#c0392b", "s"), ("moment", "Moment matching", "#e08a1e", "^"),
              ("white_probe_fir_mm", "Probe FIR + moment", "#1b6ca8", "o"), ("scratch_k", f"Scratch, {K_DIAG} labels per class", "#7b4fa0", "D")]
    for mth, lab, col, mk in styles:
        sub = sweep[sweep.method == mth].set_index("level")
        if not all(l in sub.index for l in levels):
            continue
        m = sub.loc[levels, "median_gap"].to_numpy(); lo = sub.loc[levels, "ci95_low"].to_numpy(); hi = sub.loc[levels, "ci95_high"].to_numpy()
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=9, lw=2, capsize=5, label=lab)
        allv += list(lo) + list(hi)
        if "cross" in sub.index:
            ax.axhline(sub.loc["cross", "median_gap"], color=col, ls="--", lw=1.4)
            allv.append(sub.loc["cross", "median_gap"])
    ax.plot([], [], color="0.3", ls="--", lw=1.4, label="Cross family (dashed)")
    set_bounds(ax, "x", 0, max(xs))
    set_bounds(ax, "y", min(allv), max(allv))
    ax.set_xlabel("Device-to-device parameter spread (%)")
    ax.set_ylabel("Gap to full readout refit (normalized BA)")
    style_ax(ax); outlined_legend(ax, loc="best")
    ax = axes[1]
    mom = pair[pair.method == "moment"]
    _pl = plt.get_cmap("plasma")
    cmap = lambda v: _pl(0.05 + 0.75 * v)
    for i, lvl in enumerate(levels):
        s = mom[mom.level == lvl]
        ax.scatter(s.feature_corr_mean, s.gap, s=18, alpha=0.6, color=cmap(i / max(1, len(levels) - 1)), label=f"Within, {D2D_LEVELS[i] * 100:.0f}%")
    s = mom[mom.level == "cross"]
    ax.scatter(s.feature_corr_mean, s.gap, s=18, alpha=0.5, color="0.45", label="Cross family")
    set_bounds(ax, "x", min(-0.2, float(mom.feature_corr_mean.min())), 1.0)
    set_bounds(ax, "y", min(0.0, float(mom.gap.min())), float(mom.gap.max()))
    ax.set_xlabel("Mean feature correlation, source vs target")
    ax.set_ylabel("Moment-matching gap (normalized BA)")
    style_ax(ax); outlined_legend(ax, loc="best", fontsize=11)
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig4_d2d_sweep_and_feature_correspondence")

    # Figure 5: alignment failure chain and warm start.
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    cr = pair[(pair.level == "cross") & (pair.method == "moment")]
    stages = [("white_probe_r2_on_probe", "White probe,\nprobe"), ("white_probe_r2_on_task", "White probe,\ntask"),
              ("matched_probe_r2_on_task", "Matched probe,\ntask"), ("replay_r2_on_task", "Replay,\ntask"),
              ("paired_linear_feature_r2", "Paired\nlinear"), ("paired_rff_feature_r2", "Paired\nnonlinear")]
    data = [cr[k].dropna().clip(lower=-1.0).to_numpy() for k, _ in stages]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor("#cfe0f1"); b.set_edgecolor("black"); b.set_linewidth(1.2)
    for el in ["whiskers", "caps", "medians"]:
        for l in bp[el]:
            l.set_color("black"); l.set_linewidth(1.4)
    ax.set_xticks(range(1, len(stages) + 1)); ax.set_xticklabels([l for _, l in stages], fontsize=12)
    ax.set_xlim(0.4, len(stages) + 0.6)
    wl = [float(np.min(l.get_ydata())) for l in bp["whiskers"]]
    lo = min(min(wl) if wl else 0.0, 0.0) - 0.01
    set_bounds(ax, "y", lo, 1.0)
    ax.axhline(0, color="0.4", ls=":", lw=1)
    ax.set_ylabel("Cross-family emulation $R^2$")
    style_ax(ax)
    ax = axes[1]
    for lvl, col, lab in [("cross", "0.45", "Cross family")] + [(l, cmap(i / max(1, len(levels) - 1)), f"Within, {D2D_LEVELS[i] * 100:.0f}%") for i, l in enumerate(levels)]:
        s = w[w.level == lvl]
        ax.scatter(s.readout_cosine, s.warm_advantage, s=18, alpha=0.6, color=col, label=lab)
    ax.axhline(0, color="0.4", ls=":", lw=1)
    set_bounds(ax, "x", min(-0.5, float(w.readout_cosine.min())), 1.0)
    set_bounds(ax, "y", float(w.warm_advantage.min()), float(w.warm_advantage.max()))
    ax.set_xlabel("Cosine, source readout vs target readout")
    ax.set_ylabel("Warm-start advantage (normalized BA)")
    style_ax(ax); outlined_legend(ax, loc="best", fontsize=11)
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig5_alignment_failure_and_warm_start")

    # Figure 6: capacity under read noise and emulation predictors.
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))
    ax = axes[0]
    allv = [0.0]
    lv = np.array(READ_NOISE_LEVELS)
    xpos = np.arange(len(lv))
    for fam in FAMILY_CODES:
        s = capn[capn.family == fam].sort_values("read_noise_decades")
        ax.plot(xpos, s.C_nonlinear, marker="o", ms=8, lw=2, color=FAM_COLORS[fam], label=fam)
        allv += list(s.C_nonlinear)
    ax.set_xticks(xpos); ax.set_xticklabels([f"{v:g}" for v in lv]); ax.set_xlim(-0.3, len(lv) - 0.7)
    set_bounds(ax, "y", 0, max(allv))
    ax.set_xlabel("Read noise (decades)"); ax.set_ylabel("Nonlinear capacity, degrees 2 and 3")
    style_ax(ax); outlined_legend(ax, loc="best", ncol=2)
    ax = axes[1]
    g = mem.dropna(subset=["linear_memory_deficit", "emulation_r2"])
    ax.scatter(g.linear_memory_deficit, g.emulation_r2.clip(lower=-1), s=14, alpha=0.5, color="#1b6ca8")
    set_bounds(ax, "x", 0, float(g.linear_memory_deficit.max()) if len(g) else 1)
    set_bounds(ax, "y", float(g.emulation_r2.clip(lower=-1).min()) if len(g) else -1, 1.0)
    ax.set_xlabel("Linear memory deficit, source minus target")
    ax.set_ylabel("Probe emulation $R^2$")
    style_ax(ax)
    fig.tight_layout(w_pad=3); panel_letters(fig, axes); save_fig(fig, "fig6_capacity_noise_and_memory_deficit")

    # Figure 7: shared good region of the operating grid.
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    piv = region.pivot(index="gain", columns="frac", values="rank_norm").sort_index(ascending=False)
    im = ax.imshow(piv.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    ax.set_xlabel("Drive duration / switching time"); ax.set_ylabel("Input gain")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11, color="white" if v < 0.6 else "black")
    style_ax(ax)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Mean percentile rank of target utility")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0]); cb.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cb.outline.set_linewidth(SPINE_W)
    fig.tight_layout(); save_fig(fig, "fig7_shared_operating_region")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    log("=" * 100)
    log(f"MEMRISTIVE RESERVOIR PORTABILITY v3 DIAGNOSTICS | root={ROOT} | reads v2 from {V2_ROOT}")
    log("Drive mounting is intentionally not performed by this script.")
    land_path = V2_ROOT / "tables" / "operating_point_landscape_full.csv"
    units_path = V2_ROOT / "tables" / "all_units.csv"
    if not land_path.exists() or not units_path.exists():
        raise FileNotFoundError(f"v2 results not found under {V2_ROOT}. Run the v2 script first.")
    land = pd.read_csv(land_path)
    v2units = pd.read_csv(units_path)
    abs_path = V2_ROOT / "tables" / "absolute_performance.csv"
    absperf = pd.read_csv(abs_path) if abs_path.exists() else None
    sel = land[land.selected.astype(bool)][["dataset", "seed", "instance", "op", "alpha_rel"]]

    for fam in FAMILY_CODES:
        T_SWITCH[fam] = calibrate_switch_time(fam)
        T_RECOVER[fam] = recovery_duration(fam)
    protocol = {"script_version": "3.0.0", "reads_v2_from": str(V2_ROOT), "d2d_levels": D2D_LEVELS,
                "read_noise_levels": READ_NOISE_LEVELS, "k_diag": K_DIAG, "rff_dim": RFF_DIM, "seeds": SEEDS,
                "questions": {
                    "Q1": "Is readout portability governed by feature correspondence (mean feature correlation, CKA), across D2D levels and families?",
                    "Q2": "Does label-free FIR alignment fail because of probe-task excitation mismatch, nonlinearity of the device-to-device map, or information loss? (white vs matched probe, paired linear vs paired nonlinear feature maps)",
                    "Q3": "Does warm start fail because source and target readouts point in different directions (readout cosine)?",
                    "Q4": "Did capacity fail to predict emulation because read noise suppresses nonlinear capacity, and is the linear memory deficit a better predictor?",
                    "Q5": "Does the operating point transfer because utility landscapes are correlated and broad?",
                    "Q6": "Do devices fail to beat the raw baseline because their features are linearly redundant with raw-input features?"},
                "written_utc": datetime.now(timezone.utc).isoformat()}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps({k: v for k, v in protocol.items() if k != "written_utc"}, sort_keys=True, default=str).encode()).hexdigest()[:12]
    save_json(protocol, DIRS["config"] / "protocol_v3.json")
    phash = protocol["protocol_hash"]
    log(f"Protocol hash {phash}")

    p_cap = DIRS["units"] / f"capacity_noise_{phash}.csv"
    if p_cap.exists():
        capn = pd.read_csv(p_cap)
    else:
        log("Capacity versus read-noise ablation ...")
        capn = capacity_noise_ablation(); capn.to_csv(p_cap, index=False)
    p_mem = DIRS["units"] / f"memory_deficit_{phash}.csv"
    if p_mem.exists():
        mem = pd.read_csv(p_mem)
    else:
        log("Memory-function deficit analysis ...")
        mem = memory_deficit_analysis(); mem.to_csv(p_mem, index=False)
    lc, region, near = landscape_analysis(land, v2units)

    datasets = sorted(sel.dataset.unique())
    if DATASET_FILTER:
        datasets = [d for d in datasets if d in DATASET_FILTER]
    frames, reds = [], []
    loaded = []
    for name in datasets:
        try:
            ds = synthetic_dataset(name, stable_seed(name)) if SMOKE else load_dataset(name)
            loaded.append(ds)
        except Exception as exc:
            log(f"SKIP {name}: {type(exc).__name__}: {exc}")
    loaded.sort(key=lambda d: (len(d.y_train) + len(d.y_test)) * d.U_train.shape[1])
    for ds in loaded:
        for seed in SEEDS:
            up = DIRS["units"] / f"diag_{ds.name}_s{seed}_{phash}.csv"
            rp = DIRS["units"] / f"red_{ds.name}_s{seed}_{phash}.csv"
            if up.exists() and rp.exists():
                frames.append(pd.read_csv(up)); reds.append(pd.read_csv(rp)); continue
            t0 = time.time()
            try:
                u, r = run_diag_unit(ds, seed, sel)
            except Exception:
                log(f"FAILED {ds.name} seed {seed}:\n{traceback.format_exc()}")
                continue
            u.to_csv(up, index=False); r.to_csv(rp, index=False)
            frames.append(u); reds.append(r)
            m = u[u.method == "moment"].groupby(u.pair_type).gap.median()
            log(f"{ds.name} seed {seed}: {(time.time() - t0) / 60:.1f} min | moment gap within={m.get('within', np.nan):.3f} cross={m.get('cross', np.nan):.3f}")
    if not frames:
        raise RuntimeError("No diagnostics produced.")
    units = pd.concat(frames, ignore_index=True)
    red = pd.concat(reds, ignore_index=True)
    units.to_csv(DIRS["tables"] / "all_diagnostic_units.csv", index=False)
    # Consistency with v2: cross-family full refit and moment results must reproduce.
    v2c = v2units[v2units.method.isin(["full_refit", "moment"])][["dataset", "seed", "source", "target", "pair_type", "method", "nba"]]
    mine = units[(units.pair_type == "cross") | (np.isclose(units.d2d_sigma.fillna(-1), 0.05))]
    chk = mine.merge(v2c, on=["dataset", "seed", "source", "target", "pair_type", "method"], suffixes=("", "_v2"))
    max_dev = float(np.max(np.abs(chk.nba - chk.nba_v2))) if len(chk) else float("nan")

    res = analyse_v3(units, red, capn, mem, lc, region, near, absperf, land)
    try:
        make_v3_figures(res)
    except Exception:
        log("Figure generation failed:\n" + traceback.format_exc())
    rep = res["rep"]
    elapsed = (time.time() - t_start) / 60
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 110)
    print("COPY-PASTE REPORT START - MEMRISTIVE RESERVOIR PORTABILITY v3 DIAGNOSTICS")
    print("=" * 110)
    print(f"Runtime: {elapsed:.1f} min | protocol hash {phash} | datasets {len(loaded)} | seeds {SEEDS}")
    print(f"Consistency with v2 (full refit and moment, cross family and 5% within family, max |nBA difference|): {max_dev:.6f}")
    print("\nD2D SWEEP: GAP TO FULL REFIT (normalized BA, hierarchical bootstrap CI)")
    print(rep["sweep"].to_string(index=False))
    print("\nDIAGNOSTICS BY LEVEL (pair medians)")
    print(rep["diag_by_level"].T.to_string())
    print("\nQ1 WHY READOUTS FAIL ACROSS FAMILIES:", json.dumps(rep["why_readout"], indent=1, default=str))
    print("\nQ2 ALIGNMENT FAILURE CHAIN (cross family)")
    print(rep["chain"].to_string(index=False))
    print(rep["alignment_comparisons"].to_string(index=False))
    print("\nQ3 WHY WARM START FAILS:", json.dumps(rep["why_warm"], indent=1, default=str))
    print("\nQ4 NONLINEAR CAPACITY VS READ NOISE (median over grid)")
    print(rep["capacity_noise"].to_string())
    print("Emulation R2 predictors (Spearman, pair x operating point):", json.dumps(rep["emulation_predictors"], default=str))
    print("\nQ5 WHY THE OPERATING POINT TRANSFERS:", json.dumps(rep["why_oppoint"], indent=1, default=str))
    print("\nQ6 WHY DEVICES DO NOT BEAT THE RAW BASELINE:", json.dumps(rep["why_raw"], indent=1, default=str))
    print("=" * 110)
    print("COPY-PASTE REPORT END")
    print("=" * 110)
    log(f"Done in {elapsed:.1f} min. Tables: {DIRS['tables']} | Figures: {DIRS['figures']}")


if __name__ == "__main__":
    main()
