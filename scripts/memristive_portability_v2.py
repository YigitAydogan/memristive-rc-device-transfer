#!/usr/bin/env python3
"""
Memristive Reservoir Portability v2
===================================

Research question
-----------------
When a memristive reservoir computer is moved from one device family to another,
is the change approximately a linear transformation of the measured response,
so that it can be corrected WITHOUT task labels by driving both devices with a
task-agnostic probe signal? And does the success of that correction follow from
the devices' linear and nonlinear information-processing capacity?

What this script does (single Colab cell, Google Drive already mounted)
------------------------------------------------------------------------
1. Six literature-grounded compact memristor families, each simulated with exact
   or closed-form per-symbol updates, device-to-device (D2D) parameter spread,
   cycle-to-cycle (C2C) state noise, and read noise on a measurable observable
   (log10 read conductance).
2. A 6 x 6 operating-point grid (input gain x drive duration relative to the
   family's datasheet switching time).
3. A time-binned readout (24 bins x {mean, std} of the observable) with a
   class-balanced ridge classifier, and the same readout on the raw input as a
   matched non-memristive baseline. 1-NN Euclidean and MiniRocket are reported as
   absolute references.
4. Information-processing capacity (degrees 1 to 3, Legendre basis) of every
   device instance at every operating point, from a random probe signal.
5. A transfer ladder for every ordered device pair, cross-family and
   within-family (two D2D instances):
       zero-shot, moment matching, CORAL, replay FIR alignment, probe FIR
       alignment, probe quadratic alignment, k-shot scratch / warm-start
       readouts (k = 1, 2, 5, 10 labels per class), full readout refit,
       and the target oracle.
6. Dataset-level statistics (datasets are the replication unit), a mixed-effects
   model with dataset random intercepts, hierarchical bootstrap intervals,
   publication-style figures, and a copy-paste report.

Hypotheses (written to protocol.json before any task simulation)
----------------------------------------------------------------
H1  Label-free probe FIR alignment gives lower cross-family regret than moment
    matching (one-sided dataset-level Wilcoxon, Holm-corrected across the four
    label-free methods).
H2  Across device pairs within a dataset, higher probe emulation R^2 predicts
    lower probe-aligned task regret (mean within-dataset Spearman rho < 0,
    Wilcoxon over datasets).
H3  Probe emulation R^2 increases with the product of the source and target
    linear-capacity fractions (Spearman across pairs).
H4  The source-selected operating point gives lower target regret than a randomly
    chosen operating point (one-sided dataset-level Wilcoxon).
H5  With k labels per class, a readout warm-started from the probe-aligned source
    readout has lower regret than a readout trained from scratch (Holm across k).

Execution notes
---------------
* Paste into one Colab cell and run. Drive must already be mounted; this script
  never mounts it. CPU is sufficient.
* Results checkpoint per dataset and seed, so a disconnected run resumes.
* Optional environment variables:
    PORTV2_SMOKE=1        tiny synthetic run for implementation testing only
    PORTV2_IDEAL=1        no noise and nominal parameters (sensitivity run)
    PORTV2_DATASETS=a,b   restrict to listed datasets
    PORTV2_MINIROCKET=0   skip the MiniRocket reference
    MEMRISTIVE_PROJECT_ROOT=/path   override output root
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

SCRIPT_VERSION = "2.0.0"
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
RUN_NAME = "portability_v2" + ("_ideal" if IDEAL else "") + ("_smoke" if SMOKE else "")
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
# Probe phase: capacities and emulation quality (dataset independent)
# =============================================================================


def probe_inputs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("probe_input", seed, MASTER_SEED))
    return rng.uniform(-1.0, 1.0, size=(PROBE_EPISODES, PROBE_LENGTH))


def run_probe_phase(instances: Dict[int, Dict[str, Instance]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cap_path = DIRS["probe"] / "capacities.csv"
    emu_path = DIRS["probe"] / "emulation.csv"
    t0 = max(CAP_MAX_DELAY_D1, CAP_STATE_LAGS)
    probe_traj: Dict[str, np.ndarray] = {}
    for seed, insts in instances.items():
        for inst in insts.values():
            fp = DIRS["probe"] / f"traj_{inst.iid}.npy"
            if fp.exists():
                probe_traj[inst.iid] = np.load(fp)
            else:
                probe_traj[inst.iid] = simulate(inst, probe_inputs(seed), range(N_OPS), "traj", ("probe", seed))
                np.save(fp, probe_traj[inst.iid])
    if cap_path.exists() and emu_path.exists():
        return pd.read_csv(cap_path), pd.read_csv(emu_path), probe_traj
    log("Computing information-processing capacities ...")
    cap_rows = []
    for seed, insts in instances.items():
        Up = probe_inputs(seed)
        Yc, deg = capacity_targets(Up, t0)
        raw = capacities(Up, Yc, deg, t0)
        cap_rows.append({"seed": seed, "instance": "raw_input", "family": "RAW", "role": "-", "op": -1, **raw})
        for inst in insts.values():
            tr = probe_traj[inst.iid]
            for op in range(N_OPS):
                c = capacities(tr[:, :, op], Yc, deg, t0)
                cap_rows.append({"seed": seed, "instance": inst.iid, "family": inst.family, "role": inst.role,
                                 "op": op, "gain": GAINS[OP_GAIN_IDX[op]], "frac": FRACS[OP_FRAC_IDX[op]], **c})
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(cap_path, index=False)
    log("Computing probe emulation quality for every ordered pair and operating point ...")
    emu_rows = []
    E_tr = PROBE_TRAIN_EPISODES
    for seed, insts in instances.items():
        pairs = []
        for s in FAMILY_CODES:
            for t in FAMILY_CODES:
                if s != t:
                    pairs.append((insts[f"{s}_A"], insts[f"{t}_A"], "cross"))
            pairs.append((insts[f"{s}_A"], insts[f"{s}_B"], "within"))
        for src, tgt, ptype in pairs:
            pad = rest_observable(tgt)
            for op in range(N_OPS):
                rs = probe_traj[src.iid][:, :, op]
                rt = probe_traj[tgt.iid][:, :, op]
                row = {"seed": seed, "source": src.family, "target": tgt.family, "pair_type": ptype, "op": op}
                for name, taps, quad in [("fir", FIR_TAPS, False), ("quad", QUAD_TAPS, True)]:
                    mp = fit_lag_map(rs[:E_tr], rt[:E_tr], pad, taps, quad)
                    row[f"emulation_r2_{name}"] = r2_score_flat(rs[E_tr:], mp.apply(rt[E_tr:]))
                emu_rows.append(row)
    emu = pd.DataFrame(emu_rows)
    emu.to_csv(emu_path, index=False)
    return cap, emu, probe_traj


# =============================================================================
# Per dataset and seed experiment
# =============================================================================


def run_unit(ds: Dataset, seed: int, insts: Dict[str, Instance], probe_traj: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    C = ds.n_classes
    folds = make_folds(ds.y_train, stable_seed("folds", ds.name, seed))
    # Phase 1: operating-point landscape for every instance.
    land_rows = []
    table: Dict[str, Dict[str, np.ndarray]] = {}
    for key, inst in insts.items():
        Ftr = simulate(inst, ds.U_train, range(N_OPS), "features", ("task", ds.name, seed, "train"))
        Fte = simulate(inst, ds.U_test, range(N_OPS), "features", ("task", ds.name, seed, "test"))
        cv = np.zeros(N_OPS); ai = np.zeros(N_OPS, int); te = np.zeros(N_OPS)
        cache: Dict[Tuple, Tuple] = {}
        for op in range(N_OPS):
            k = op_key(inst.family, op)
            if k not in cache:
                cache[k] = select_and_test(Ftr[op], Fte[op], ds.y_train, ds.y_test, C, folds)
            cv[op], ai[op], te[op] = cache[k]
        best = int(np.argmax(cv))
        table[key] = {"cv": cv, "ai": ai, "test": te, "best": best}
        for op in range(N_OPS):
            land_rows.append({"dataset": ds.name, "tier": ds.tier, "seed": seed, "instance": key,
                              "family": inst.family, "role": inst.role, "op": op,
                              "gain": GAINS[OP_GAIN_IDX[op]], "frac": FRACS[OP_FRAC_IDX[op]],
                              "duration": op_duration(inst.family, op), "cv_ba": cv[op],
                              "alpha_rel": ALPHA_REL[ai[op]], "test_ba": te[op], "test_nba": nba(te[op], C),
                              "selected": op == best})
        del Ftr, Fte
    landscape = pd.DataFrame(land_rows)

    # Phase 2: transfer ladder.
    traj_cache: Dict[Tuple[str, int, str], np.ndarray] = {}

    def traj(key: str, op: int, split: str) -> np.ndarray:
        ck = (key, op_key(insts[key].family, op), split)
        if ck not in traj_cache:
            U = ds.U_train if split == "train" else ds.U_test
            traj_cache[ck] = simulate(insts[key], U, [op], "traj", ("task", ds.name, seed, split))[:, :, 0]
        return traj_cache[ck]

    rows = []
    rng_k = np.random.default_rng(stable_seed("kshot", ds.name, seed))
    replay_n = min(len(ds.y_train), max(32, int(300_000 / ds.U_train.shape[1])))
    for s in FAMILY_CODES:
        skey = f"{s}_A"
        op_s = table[skey]["best"]
        alpha_s = ALPHA_REL[table[skey]["ai"][op_s]]
        rs_tr = traj(skey, op_s, "train")
        Fs_tr = features_from_traj(rs_tr)
        mu_s, sd_s = std_stats(Fs_tr)
        Zs_tr = (Fs_tr - mu_s) / sd_s
        ro_s = ridge_fit(Zs_tr, ds.y_train, C, alpha_s * class_weights(ds.y_train, C).sum())
        targets = [(f"{t}_A", "cross") for t in FAMILY_CODES if t != s] + [(f"{s}_B", "within")]
        for tkey, ptype in targets:
            tinst = insts[tkey]
            tt = table[tkey]
            oracle_ba = tt["test"][tt["best"]]
            oracle = nba(oracle_ba, C)
            rt_tr = traj(tkey, op_s, "train")
            rt_te = traj(tkey, op_s, "test")
            Ft_tr, Ft_te = features_from_traj(rt_tr), features_from_traj(rt_te)
            mu_t, sd_t = std_stats(Ft_tr)
            Zt_tr, Zt_te = (Ft_tr - mu_t) / sd_t, (Ft_te - mu_t) / sd_t
            pad = rest_observable(tinst)
            res: Dict[str, float] = {}
            extra: Dict[str, float] = {}
            res["zero_shot"] = balanced_accuracy(ds.y_test, ro_s.predict((Ft_te - mu_s) / sd_s), C)
            res["moment"] = balanced_accuracy(ds.y_test, ro_s.predict(Zt_te), C)
            A = coral_matrix(Zs_tr, Zt_tr)
            res["coral"] = balanced_accuracy(ds.y_test, ro_s.predict(Zt_te @ A), C)
            # Probe-based alignment (task agnostic, no task inputs, no labels).
            E_tr = PROBE_TRAIN_EPISODES
            ps = probe_traj[insts[skey].iid][:, :, op_s]
            pt = probe_traj[tinst.iid][:, :, op_s]
            Zal: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for name, taps, quad in [("probe_fir", FIR_TAPS, False), ("probe_quad", QUAD_TAPS, True)]:
                mp = fit_lag_map(ps[:E_tr], pt[:E_tr], pad, taps, quad)
                extra[f"{name}_emulation_r2"] = r2_score_flat(ps[E_tr:], mp.apply(pt[E_tr:]))
                za_tr = (features_from_traj(mp.apply(rt_tr)) - mu_s) / sd_s
                za_te = (features_from_traj(mp.apply(rt_te)) - mu_s) / sd_s
                Zal[name] = (za_tr, za_te)
                res[name] = balanced_accuracy(ds.y_test, ro_s.predict(za_te), C)
                if name == "probe_fir":
                    fa_tr = features_from_traj(mp.apply(rt_tr))
                    mu_a, sd_a = std_stats(fa_tr)
                    res["probe_fir_mm"] = balanced_accuracy(ds.y_test, ro_s.predict((features_from_traj(mp.apply(rt_te)) - mu_a) / sd_a), C)
            # Replay alignment: same task training inputs, unlabeled, driven through both devices.
            ridx = rng_k.permutation(len(ds.y_train))[:replay_n]
            cut = max(1, int(0.75 * len(ridx)))
            mp = fit_lag_map(rs_tr[ridx[:cut]], rt_tr[ridx[:cut]], pad, FIR_TAPS, False)
            extra["replay_fir_emulation_r2"] = r2_score_flat(rs_tr[ridx[cut:]], mp.apply(rt_tr[ridx[cut:]])) if cut < len(ridx) else float("nan")
            res["replay_fir"] = balanced_accuracy(ds.y_test, ro_s.predict((features_from_traj(mp.apply(rt_te)) - mu_s) / sd_s), C)
            # Full refit at the source-selected operating point (target labels, target CV ridge penalty).
            alpha_t = ALPHA_REL[tt["ai"][op_s]]
            ro_f = ridge_fit(Zt_tr, ds.y_train, C, alpha_t * class_weights(ds.y_train, C).sum())
            res["full_refit"] = balanced_accuracy(ds.y_test, ro_f.predict(Zt_te), C)
            # k-shot readouts.
            for k in K_SHOTS:
                idx = np.concatenate([rng_k.permutation(np.flatnonzero(ds.y_train == c))[:k] for c in range(C)])
                yk = ds.y_train[idx]
                ro = ridge_fit(Zt_tr[idx], yk, C, LAMBDA_PRIOR)
                res[f"scratch_k{k}"] = balanced_accuracy(ds.y_test, ro.predict(Zt_te), C)
                ro = ridge_fit(Zt_tr[idx], yk, C, LAMBDA_PRIOR, prior=ro_s.W)
                res[f"moment_warm_k{k}"] = balanced_accuracy(ds.y_test, ro.predict(Zt_te), C)
                za_tr, za_te = Zal["probe_fir"]
                ro = ridge_fit(za_tr[idx], yk, C, LAMBDA_PRIOR, prior=ro_s.W)
                res[f"probe_warm_k{k}"] = balanced_accuracy(ds.y_test, ro.predict(za_te), C)
            res["oracle"] = oracle_ba
            same_op = op_key(tinst.family, op_s) == op_key(tinst.family, tt["best"])
            base = {"dataset": ds.name, "tier": ds.tier, "seed": seed, "source": s, "target": tinst.family,
                    "pair_type": ptype, "source_op": op_s, "target_op": tt["best"], "same_effective_op": same_op,
                    "target_test_nba_at_source_op_phase1": nba(tt["test"][op_s], C),
                    "target_mean_test_nba_over_grid": float(np.mean([nba(v, C) for v in tt["test"]])),
                    "oracle_nba": oracle, "n_classes": C, **extra}
            for mth, ba in res.items():
                rows.append({**base, "method": mth, "ba": ba, "nba": nba(ba, C), "regret": oracle - nba(ba, C)})
        traj_cache.clear()
    return pd.DataFrame(rows), landscape


def reference_baselines(ds: Dataset, seeds: Sequence[int]) -> Dict[str, float]:
    C = ds.n_classes
    out: Dict[str, float] = {"dataset": ds.name, "tier": ds.tier, "n_train": len(ds.y_train),
                             "n_test": len(ds.y_test), "length": ds.U_train.shape[1], "classes": C,
                             "chance_ba": 1.0 / C}
    Ftr, Fte = features_from_traj(ds.U_train), features_from_traj(ds.U_test)
    vals = []
    for seed in seeds:
        _, _, te = select_and_test(Ftr, Fte, ds.y_train, ds.y_test, C, make_folds(ds.y_train, stable_seed("folds", ds.name, seed)))
        vals.append(te)
    out["raw_binned_ridge_ba"] = float(np.median(vals))
    knn = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(ds.X_train_raw, ds.y_train)
    out["nn1_euclidean_ba"] = balanced_accuracy(ds.y_test, knn.predict(ds.X_test_raw), C)
    out["minirocket_ba"] = float("nan")
    if RUN_MINIROCKET:
        try:
            from aeon.classification.convolution_based import MiniRocketClassifier
            clf = MiniRocketClassifier(random_state=0, n_jobs=-1)
            clf.fit(ds.X_train_raw[:, None, :], ds.y_train)
            out["minirocket_ba"] = balanced_accuracy(ds.y_test, np.asarray(clf.predict(ds.X_test_raw[:, None, :])).astype(int), C)
        except Exception as exc:
            log(f"MiniRocket failed on {ds.name}: {type(exc).__name__}: {exc}")
    return out


# =============================================================================
# Analysis
# =============================================================================


def analyse(units: pd.DataFrame, land: pd.DataFrame, refs: pd.DataFrame, cap: pd.DataFrame,
            emu: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    # Pair level (median over seeds), then dataset level (median over pairs).
    pair = (units.groupby(["dataset", "tier", "source", "target", "pair_type", "method"], as_index=False)
            .agg(regret=("regret", "median"), nba=("nba", "median"),
                 probe_fir_emulation_r2=("probe_fir_emulation_r2", "median"),
                 probe_quad_emulation_r2=("probe_quad_emulation_r2", "median"),
                 replay_fir_emulation_r2=("replay_fir_emulation_r2", "median")))
    pair.to_csv(DIRS["tables"] / "pair_level_results.csv", index=False)
    dsl = pair.groupby(["dataset", "tier", "pair_type", "method"], as_index=False).agg(regret=("regret", "median"), nba=("nba", "median"))
    dsl.to_csv(DIRS["tables"] / "dataset_level_results.csv", index=False)
    wide = dsl[dsl.pair_type == "cross"].pivot(index="dataset", columns="method", values="regret")
    wide_w = dsl[dsl.pair_type == "within"].pivot(index="dataset", columns="method", values="regret")

    # Method summary.
    rows = []
    methods = list(PRETTY_METHOD) + [f"{p}_k{k}" for p in ("scratch", "moment_warm", "probe_warm") for k in K_SHOTS]
    for mth in methods:
        for ptype, w in [("cross", wide), ("within", wide_w)]:
            if mth not in w:
                continue
            sub = pair[(pair.method == mth) & (pair.pair_type == ptype)]
            med, lo, hi = hier_boot_ci(sub, "regret", stable_seed("boot", mth, ptype))
            rows.append({"method": mth, "pair_type": ptype, "datasets": int(w[mth].notna().sum()),
                         "median_regret": med, "ci95_low": lo, "ci95_high": hi,
                         "median_nba": float(np.median(dsl[(dsl.method == mth) & (dsl.pair_type == ptype)].nba))})
    msum = pd.DataFrame(rows)
    msum.to_csv(DIRS["tables"] / "method_summary.csv", index=False)
    report["method_summary"] = msum

    # H1: label-free alignment vs moment matching.
    pv = {m: wilcoxon_less(wide[m], wide["moment"])[0] for m in LABEL_FREE if m in wide}
    ph = holm(pv)
    h1 = pd.DataFrame([{"method": m, "median_regret_reduction_vs_moment": float(np.median((wide["moment"] - wide[m]).dropna())),
                        "datasets_improved": int(((wide["moment"] - wide[m]) > 0).sum()), "datasets": int(wide[m].notna().sum()),
                        "p_one_sided": pv[m], "p_holm": ph[m]} for m in pv])
    for tier in sorted(dsl.tier.unique()):
        wt = dsl[(dsl.pair_type == "cross") & (dsl.tier == tier)].pivot(index="dataset", columns="method", values="regret")
        h1[f"median_reduction_{tier}"] = [float(np.median((wt["moment"] - wt[m]).dropna())) if m in wt else np.nan for m in h1.method]
    h1.to_csv(DIRS["tables"] / "H1_label_free_alignment.csv", index=False)
    report["H1"] = h1

    # Mixed model on probe_fir minus moment at pair level.
    try:
        import statsmodels.formula.api as smf
        warnings.simplefilter("ignore")
        a = pair[(pair.pair_type == "cross") & (pair.method == "probe_fir")][["dataset", "source", "target", "regret"]]
        b = pair[(pair.pair_type == "cross") & (pair.method == "moment")][["dataset", "source", "target", "regret"]]
        mm = a.merge(b, on=["dataset", "source", "target"], suffixes=("_probe", "_moment"))
        mm["reduction"] = mm["regret_moment"] - mm["regret_probe"]
        fit = smf.mixedlm("reduction ~ C(source, Sum) + C(target, Sum)", mm, groups=mm["dataset"]).fit(reml=True)
        ci = fit.conf_int().loc["Intercept"]
        report["mixed_model"] = {"mean_reduction": float(fit.params["Intercept"]), "ci95_low": float(ci[0]),
                                 "ci95_high": float(ci[1]), "p_two_sided": float(fit.pvalues["Intercept"]),
                                 "n_pairs": int(len(mm)), "n_datasets": int(mm.dataset.nunique())}
    except Exception as exc:
        report["mixed_model"] = {"error": f"{type(exc).__name__}: {exc}"}

    # H2: emulation R2 predicts probe-aligned regret within dataset.
    pp = pair[(pair.pair_type == "cross") & (pair.method == "probe_fir")]
    rhos = []
    for dname, g in pp.groupby("dataset"):
        g = g.dropna(subset=["probe_fir_emulation_r2", "regret"])
        if len(g) >= 5 and g.regret.std() > 0:
            rhos.append({"dataset": dname, "rho": float(spearmanr(g.probe_fir_emulation_r2, g.regret)[0])})
    rh = pd.DataFrame(rhos)
    rh.to_csv(DIRS["tables"] / "H2_within_dataset_spearman.csv", index=False)
    p2 = wilcoxon_less(rh.rho.to_numpy(), np.zeros(len(rh)))[0] if len(rh) >= 3 else float("nan")
    report["H2"] = {"mean_rho": float(rh.rho.mean()) if len(rh) else float("nan"),
                    "median_rho": float(rh.rho.median()) if len(rh) else float("nan"),
                    "datasets": int(len(rh)), "p_one_sided": p2}

    # H3: emulation R2 vs linear-fraction product (probe level, median over seeds).
    capA = cap[cap.role == "A"][["seed", "family", "op", "linear_fraction", "C_total"]]
    capB = cap[cap.role == "B"][["seed", "family", "op", "linear_fraction"]]
    e = emu.copy()
    e = e.merge(capA.rename(columns={"family": "source", "linear_fraction": "lf_source", "C_total": "C_source"}), on=["seed", "source", "op"], how="left")
    e_cross = e[e.pair_type == "cross"].merge(capA[["seed", "family", "op", "linear_fraction"]].rename(columns={"family": "target", "linear_fraction": "lf_target"}), on=["seed", "target", "op"], how="left")
    e_within = e[e.pair_type == "within"].merge(capB.rename(columns={"family": "target", "linear_fraction": "lf_target"}), on=["seed", "target", "op"], how="left")
    e = pd.concat([e_cross, e_within])
    e["lf_product"] = e.lf_source * e.lf_target
    e.to_csv(DIRS["tables"] / "emulation_vs_capacity.csv", index=False)
    ec = e[e.pair_type == "cross"].groupby(["source", "target", "op"], as_index=False).agg(
        r2=("emulation_r2_fir", "median"), lf_product=("lf_product", "median"))
    ec = ec.dropna()
    rho_op = spearmanr(ec.lf_product, ec.r2) if len(ec) > 5 else (np.nan, np.nan)
    epair = ec.groupby(["source", "target"], as_index=False).agg(r2=("r2", "median"), lf_product=("lf_product", "median"))
    rho_pair = spearmanr(epair.lf_product, epair.r2) if len(epair) > 5 else (np.nan, np.nan)
    report["H3"] = {"spearman_rho_pair_op": float(rho_op[0]), "p_pair_op": float(rho_op[1]), "n_pair_op": int(len(ec)),
                    "spearman_rho_pair": float(rho_pair[0]), "p_pair": float(rho_pair[1]), "n_pairs": int(len(epair))}

    # H4: source-selected versus random operating point on target, full refit.
    u = units[(units.method == "full_refit") & (units.pair_type == "cross")].copy()
    u["regret_selected"] = u.oracle_nba - u.target_test_nba_at_source_op_phase1
    u["regret_random"] = u.oracle_nba - u.target_mean_test_nba_over_grid
    h4p = (u.groupby(["dataset", "source", "target"], as_index=False)[["regret_selected", "regret_random", "same_effective_op"]].median()
           .groupby("dataset", as_index=False)[["regret_selected", "regret_random", "same_effective_op"]].median())
    h4p.to_csv(DIRS["tables"] / "H4_operating_point_portability.csv", index=False)
    report["H4"] = {"median_regret_selected": float(h4p.regret_selected.median()),
                    "median_regret_random": float(h4p.regret_random.median()),
                    "p_one_sided": wilcoxon_less(h4p.regret_selected, h4p.regret_random)[0],
                    "fraction_same_effective_op": float(u.same_effective_op.mean()),
                    "consistency_max_abs_nba_diff_phase1_vs_phase2": float(np.max(np.abs(
                        u.target_test_nba_at_source_op_phase1 - u.nba))) if len(u) else float("nan")}

    # H5: probe warm start vs scratch for each k.
    pv5 = {}
    rows5 = []
    for k in K_SHOTS:
        a, b = f"probe_warm_k{k}", f"scratch_k{k}"
        if a in wide and b in wide:
            pv5[k] = wilcoxon_less(wide[a], wide[b])[0]
            rows5.append({"k": k, "median_regret_scratch": float(wide[b].median()), "median_regret_moment_warm": float(wide[f"moment_warm_k{k}"].median()),
                          "median_regret_probe_warm": float(wide[a].median()),
                          "datasets_improved": int(((wide[b] - wide[a]) > 0).sum())})
    ph5 = holm({str(k): v for k, v in pv5.items()})
    for r in rows5:
        r["p_one_sided"] = pv5[r["k"]]; r["p_holm"] = ph5[str(r["k"])]
    h5 = pd.DataFrame(rows5)
    h5.to_csv(DIRS["tables"] / "H5_label_efficiency.csv", index=False)
    report["H5"] = h5

    # Absolute performance table.
    orc = (land[(land.role == "A") & land.selected].groupby(["dataset", "family"], as_index=False).test_ba.median()
           .pivot(index="dataset", columns="family", values="test_ba"))
    orc.columns = [f"oracle_{c}_ba" for c in orc.columns]
    absperf = refs.set_index("dataset").join(orc).reset_index()
    absperf.to_csv(DIRS["tables"] / "absolute_performance.csv", index=False)
    report["absolute"] = absperf

    # Capacity at selected operating points.
    sel = land[(land.role == "A") & land.selected][["dataset", "seed", "family", "op"]]
    capsel = sel.merge(capA.rename(columns={"linear_fraction": "lf"}), on=["seed", "family", "op"], how="left").merge(
        cap[cap.role == "A"][["seed", "family", "op", "C1", "C2", "C3"]], on=["seed", "family", "op"], how="left")
    capfam = capsel.groupby("family", as_index=False)[["C1", "C2", "C3", "C_total", "lf"]].median()
    capfam.to_csv(DIRS["tables"] / "capacity_at_selected_ops.csv", index=False)
    report["capacity"] = capfam
    report["raw_capacity"] = cap[cap.family == "RAW"][["C1", "C2", "C3", "C_total"]].median().to_dict()

    # Operating-point landscape flatness.
    flat = (land[land.role == "A"].groupby(["dataset", "seed", "family"], as_index=False)
            .agg(nba_max=("test_nba", "max"), nba_min=("test_nba", "min"), nba_median=("test_nba", "median")))
    flat["range"] = flat.nba_max - flat.nba_min
    flatf = flat.groupby("family", as_index=False)[["range", "nba_max", "nba_median"]].median()
    flatf.to_csv(DIRS["tables"] / "operating_point_landscape.csv", index=False)
    report["landscape"] = flatf
    return {"report": report, "pair": pair, "dsl": dsl, "wide": wide, "emu": e, "capsel": capsel, "h4": h4p}


def make_figures(res: Dict[str, Any]) -> None:
    wide, pair, e, capsel, h4 = res["wide"], res["pair"], res["emu"], res["capsel"], res["h4"]
    # Figure 1: transfer ladder and label efficiency.
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    meths = [m for m in PRETTY_METHOD if m in wide]
    rng = np.random.default_rng(1)
    allv = []
    for i, m in enumerate(meths):
        v = wide[m].dropna().to_numpy()
        allv.extend(v)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=28, color="0.65", zorder=2,
                   label="Dataset" if i == 0 else None)
        med, lo, hi = hier_boot_ci(pair[(pair.method == m) & (pair.pair_type == "cross")], "regret", stable_seed("fig", m))
        allv.extend([lo, hi])
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], fmt="o", color="black", ms=9, capsize=6, lw=2, zorder=3,
                    label="Median, 95% CI" if i == 0 else None)
    ax.axhline(0, color="0.4", lw=1.0, ls=":")
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([PRETTY_METHOD[m] for m in meths], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(meths) - 0.4)
    set_bounds(ax, "y", min(allv + [0]), max(allv))
    ax.set_ylabel("Regret to target oracle (normalized BA)")
    style_ax(ax)
    outlined_legend(ax, loc="upper right")
    ax = axes[1]
    ks = [k for k in K_SHOTS if f"scratch_k{k}" in wide]
    allv = []
    for pref, lab, col, mk in [("scratch", "Scratch", "#c0392b", "s"), ("moment_warm", "Warm start, moment", "#e08a1e", "^"),
                               ("probe_warm", "Warm start, probe FIR", "#1b6ca8", "o")]:
        meds, los, his = [], [], []
        for k in ks:
            med, lo, hi = hier_boot_ci(pair[(pair.method == f"{pref}_k{k}") & (pair.pair_type == "cross")], "regret", stable_seed("fig5", pref, k))
            meds.append(med); los.append(lo); his.append(hi)
        meds, los, his = map(np.array, (meds, los, his))
        allv.extend(list(los) + list(his))
        ax.errorbar(ks, meds, yerr=[meds - los, his - meds], marker=mk, color=col, ms=9, capsize=5, lw=2, label=lab)
    for m, ls, lab in [("probe_fir", "--", "Probe FIR, no labels"), ("full_refit", ":", "Full refit")]:
        if m in wide:
            v = float(wide[m].median()); allv.append(v)
            ax.axhline(v, color="0.3", ls=ls, lw=1.6, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
    ax.set_xlim(ks[0] / 1.35, ks[-1] * 1.35)
    set_bounds(ax, "y", min(allv + [0]), max(allv))
    ax.set_xlabel("Target labels per class")
    ax.set_ylabel("Regret to target oracle (normalized BA)")
    style_ax(ax)
    outlined_legend(ax, loc="best")
    fig.tight_layout(w_pad=3)
    panel_letters(fig, axes)
    save_fig(fig, "fig1_transfer_ladder_and_label_efficiency")

    # Figure 2: mechanism.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    ax = axes[0]
    capf = capsel.groupby("family")[["C1", "C2", "C3"]].median().reindex(FAMILY_CODES)
    bottom = np.zeros(len(capf))
    for col, colr, lab in [("C1", "#1b6ca8", "Degree 1"), ("C2", "#e08a1e", "Degree 2"), ("C3", "#2a9d5c", "Degree 3")]:
        ax.bar(range(len(capf)), capf[col].fillna(0), bottom=bottom, color=colr, edgecolor="black", lw=1.0, label=lab)
        bottom += capf[col].fillna(0).to_numpy()
    ax.set_xticks(range(len(capf))); ax.set_xticklabels(capf.index)
    ax.set_xlim(-0.6, len(capf) - 0.4)
    set_bounds(ax, "y", 0, max(bottom.max(), 1.0))
    ax.set_xlabel("Device family"); ax.set_ylabel("Information-processing capacity")
    style_ax(ax); outlined_legend(ax, loc="upper right")
    ax = axes[1]
    pp = pair[(pair.pair_type == "cross") & (pair.method == "probe_fir")].dropna(subset=["probe_fir_emulation_r2", "regret"])
    xv = pp.probe_fir_emulation_r2.clip(lower=-1.0)
    for fam in FAMILY_CODES:
        s = pp.target == fam
        ax.scatter(xv[s], pp.regret[s], s=26, alpha=0.75, color=FAM_COLORS[fam], label=f"Target {fam}")
    set_bounds(ax, "x", float(xv.min()) if len(xv) else -1, 1.0)
    set_bounds(ax, "y", min(0, float(pp.regret.min()) if len(pp) else 0), float(pp.regret.max()) if len(pp) else 1)
    ax.set_xlabel("Probe emulation $R^2$"); ax.set_ylabel("Probe FIR regret (normalized BA)")
    style_ax(ax); outlined_legend(ax, loc="best", ncol=2, fontsize=11)
    ax = axes[2]
    ec = e[e.pair_type == "cross"].groupby(["source", "target", "op"], as_index=False).agg(
        r2=("emulation_r2_fir", "median"), lf=("lf_product", "median")).dropna()
    ax.scatter(ec.lf, ec.r2.clip(lower=-1.0), s=12, alpha=0.5, color="#1b6ca8")
    set_bounds(ax, "x", 0, 1.0); set_bounds(ax, "y", float(ec.r2.clip(lower=-1).min()) if len(ec) else -1, 1.0)
    ax.set_xlabel("Linear-capacity fraction, source x target"); ax.set_ylabel("Probe emulation $R^2$")
    style_ax(ax)
    fig.tight_layout(w_pad=3)
    panel_letters(fig, axes)
    save_fig(fig, "fig2_capacity_mechanism")

    # Figure 3: operating-point portability.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [0.8, 1.0]})
    ax = axes[0]
    for _, r in h4.iterrows():
        ax.plot([0, 1], [r.regret_selected, r.regret_random], color="0.6", lw=1.2, marker="o", ms=6)
    ax.plot([0, 1], [h4.regret_selected.median(), h4.regret_random.median()], color="black", lw=3, marker="o", ms=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Source selected", "Random"]); ax.set_xlim(-0.35, 1.35)
    vals = np.concatenate([h4.regret_selected, h4.regret_random, [0]])
    set_bounds(ax, "y", float(np.nanmin(vals)), float(np.nanmax(vals)))
    ax.set_ylabel("Target regret, full refit (normalized BA)")
    style_ax(ax)
    ax = axes[1]
    u = res["pair"]
    units = res.get("units_df")
    mat = np.full((len(FAMILY_CODES), len(FAMILY_CODES)), np.nan)
    if units is not None:
        uu = units[(units.method == "full_refit") & (units.pair_type == "cross")]
        g = uu.groupby(["source", "target"]).same_effective_op.mean()
        for (s, t), v in g.items():
            mat[FAMILY_CODES.index(s), FAMILY_CODES.index(t)] = v
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(FAMILY_CODES))); ax.set_xticklabels(FAMILY_CODES)
    ax.set_yticks(range(len(FAMILY_CODES))); ax.set_yticklabels(FAMILY_CODES)
    ax.set_xlabel("Target family"); ax.set_ylabel("Source family")
    for i in range(len(FAMILY_CODES)):
        for j in range(len(FAMILY_CODES)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if mat[i, j] < 0.6 else "black")
    style_ax(ax)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Fraction with identical operating point")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0]); cb.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cb.outline.set_linewidth(SPINE_W)
    fig.tight_layout(w_pad=3)
    panel_letters(fig, axes)
    save_fig(fig, "fig3_operating_point_portability")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    t_start = time.time()
    log("=" * 100)
    log(f"MEMRISTIVE RESERVOIR PORTABILITY v{SCRIPT_VERSION} | root={ROOT} | smoke={SMOKE} | ideal={IDEAL}")
    log("Drive mounting is intentionally not performed by this script.")

    for fam in FAMILY_CODES:
        T_SWITCH[fam] = calibrate_switch_time(fam)
        T_RECOVER[fam] = recovery_duration(fam)
    cal = pd.DataFrame([{"family": f, "name": FAMILIES[f].name, "citation": FAMILIES[f].citation,
                         "voltage_mode": FAMILIES[f].voltage_mode, "v_pos": FAMILIES[f].v_pos,
                         "v_mid": FAMILIES[f].v_mid, "discrete": FAMILIES[f].discrete,
                         "t_switch": T_SWITCH[f], "t_recover": T_RECOVER[f],
                         "effective_durations": "; ".join(f"{op_duration(f, o):.4g}" for o in range(len(FRACS)))}
                        for f in FAMILY_CODES])
    cal.to_csv(DIRS["tables"] / "device_interface.csv", index=False)
    log("Interface calibration:\n" + cal[["family", "t_switch", "t_recover", "effective_durations"]].to_string(index=False))
    st = self_tests()
    st.to_csv(DIRS["tables"] / "solver_self_tests.csv", index=False)
    log("Closed-form solver self-tests:\n" + st.to_string(index=False))
    if st.abs_error.max() > 1e-3:
        raise RuntimeError("Closed-form solver self-test failed.")

    datasets = (["SmokeA", "SmokeB", "SmokeC"] if SMOKE else PRIOR_DATASETS + FRESH_DATASETS)
    if DATASET_FILTER:
        datasets = [d for d in datasets if d in DATASET_FILTER]
    protocol = {
        "script_version": SCRIPT_VERSION, "smoke": SMOKE, "ideal": IDEAL, "datasets": datasets,
        "prior_datasets": PRIOR_DATASETS, "fresh_datasets": FRESH_DATASETS, "seeds": SEEDS,
        "families": {f: FAMILIES[f].__dict__ for f in FAMILY_CODES}, "gains": GAINS.tolist(), "fracs": FRACS.tolist(),
        "n_bins": N_BINS, "alpha_rel": ALPHA_REL.tolist(), "k_shots": K_SHOTS, "lambda_prior": LAMBDA_PRIOR,
        "coral_eps": CORAL_EPS, "fir_taps": FIR_TAPS, "quad_taps": QUAD_TAPS, "probe": [PROBE_EPISODES, PROBE_LENGTH],
        "noise": {"d2d": SIGMA_D2D, "c2c": SIGMA_C2C, "read_decades": SIGMA_READ},
        "retention_per_symbol": RETENTION_PER_SYMBOL,
        "hypotheses": {
            "H1": "probe FIR alignment lowers cross-family regret vs moment matching (dataset-level one-sided Wilcoxon, Holm over label-free methods)",
            "H2": "within datasets, higher probe emulation R2 predicts lower probe-aligned regret (mean Spearman rho < 0)",
            "H3": "probe emulation R2 increases with source x target linear-capacity fraction (Spearman)",
            "H4": "source-selected operating point beats a random operating point on the target (dataset-level one-sided Wilcoxon)",
            "H5": "probe warm start beats scratch for k labels per class (Holm over k)",
        },
        "replication_unit": "dataset (pairs are seed medians, datasets are pair medians)",
    }
    phash = hashlib.sha256(json.dumps(protocol, sort_keys=True, default=str).encode()).hexdigest()[:12]
    protocol["protocol_hash"] = phash
    protocol["written_utc"] = datetime.now(timezone.utc).isoformat()
    ppath = DIRS["config"] / "protocol.json"
    if ppath.exists():
        old = json.loads(ppath.read_text())
        if old.get("protocol_hash") != phash:
            log(f"Protocol changed (old {old.get('protocol_hash')}, new {phash}); unit checkpoints from the old protocol are ignored.")
    save_json(protocol, ppath)
    log(f"Protocol hash {phash} written to {ppath}")

    instances: Dict[int, Dict[str, Instance]] = {}
    for seed in SEEDS:
        instances[seed] = {}
        for fam in FAMILY_CODES:
            for role in ("A", "B"):
                instances[seed][f"{fam}_{role}"] = make_instance(fam, seed, role)
    save_json({s: {k: v.params for k, v in d.items()} for s, d in instances.items()}, DIRS["config"] / "device_instances.json")

    tp = time.time()
    cap, emu, probe_traj = run_probe_phase(instances)
    log(f"Probe phase done in {(time.time() - tp) / 60:.1f} min")

    ref_rows, unit_frames, land_frames = [], [], []
    loaded: List[Dataset] = []
    for name in datasets:
        try:
            ds = synthetic_dataset(name, stable_seed(name)) if SMOKE else load_dataset(name)
        except Exception as exc:
            log(f"SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        loaded.append(ds)
        log(f"{name}: train={len(ds.y_train)} test={len(ds.y_test)} length={ds.U_train.shape[1]} classes={ds.n_classes}")
    loaded.sort(key=lambda d: (len(d.y_train) + len(d.y_test)) * d.U_train.shape[1])

    for ds in loaded:
        rp = DIRS["units"] / f"ref_{ds.name}_{phash}.json"
        if rp.exists():
            ref_rows.append(json.loads(rp.read_text()))
        else:
            r = reference_baselines(ds, SEEDS)
            save_json(r, rp)
            ref_rows.append(r)
        for seed in SEEDS:
            up = DIRS["units"] / f"units_{ds.name}_s{seed}_{phash}.csv"
            lp = DIRS["units"] / f"land_{ds.name}_s{seed}_{phash}.csv"
            if up.exists() and lp.exists():
                unit_frames.append(pd.read_csv(up)); land_frames.append(pd.read_csv(lp))
                log(f"{ds.name} seed {seed}: checkpoint reused")
                continue
            t0 = time.time()
            try:
                u, l = run_unit(ds, seed, instances[seed], probe_traj)
            except Exception:
                log(f"FAILED {ds.name} seed {seed}:\n{traceback.format_exc()}")
                continue
            u.to_csv(up, index=False); l.to_csv(lp, index=False)
            unit_frames.append(u); land_frames.append(l)
            fz = u[(u.pair_type == "cross")].groupby("method").regret.median()
            log(f"{ds.name} seed {seed}: {(time.time() - t0) / 60:.1f} min | median cross regret "
                f"zero={fz.get('zero_shot', np.nan):.3f} moment={fz.get('moment', np.nan):.3f} "
                f"probe_fir={fz.get('probe_fir', np.nan):.3f} full={fz.get('full_refit', np.nan):.3f}")

    if not unit_frames:
        raise RuntimeError("No dataset produced results.")
    units = pd.concat(unit_frames, ignore_index=True)
    land = pd.concat(land_frames, ignore_index=True)
    refs = pd.DataFrame(ref_rows)
    units.to_csv(DIRS["tables"] / "all_units.csv", index=False)
    land.to_csv(DIRS["tables"] / "operating_point_landscape_full.csv", index=False)
    res = analyse(units, land, refs, cap, emu)
    res["units_df"] = units
    try:
        make_figures(res)
    except Exception:
        log("Figure generation failed:\n" + traceback.format_exc())
    rep = res["report"]

    elapsed = (time.time() - t_start) / 60
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 110)
    print("COPY-PASTE REPORT START - MEMRISTIVE RESERVOIR PORTABILITY v2")
    print("=" * 110)
    print(f"Runtime: {elapsed:.1f} min | protocol hash {phash} | datasets {len(loaded)} | seeds {SEEDS} | ideal={IDEAL}")
    print("\nDEVICE INTERFACE")
    print(cal[["family", "name", "voltage_mode", "t_switch", "t_recover", "effective_durations"]].to_string(index=False))
    print("\nABSOLUTE PERFORMANCE (balanced accuracy)")
    print(rep["absolute"].drop(columns=["n_train", "n_test"], errors="ignore").to_string(index=False))
    print("\nOPERATING-POINT LANDSCAPE (normalized BA across the grid, medians)")
    print(rep["landscape"].to_string(index=False))
    print("\nCAPACITY AT SELECTED OPERATING POINTS (medians); raw input:", {k: round(v, 3) for k, v in rep["raw_capacity"].items()})
    print(rep["capacity"].to_string(index=False))
    print("\nMETHOD SUMMARY (regret to target oracle, normalized BA; hierarchical bootstrap CI)")
    print(rep["method_summary"].to_string(index=False))
    print("\nH1 LABEL-FREE ALIGNMENT VS MOMENT MATCHING (cross-family, dataset level)")
    print(rep["H1"].to_string(index=False))
    print("\nMIXED MODEL (probe FIR reduction vs moment, dataset random intercept):", rep["mixed_model"])
    print("\nH2 EMULATION R2 VS PROBE REGRET (within-dataset Spearman):", rep["H2"])
    print("\nH3 EMULATION R2 VS LINEAR-CAPACITY PRODUCT:", rep["H3"])
    print("\nH4 OPERATING-POINT PORTABILITY:", rep["H4"])
    print("\nH5 LABEL EFFICIENCY")
    print(rep["H5"].to_string(index=False))
    print("=" * 110)
    print("COPY-PASTE REPORT END")
    print("=" * 110)
    log(f"Done in {elapsed:.1f} min. Tables: {DIRS['tables']} | Figures: {DIRS['figures']}")


if __name__ == "__main__":
    main()
