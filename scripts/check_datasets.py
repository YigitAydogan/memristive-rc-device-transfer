"""
Records the official training and test sizes, series length and number of classes of the 28 UCR datasets used in the
study, to verify table 2 of the manuscript. Uses the dataset cache of the simulation scripts.
Set MEMRISTIVE_PROJECT_ROOT to choose the project root; on Colab the default is the Drive folder used by the scripts.
"""
import importlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MEMRISTIVE_PROJECT_ROOT", "/content/drive/MyDrive/memristive_device_generalization"))
DATA_DIR = ROOT / "data" / "tsml_ucr"
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    importlib.import_module("aeon")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "aeon>=1.0"])
from aeon.datasets import load_classification

DEVELOPMENT = ["ECG5000", "FordA", "Wafer", "UWaveGestureLibraryX", "CricketX", "RefrigerationDevices", "Earthquakes",
               "StarLightCurves", "GunPoint", "TwoPatterns", "SyntheticControl", "Trace", "FaceAll", "Strawberry", "Yoga",
               "ECG200", "Plane", "Symbols"]
HELD_OUT = ["MoteStrain", "SonyAIBORobotSurface1", "Coffee", "Meat", "Lightning7", "MedicalImages",
            "ProximalPhalanxOutlineCorrect", "Ham", "OSULeaf", "Fish"]

rows = []
for name in DEVELOPMENT + HELD_OUT:
    Xtr, ytr = load_classification(name=name, split="train", extract_path=str(DATA_DIR))
    Xte, yte = load_classification(name=name, split="test", extract_path=str(DATA_DIR))
    L = {np.asarray(x).shape[-1] for x in list(Xtr) + list(Xte)}
    rows.append({"dataset": name, "set": "D" if name in DEVELOPMENT else "H", "n_train": len(ytr), "n_test": len(yte),
                 "length": max(L), "equal_length": len(L) == 1,
                 "classes": len(set(np.asarray(ytr).astype(str)) | set(np.asarray(yte).astype(str)))})
df = pd.DataFrame(rows)
out = ROOT / "dataset_check"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "dataset_sizes.csv", index=False)
print(df.to_string(index=False))
