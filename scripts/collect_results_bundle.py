"""
Collect every result needed for the manuscript into one zip file.

Gathers, from each portability run folder (v2 to v9) under MyDrive/memristive_device_generalization:
  tables/   all CSV and JSON result tables
  config/   protocols, preregistration record, calibrations, universal operating point
  figures/  PNG previews (downscaled to 2000 px wide; the PDFs are skipped because the figures will be redrawn)
  logs/     the last 3000 lines of each run log
plus a manifest with file sizes, row and column counts, and the software versions.

Paste into one Colab cell with Drive already mounted. The script does not mount Drive.
The zip is written to MyDrive/memristive_device_generalization/collected/ and offered for download.
"""

import io
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MEMRISTIVE_PROJECT_ROOT", "/content/drive/MyDrive/memristive_device_generalization"))
OUT_DIR = ROOT / "collected"
MAX_PNG_WIDTH = 2000
LOG_TAIL_LINES = 3000
TARGET_MB = 45.0          # if the bundle exceeds this, large CSVs are rewritten with 5 significant digits
SKIP_SUFFIXES = ("_smoke",)
EXPECTED = ["portability_v2", "portability_v3_diagnostics", "portability_v4_main_evaluation", "portability_v5_operating_point",
            "portability_v6_measured_validation", "portability_v7_measured_pair", "portability_v8_corrected_correspondence",
            "portability_v9_generality"]


def human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def csv_info(p: Path) -> dict:
    try:
        df = pd.read_csv(p, nrows=5)
        with p.open("rb") as f:
            rows = sum(1 for _ in f) - 1
        return {"rows": rows, "columns": list(df.columns)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def png_bytes(p: Path) -> bytes:
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(p)
        if im.width > MAX_PNG_WIDTH:
            h = int(im.height * MAX_PNG_WIDTH / im.width)
            im = im.resize((MAX_PNG_WIDTH, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return p.read_bytes()


def compact_csv(p: Path) -> bytes:
    df = pd.read_csv(p)
    return df.to_csv(index=False, float_format="%.5g").encode()


def build(compact_threshold_mb: float = None) -> tuple:
    runs = sorted([d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("portability_") and not d.name.endswith(SKIP_SUFFIXES)])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zpath = OUT_DIR / f"results_bundle_{stamp}.zip"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                "runs_found": [r.name for r in runs], "runs_missing": [e for e in EXPECTED if e not in [r.name for r in runs]],
                "compacted_csvs": [], "files": []}
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for run in runs:
            for sub in ["tables", "config"]:
                d = run / sub
                if not d.exists():
                    continue
                for p in sorted(d.rglob("*")):
                    if not p.is_file() or p.suffix.lower() not in (".csv", ".json", ".txt"):
                        continue
                    arc = f"{run.name}/{sub}/{p.relative_to(d)}"
                    size = p.stat().st_size
                    rec = {"path": arc, "bytes": size, "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
                    if p.suffix.lower() == ".csv":
                        rec.update(csv_info(p))
                        if compact_threshold_mb is not None and size > compact_threshold_mb * 1024 * 1024:
                            z.writestr(arc, compact_csv(p))
                            manifest["compacted_csvs"].append(arc)
                        else:
                            z.write(p, arc)
                    else:
                        z.write(p, arc)
                    manifest["files"].append(rec)
            fd = run / "figures"
            if fd.exists():
                for p in sorted(fd.glob("*.png")):
                    arc = f"{run.name}/figures/{p.name}"
                    z.writestr(arc, png_bytes(p))
                    manifest["files"].append({"path": arc, "bytes": p.stat().st_size, "note": f"downscaled to <= {MAX_PNG_WIDTH} px wide"})
            ld = run / "logs"
            if ld.exists():
                for p in sorted(ld.glob("*.log")):
                    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:]
                    arc = f"{run.name}/logs/{p.name}"
                    z.writestr(arc, "\n".join(lines))
                    manifest["files"].append({"path": arc, "bytes": p.stat().st_size, "note": f"last {LOG_TAIL_LINES} lines"})
        meas = ROOT / "data" / "measured"
        if meas.exists():
            for p in sorted(meas.iterdir()):
                if p.is_file():
                    manifest["files"].append({"path": f"(not bundled) data/measured/{p.name}", "bytes": p.stat().st_size})
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2, default=str))
    return zpath, manifest


def main() -> None:
    t0 = time.time()
    if not ROOT.exists():
        raise FileNotFoundError(f"{ROOT} not found. Mount Drive first, or set MEMRISTIVE_PROJECT_ROOT.")
    zpath, manifest = build()
    mb = zpath.stat().st_size / 1024 / 1024
    if mb > TARGET_MB:
        print(f"Bundle is {mb:.1f} MB; rewriting CSVs above 5 MB with 5 significant digits ...")
        zpath.unlink()
        zpath, manifest = build(compact_threshold_mb=5.0)
        mb = zpath.stat().st_size / 1024 / 1024
    print("=" * 90)
    print(f"Runs found ({len(manifest['runs_found'])}): {', '.join(manifest['runs_found'])}")
    if manifest["runs_missing"]:
        print(f"Runs missing: {', '.join(manifest['runs_missing'])}")
    n_csv = sum(1 for f in manifest["files"] if f["path"].endswith(".csv"))
    n_png = sum(1 for f in manifest["files"] if f["path"].endswith(".png"))
    n_json = sum(1 for f in manifest["files"] if f["path"].endswith(".json"))
    print(f"Files: {n_csv} CSV, {n_json} JSON, {n_png} PNG | compacted CSVs: {len(manifest['compacted_csvs'])}")
    print(f"Bundle: {zpath} ({mb:.1f} MB) | {time.time() - t0:.0f} s")
    print("=" * 90)
    try:
        from google.colab import files
        files.download(str(zpath))
    except Exception:
        print("Download it from the path above (Drive) and attach it in the chat.")


if __name__ == "__main__":
    main()
