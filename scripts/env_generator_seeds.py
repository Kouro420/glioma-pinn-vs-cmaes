#!/usr/bin/env python
"""Write the generator's tumor seed position of each design-range case, for the refit with that position.

Like scripts/x1_generator_seeds.py, but the design-range cases were generated on the 0.5 mm
grid: the generator places the tumor seed at the SOURCE patient's tumor-core centroid, and the
fine grid (align_corners=False) puts that point 0.25 voxel lower on every axis of the 1 mm
fit grid. The source centroids are read from configs/x1_generator_seeds.json (same two
source brains), so no patient data is needed.

Writes configs/env_generator_seeds.json, read by experiments/run_cmaes.py through
``initial_condition.x0_file``.

    python scripts/env_generator_seeds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.predict_gbm import tumour_centroid  # noqa: E402

DATA = REPO / "outputs" / "synthetic_env"
X1 = REPO / "configs" / "x1_generator_seeds.json"
OUT = REPO / "configs" / "env_generator_seeds.json"
FINE_SHIFT = -0.25  # voxels per axis, 0.5 mm generation grid -> 1 mm fit grid


def main() -> int:
    source = {}
    for case in json.loads(X1.read_text(encoding="utf-8"))["cases"].values():
        source.setdefault(case["source_case"], np.asarray(case["x0_vox"], dtype=float))
    cases = {}
    for case_dir in sorted(p for p in DATA.glob("ENV-*") if p.is_dir()):
        truth = json.loads((case_dir / "truth.json").read_text(encoding="utf-8"))
        if truth["fine_voxel_mm"] != 0.5:
            raise ValueError(f"{case_dir.name}: expected the 0.5 mm generation grid")
        x0_gen = source[truth["source_case"]] + FINE_SHIFT
        x0_fit = tumour_centroid(np.asarray(nib.load(case_dir / "tumor_seg.nii.gz").dataobj))
        cases[case_dir.name] = {"source_case": truth["source_case"],
                                "x0_vox": [float(v) for v in x0_gen],
                                "fit_centroid_vox": [float(v) for v in x0_fit],
                                "offset_mm": float(np.linalg.norm(x0_gen - x0_fit))}
    OUT.write_text(json.dumps({
        "what": "the generator's tumor seed position x0 of the design-range cases (source tumor-core centroid, shifted "
                "-0.25 voxel per axis for the 0.5 mm generation grid), zero-based voxel indices of the "
                "1 mm grid",
        "made_by": "scripts/env_generator_seeds.py",
        "cases": cases}, indent=2) + "\n", encoding="utf-8")
    for name, c in cases.items():
        print(f"{name}: offset {c['offset_mm']:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
