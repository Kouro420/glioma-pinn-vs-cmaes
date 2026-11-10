#!/usr/bin/env python
"""Write the generator's tumor seed position of each of the 16 noise-free fit-grid cases, for the
refit with thresholds and tumor seed position fixed at the generator's values.

The generator places the tumor seed at the centroid of the SOURCE patient's tumor core
(src/data/make_synthetic.py, make_case: ``x0_mm = case.x0``); the fit places it at
the centroid of the SYNTHETIC case's own mask (src/data/predict_gbm.py,
``tumour_centroid``). In these cases (``outputs/synthetic_e1_both``) the truth is
generated on the 1 mm fit grid, so the generator's position in fit voxels is the
source centroid itself (no fine-grid shift).

The output, configs/x1_generator_seeds.json, is read by experiments/run_cmaes.py
through ``initial_condition.x0_file``, so the refit needs no source patient data.

    python scripts/x1_generator_seeds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.predict_gbm import tumour_centroid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO / "outputs" / "synthetic_e1_both")
    ap.add_argument("--source-root", type=Path,
                    default=REPO / "src" / "data" / "predict_gbm" / "predict_gbm")
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "x1_generator_seeds.json")
    args = ap.parse_args()

    cases = {}
    for case_dir in sorted(p for p in args.data_root.glob("SYN-*") if p.is_dir()):
        truth = json.loads((case_dir / "truth.json").read_text(encoding="utf-8"))
        if truth["fine_voxel_mm"] != 1.0:
            raise ValueError(f"{case_dir.name}: generated on a {truth['fine_voxel_mm']} mm grid; "
                             "this file is only valid for cases generated on the 1 mm fit grid")
        source = np.asarray(nib.load(args.source_root / truth["source_case"] / "tumor_seg.nii.gz").dataobj)
        own = np.asarray(nib.load(case_dir / "tumor_seg.nii.gz").dataobj)
        x0_gen, x0_fit = tumour_centroid(source), tumour_centroid(own)
        cases[case_dir.name] = {
            "source_case": truth["source_case"],
            "x0_vox": [float(v) for v in x0_gen],
            "fit_centroid_vox": [float(v) for v in x0_fit],
            "offset_mm": float(np.linalg.norm(x0_gen - x0_fit)),  # 1 mm voxels
        }
    if len(cases) != 16:
        raise ValueError(f"expected the 16 noise-free fit-grid cases, found {len(cases)}")
    record = {
        "what": "the generator's tumor seed position x0 (source tumor-core centroid), zero-based voxel "
                "indices of the 1 mm grid; read by the CMA-ES refits with the tumor seed position at the "
                "generator's value",
        "made_by": "scripts/x1_generator_seeds.py",
        "data_root": str(args.data_root.relative_to(REPO)).replace("\\", "/"),
        "cases": cases,
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    offsets = [c["offset_mm"] for c in cases.values()]
    print(f"{len(cases)} cases -> {args.out}; offset {min(offsets):.2f}-{max(offsets):.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
