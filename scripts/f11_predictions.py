#!/usr/bin/env python
"""Data of Figure 1 (script label F11): what the fitted models predict, on three cases chosen by a rule.

For each case the finite-difference solution at each arm's estimate is computed once,
multiplied by the phase field and thresholded at that arm's own fitted thresholds, as
``main()`` of ``experiments/run_cmaes.py`` and of ``experiments/run_pinn.py`` do when they
compute the Dice of a run record (``segmentation_dice``). The run record's ``d_eff`` and
``rho_eff`` are used, so the PINN is solved at the values its own Dice was computed at. The
Dice of every panel is recomputed on the whole volume and compared with the run record.

Case rule (ties to the lower ID):
* synthetic: SYN-RHUH-0002-d1.2-r0.6, the case in row 1 of Figure 1;
* RHUH: of the 20 prior-off patients, the one whose per-patient Dice gap (CMA-ES minus the
  PINN on the training schedule of Zhang et al., mean of the whole-tumor and core gaps) is
  nearest the median gap;
* clinical pilot (17 patients, not public): the case with the median CMA-ES Dice WT of the 17
  (the 9th of 17).

Writes ``outputs/tables/f11_slices.npz`` (one axial slice per panel, R-A-S, through the core
centroid) and ``outputs/tables/f11_predictions.json`` (rule, cases, estimates, Dice check).
Run from the repository root; needs one GPU. The clinical pilot row needs ``EVKB_ROOT``, the
local folder of the pilot data. Without EVKB_ROOT the figure has the synthetic and RHUH rows only.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.profile_objective import build, git_commit  # noqa: E402
from src.data.predict_gbm import find_case  # noqa: E402
from src.evaluation.metrics import segmentation_dice  # noqa: E402
from src.models.fisher_kpp import solve_fisher_kpp  # noqa: E402

SYN_ROOT = REPO / "outputs" / "synthetic_local"
RHUH_ROOT = REPO / "src" / "data" / "predict_gbm" / "predict_gbm"
# The clinical pilot data are not public; set EVKB_ROOT to their local folder to draw row 3.
# Without EVKB_ROOT, F11 has the synthetic and RHUH rows only.
EVKB_ROOT = Path(os.environ["EVKB_ROOT"]) if os.environ.get("EVKB_ROOT") else None
EVKB_BUNDLE = EVKB_ROOT / "evkb_pilot" if EVKB_ROOT else Path("__no_evkb__")
EVKB_PREP = EVKB_ROOT / "nifti_prep" if EVKB_ROOT else Path("__no_evkb__")
EVKB_RUNS = EVKB_ROOT / "pilot_runs" if EVKB_ROOT else Path("__no_evkb__")
EVKB_PINN_RUNS = EVKB_ROOT / "pilot_runs_pinn" if EVKB_ROOT else Path("__no_evkb__")
SYN_CASE = "SYN-RHUH-0002-d1.2-r0.6"
TOLERANCE = 0.005  # largest Dice difference against the record accepted as "equal"


def records(folder: Path, method: str) -> dict[str, dict]:
    return {r["case_id"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob(f"*__{method}*.json")))}


def nearest_median(values: dict[str, float]) -> tuple[str, float]:
    median = statistics.median(values.values())
    key = min(sorted(values), key=lambda k: (abs(values[k] - median), k))
    return key, median


def to_ras(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ornt = nib.orientations.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(("R", "A", "S"))
    return nib.orientations.apply_orientation(volume, nib.orientations.ornt_transform(ornt, target))


def predict(case, u0, config, record, device) -> tuple[np.ndarray, dict]:
    est = record["estimate"]
    final = solve_fisher_kpp(
        case.p_wm, case.p_gm, case.phi, u0, d_w=float(est["d_eff"]), rho=float(est["rho_eff"]),
        t_final=1.0, factor=config["model"]["factor"], n_save=2, h=config["model"]["voxel_mm"],
        cfl_safety=config["model"]["cfl_safety"], device=device,
    ).final
    density = np.asarray(final, dtype=np.float32) * case.phi
    dice = segmentation_dice(density, case.segmentation, th_wt=est["th_wt"], th_tc=est["th_tc"])
    check = {"mu_d": est["mu_d"], "mu_rho": est["mu_rho"], "th_wt": est["th_wt"], "th_tc": est["th_tc"],
             "dice_wt": dice["dice_wt"], "dice_tc": dice["dice_tc"],
             "record_dice_wt": record["metrics"]["dice_wt"], "record_dice_tc": record["metrics"]["dice_tc"]}
    check["equal"] = (abs(check["dice_wt"] - check["record_dice_wt"]) <= TOLERANCE
                      and abs(check["dice_tc"] - check["record_dice_tc"]) <= TOLERANCE)
    return density, check


def main() -> int:
    config = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- the case rule, on the records only ---
    cm, pn = records(REPO / "outputs" / "runs_patient_np", "cmaes"), records(REPO / "outputs" / "runs_patient_np_paper", "pinn")
    gaps = {k: 0.5 * ((cm[k]["metrics"]["dice_wt"] - pn[k]["metrics"]["dice_wt"])
                      + (cm[k]["metrics"]["dice_tc"] - pn[k]["metrics"]["dice_tc"]))
            for k in sorted(set(cm) & set(pn))}
    rhuh_case, gap_median = nearest_median(gaps)
    # The clinical pilot row needs the pilot images, which are not public; without them, F11 has
    # the synthetic and RHUH rows only.
    with_evkb = EVKB_RUNS.is_dir() and EVKB_BUNDLE.is_dir()
    ev = records(EVKB_RUNS, "cmaes") if with_evkb else {}
    order = sorted(ev, key=lambda k: (ev[k]["metrics"]["dice_wt"], k))
    evkb_case = order[len(order) // 2] if order else None
    evkb_index = sorted(ev).index(evkb_case) + 1 if order else None  # "pilot case k": position in ID order, 1-17

    rows = [
        {"row": "synthetic", "case": SYN_CASE, "root": SYN_ROOT, "paths": None,
         "arms": {"cmaes": records(REPO / "outputs" / "runs_syn_np", "cmaes")[SYN_CASE],
                  "pinn": records(REPO / "outputs" / "runs_syn_np_paper", "pinn")[SYN_CASE]},
         "background": ("wm_pbmap", None)},
        {"row": "rhuh", "case": rhuh_case, "root": RHUH_ROOT, "paths": find_case(RHUH_ROOT, rhuh_case),
         "arms": {"cmaes": cm[rhuh_case], "pinn": pn[rhuh_case]},
         "background": ("flair", None)},
    ]
    if with_evkb:
        rows.append(
            {"row": "evkb", "case": evkb_case, "root": EVKB_BUNDLE, "paths": find_case(EVKB_BUNDLE, evkb_case),
             # The PINN on this one pilot case, if its record is present.
             "arms": {"cmaes": ev[evkb_case],
                      **({"pinn": records(EVKB_PINN_RUNS, "pinn")[evkb_case]}
                         if (EVKB_PINN_RUNS / f"{evkb_case}__pinn.json").exists() else {})},
             "background": ("flair", EVKB_PREP / evkb_case / "flair_bet.nii.gz")})

    slices, report = {}, {"rule": __doc__.split("Case rule")[1].split("Writes")[0].strip(),
                          "rhuh_gap_median": gap_median, "rhuh_gaps": gaps,
                          "evkb_dice_wt_sorted": [[k, ev[k]["metrics"]["dice_wt"]] for k in order],
                          "evkb_pilot_index": evkb_index, "rows": {}}
    for row in rows:
        case, _, u0, _ = build(row["case"], row["root"], config, 0.0, device, paths=row["paths"])
        ref = Path(row["paths"].tumor_seg) if row["paths"] else row["root"] / row["case"] / "tumor_seg.nii.gz"
        affine = nib.load(str(ref)).affine
        tag = row["row"]
        mask_wt, mask_tc = to_ras(case.mask_wt, affine), to_ras(case.mask_tc, affine)
        brain = to_ras(case.brain_mask, affine) > 0
        z = int(round(np.argwhere(mask_tc if mask_tc.any() else mask_wt)[:, 2].mean()))
        kind, path = row["background"]
        if kind == "wm_pbmap":
            back = to_ras(case.p_wm, affine)
        else:
            path = path or Path(row["paths"].tumor_seg).parent / "flair_bet_normalized.nii.gz"
            image = nib.load(str(path))
            data = np.asarray(image.dataobj, dtype=np.float32)
            if data.shape != case.mask_wt.shape:
                raise ValueError(f"{path}: shape {data.shape} != case grid {case.mask_wt.shape}")
            back = to_ras(data, image.affine)
            if back.shape != mask_wt.shape:
                raise ValueError(f"{path}: orientation differs from the masks")
            hi = np.percentile(back[brain], 99.5)
            back = np.clip(back / hi, 0, 1)
        slices[f"{tag}_back"] = back[:, :, z]
        slices[f"{tag}_obs_wt"], slices[f"{tag}_obs_tc"] = mask_wt[:, :, z], mask_tc[:, :, z]
        slices[f"{tag}_brain3d_inplane"] = brain.any(axis=2)
        out = {"case": row["case"], "z_ras": z, "arms": {}}
        for arm, record in row["arms"].items():
            density, check = predict(case, u0, config, record, device)
            density = to_ras(density, affine)
            slices[f"{tag}_{arm}_wt"] = density[:, :, z] > record["estimate"]["th_wt"]
            slices[f"{tag}_{arm}_tc"] = density[:, :, z] > record["estimate"]["th_tc"]
            out["arms"][arm] = check
            print(f"{tag} {row['case']} {arm}: Dice WT {check['dice_wt']:.4f} (record {check['record_dice_wt']:.4f}), "
                  f"TC {check['dice_tc']:.4f} (record {check['record_dice_tc']:.4f}) -> "
                  f"{'equal' if check['equal'] else 'DIFFERENT'}", flush=True)
        report["rows"][tag] = out

    report["provenance"] = {"git_commit": git_commit(), "torch": torch.__version__, "device": device,
                            "config": "configs/base.yaml", "tolerance": TOLERANCE}
    tables = REPO / "outputs" / "tables"
    np.savez_compressed(tables / "f11_slices.npz", **slices)
    (tables / "f11_predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RHUH case {rhuh_case} (median gap {gap_median:.4f}); "
          + (f"clinical pilot case {evkb_index}" if with_evkb else "no clinical pilot data: two rows only"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
