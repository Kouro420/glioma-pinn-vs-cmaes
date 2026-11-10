#!/usr/bin/env python
"""Re-score the truth and the refit point with the thresholds and the tumor seed position fixed.

On the 16 noise-free cases (``outputs/synthetic_e1_both``), with both thresholds at the
generator's values and the tumor seed position at the generator's (as in the refit with both fixed), the
forward model is solved once at the truth and once at the CMA-ES point of that refit
(``outputs/runs_x1_thfix_seed``), each with the second tissue renormalization inside the
tumor ON (what every fit did) and OFF (the generator's own field). Each solution is then
scored, prior off, with three versions of the mask term:

* ``a20``  -- the shared objective, smoothed Heaviside with steepness 20;
* ``a200`` -- the same with steepness 200;
* ``hard`` -- the indicator 1{u > threshold}; the mask term is then the normalized, phi-weighted
  count of mismatched voxels.

Reported per case and setting: the ratio objective(truth) / objective(refit point) and whether
the truth fits at least as well (ratio <= 1). The two points are solved at their physical
``d_eff``, ``rho_eff`` (truth.json and the run record), so no scale conversion enters. As a check,
the ``a20`` / ON score at the refit point is compared with the run record's own final mask terms.

Writes ``outputs/tables/b1_b2_rescore.json``. One GPU; 64 solves.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.profile_objective import build, git_commit  # noqa: E402
from src.models.fisher_kpp import gaussian_seed, solve_fisher_kpp  # noqa: E402

DATA = REPO / "outputs" / "synthetic_e1_both"
RUNS = REPO / "outputs" / "runs_x1_thfix_seed"
SEEDS = REPO / "configs" / "x1_generator_seeds.json"
SEG_WEIGHT = 1e-5  # configs/base.yaml objective.weights.seg_wt = seg_tc; prior off


def mask_term(u, mask, phi, threshold, norm, mode):
    if mode == "hard":
        pred = (u > threshold).astype(np.float32)
    else:
        a = 20.0 if mode == "a20" else 200.0
        pred = 1.0 / (1.0 + np.exp(-a * (u - threshold)))
    return float(np.mean(((pred - mask) * phi) ** 2) / norm)


def score(u, case, th_wt, th_tc, mode):
    mw, mt = case.mask_wt.astype(np.float32), case.mask_tc.astype(np.float32)
    phi = case.phi.astype(np.float32)
    s_wt = mask_term(u, mw, phi, th_wt, float(np.mean(mw ** 2)), mode)
    s_tc = mask_term(u, mt, phi, th_tc, float(np.mean(mt ** 2)), mode)
    return {"total": SEG_WEIGHT * (s_wt + s_tc), "seg_wt": s_wt, "seg_tc": s_tc}


def main() -> int:
    config = yaml.safe_load((REPO / "configs" / "base.yaml").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))["cases"]
    records = {p.name.split("__")[0]: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(RUNS.glob("*__cmaes*.json"))}
    rows = []
    for case_id in sorted(records):
        started = time.perf_counter()
        truth = json.loads((DATA / case_id / "truth.json").read_text(encoding="utf-8"))
        rec = records[case_id]
        th_wt, th_tc = float(truth["th_wt"]), float(truth["th_tc"])
        x0 = np.asarray(seeds[case_id]["x0_vox"], dtype=float)
        points = {"truth": (truth["d_eff"], truth["rho_eff"]),
                  "refit": (rec["estimate"]["d_eff"], rec["estimate"]["rho_eff"])}
        row = {"case": case_id, "brain": truth["source_case"], "mu_d_true": truth["mu_d_true"],
               "mu_rho_target": truth["mu_rho_target"],
               "refit_mu_d": rec["estimate"]["mu_d"], "settings": {}}
        for setting, renorm in (("on", True), ("off", False)):
            case, _, _, _ = build(case_id, DATA, config, 0.0, device, renormalise_tumour=renorm)
            u0 = gaussian_seed(case.shape, x0, h_init=config["initial_condition"]["h_init"],
                               r_init=config["initial_condition"]["r_init"],
                               h=config["model"]["voxel_mm"])
            out = {}
            for name, (d_eff, rho_eff) in points.items():
                u = np.asarray(solve_fisher_kpp(
                    case.p_wm, case.p_gm, case.phi, u0, d_w=float(d_eff), rho=float(rho_eff),
                    t_final=1.0, factor=config["model"]["factor"], n_save=2,
                    h=config["model"]["voxel_mm"], cfl_safety=config["model"]["cfl_safety"],
                    device=device).final, dtype=np.float32)
                out[name] = {mode: score(u, case, th_wt, th_tc, mode) for mode in ("a20", "a200", "hard")}
            row["settings"][setting] = {
                mode: {"truth": out["truth"][mode]["total"], "refit": out["refit"][mode]["total"],
                       "ratio": out["truth"][mode]["total"] / out["refit"][mode]["total"]}
                for mode in ("a20", "a200", "hard")}
            if setting == "on":
                row["check_refit_a20_on"] = {"seg_wt": out["refit"]["a20"]["seg_wt"],
                                             "seg_tc": out["refit"]["a20"]["seg_tc"]}
        rows.append(row)
        r = row["settings"]
        print(f"{case_id}: a20 on {r['on']['a20']['ratio']:.3f} off {r['off']['a20']['ratio']:.3f} | "
              f"a200 off {r['off']['a200']['ratio']:.3f} | hard off {r['off']['hard']['ratio']:.3f} "
              f"({time.perf_counter() - started:.0f} s)", flush=True)

    summary = {}
    for setting in ("on", "off"):
        for mode in ("a20", "a200", "hard"):
            ratios = [r["settings"][setting][mode]["ratio"] for r in rows]
            summary[f"{mode}_{setting}"] = {
                "truth_at_least_as_good": sum(x <= 1.0 for x in ratios),
                "median_ratio": statistics.median(ratios), "min": min(ratios), "max": max(ratios),
                "per_brain_truth_at_least_as_good": {
                    b: sum(r["settings"][setting][mode]["ratio"] <= 1.0 for r in rows if r["brain"] == b)
                    for b in sorted({r["brain"] for r in rows})}}
    out = {"what": __doc__.split("Writes")[0].strip(), "n": len(rows), "summary": summary, "cases": rows,
           "provenance": {"git_commit": git_commit(), "torch": torch.__version__, "device": device,
                          "config": "configs/base.yaml", "data": str(DATA.relative_to(REPO)),
                          "refit_records": str(RUNS.relative_to(REPO))}}
    (REPO / "outputs" / "tables" / "b1_b2_rescore.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
