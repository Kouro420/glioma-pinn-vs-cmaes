#!/usr/bin/env python
"""Check of the tumor seed position, design-range cases: the objective at the truth against the refit point with the generator's tumor seed position.

For each of the 12 design-range cases (``outputs/synthetic_env``), the forward model is solved with the tumor seed position
at the generator's position (``configs/env_generator_seeds.json``, as in the refit with that position) once at the
truth (``truth.json`` d_eff, rho_eff) and once at the CMA-ES point of that refit (``outputs/runs_d104c``), each
scored on the shared objective (prior off, a = 20) with both thresholds re-optimized on the stored field, as
``scripts/objective_at_estimates.py`` does. Reported per case: the ratio objective(truth) / objective(refit);
<= 1 means the truth fits at least as well, so the search stopped short of the truth's level. All these
runs stopped on the 600-evaluation budget, so this is the check of whether the search had finished. As a check,
the refit point's re-optimized score is compared with the record's best score (its thresholds were fitted).

Writes ``outputs/tables/d104c_objective.json``. One GPU; 24 solves.
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

from scripts.profile_objective import ThresholdSearch, build, git_commit  # noqa: E402
from src.models.fisher_kpp import gaussian_seed, solve_fisher_kpp  # noqa: E402

DATA = REPO / "outputs" / "synthetic_env"
RUNS = REPO / "outputs" / "runs_d104c"
SEEDS = REPO / "configs" / "env_generator_seeds.json"


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
        case, scaling, _, objective = build(case_id, DATA, config, 0.0, device)
        x0 = np.asarray(seeds[case_id]["x0_vox"], dtype=float)
        u0 = gaussian_seed(case.shape, x0, h_init=config["initial_condition"]["h_init"],
                           r_init=config["initial_condition"]["r_init"], h=config["model"]["voxel_mm"])
        search = ThresholdSearch(objective, torch.device(device))
        points = {"truth": (truth["d_eff"], truth["rho_eff"], truth["mu_d_true"], truth["mu_rho_true"]),
                  "refit": (rec["estimate"]["d_eff"], rec["estimate"]["rho_eff"],
                            rec["estimate"]["mu_d"], rec["estimate"]["mu_rho"])}
        row = {"case": case_id, "brain": truth["source_case"], "seed_offset_mm": seeds[case_id]["offset_mm"],
               "mu_d_true": truth["mu_d_true"], "refit_mu_d": rec["estimate"]["mu_d"],
               "record_best_score": min(h["score"] for h in rec["history"])}
        for name, (d_eff, rho_eff, mu_d, mu_rho) in points.items():
            solution = solve_fisher_kpp(
                case.p_wm, case.p_gm, case.phi, u0, d_w=float(d_eff), rho=float(rho_eff), t_final=1.0,
                factor=config["model"]["factor"], n_save=2, h=config["model"]["voxel_mm"],
                cfl_safety=config["model"]["cfl_safety"], device=device)
            search.load(solution.final)
            th_wt, th_tc = search.minimise("wt"), search.minimise("tc")
            terms = objective(solution.final, mu_d=float(mu_d), mu_rho=float(mu_rho), th_wt=th_wt, th_tc=th_tc)
            row[name] = {"total": float(terms.total), "th_wt": th_wt, "th_tc": th_tc}
        row["ratio_truth_over_refit"] = row["truth"]["total"] / row["refit"]["total"]
        row["check_refit_over_record"] = row["refit"]["total"] / row["record_best_score"]
        rows.append(row)
        print(f"{case_id}: truth/refit {row['ratio_truth_over_refit']:.3f}  (refit re-scored / record "
              f"{row['check_refit_over_record']:.3f}; {time.perf_counter() - started:.0f} s)", flush=True)

    summary = {}
    for b in sorted({r["brain"] for r in rows}):
        ratios = [r["ratio_truth_over_refit"] for r in rows if r["brain"] == b]
        summary[b] = {"n": len(ratios), "truth_at_least_as_good": sum(x <= 1.0 for x in ratios),
                      "median_ratio": statistics.median(ratios), "min": min(ratios), "max": max(ratios)}
    out = {"what": __doc__.split("Writes")[0].strip(), "summary": summary, "cases": rows,
           "provenance": {"git_commit": git_commit(), "torch": torch.__version__, "device": device,
                          "config": "configs/base.yaml", "data": str(DATA.relative_to(REPO)),
                          "refit_records": str(RUNS.relative_to(REPO)), "seeds": str(SEEDS.relative_to(REPO))}}
    (REPO / "outputs" / "tables" / "d104c_objective.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
