#!/usr/bin/env python
"""The objective at each arm's estimate.

For every synthetic case with both a PINN and a CMA-ES record, solve the forward model once at
each arm's (mu_D, mu_rho) and once at the truth, minimize the two thresholds on the stored field
exactly as scripts/profile_objective.py does, and report the objective (prior off). The outcome
rule: the PINN is "at the best point" if its objective is within 1.05 x CMA-ES's in >= 12 of 16
cases.

Both arms are scored with optimal thresholds, so the comparison is on (mu_D, mu_rho) alone.

    python scripts/objective_at_estimates.py --pinn outputs/runs_syn_np_paper --cmaes outputs/runs_syn_np --out outputs/tables/e2_objective.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.profile_objective import ThresholdSearch, build, git_commit  # noqa: E402
from src.data.predict_gbm import find_case  # noqa: E402
from src.models.fisher_kpp import solve_fisher_kpp  # noqa: E402

RATIO = 1.05
NEEDED = 12


def records(folder: str, method: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(folder, f"*__{method}.json")):
        out[os.path.basename(f).split("__")[0]] = json.load(open(f, encoding="utf-8"))
    return out


def evaluate(case, scaling, u0, objective, search, config, mu_d, mu_rho, device):
    d_eff, rho_eff = scaling.to_physical(mu_d, mu_rho)
    solution = solve_fisher_kpp(
        case.p_wm, case.p_gm, case.phi, u0, d_w=d_eff, rho=rho_eff, t_final=1.0,
        factor=config["model"]["factor"], n_save=2, h=config["model"]["voxel_mm"],
        cfl_safety=config["model"]["cfl_safety"], device=device,
    )
    search.load(solution.final)
    th_wt, th_tc = search.minimise("wt"), search.minimise("tc")
    terms = objective(solution.final, mu_d=mu_d, mu_rho=mu_rho, th_wt=th_wt, th_tc=th_tc)
    return {"mu_d": mu_d, "mu_rho": mu_rho, "total": float(terms.total),
            "seg_wt": float(terms.seg_wt), "seg_tc": float(terms.seg_tc),
            "th_wt": th_wt, "th_tc": th_tc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", required=True)
    ap.add_argument("--cmaes", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-root", type=Path, default=REPO / "outputs" / "synthetic_local")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--patients", action="store_true",
                    help="real cases (no truth): --data-root is the PREDICT-GBM folder; rule not applied")
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    P, C = records(args.pinn, "pinn"), records(args.cmaes, "cmaes")
    cases = sorted(set(P) & set(C))
    rows = []
    for k in cases:
        started = time.perf_counter()
        paths = find_case(args.data_root, k) if args.patients else None
        case, scaling, u0, objective = build(k, args.data_root, config, 0.0, device, paths=paths)
        search = ThresholdSearch(objective, torch.device(device))
        m = C[k]["metrics"]
        row = {"case": k}
        points = {"pinn": (P[k]["estimate"]["mu_d"], P[k]["estimate"]["mu_rho"]),
                  "cmaes": (C[k]["estimate"]["mu_d"], C[k]["estimate"]["mu_rho"])}
        if not args.patients:
            points["truth"] = (m["mu_d_true"], m["mu_rho_true"])
        for name, (a, b) in points.items():
            row[name] = evaluate(case, scaling, u0, objective, search, config, float(a), float(b), device)
        row["ratio_pinn_over_cmaes"] = row["pinn"]["total"] / row["cmaes"]["total"]
        if "truth" in row:
            row["ratio_truth_over_cmaes"] = row["truth"]["total"] / row["cmaes"]["total"]
        rows.append(row)
        truth_text = (f"  truth/cmaes {row['ratio_truth_over_cmaes']:.3f}"
                      if "truth" in row else "")
        print(f"{k}: pinn/cmaes {row['ratio_pinn_over_cmaes']:.3f}{truth_text}  "
              f"({time.perf_counter() - started:.0f} s)", flush=True)

    within = sum(r["ratio_pinn_over_cmaes"] <= RATIO for r in rows)
    ratios = sorted(r["ratio_pinn_over_cmaes"] for r in rows)
    # "12 of 16" scales with the number of cases: the 12 design-range cases need "9 of 12".
    needed = -(-NEEDED * len(rows) // 16)
    summary = {
        "n": len(rows),
        "rule": None if args.patients else f"PINN within {RATIO} x CMA-ES in >= {needed} of {len(rows)}",
        "pinn_within": within,
        "rule_met": None if args.patients else within >= needed,
        "ratio_min": ratios[0], "ratio_median": statistics.median(ratios), "ratio_max": ratios[-1],
        "truth_below_cmaes": None if args.patients else sum(r["ratio_truth_over_cmaes"] < 1.0 for r in rows),
        "pinn_below_cmaes": sum(r["ratio_pinn_over_cmaes"] < 1.0 for r in rows),
        "missing": sorted(set(C) - set(P)),
        "provenance": {"git_commit": git_commit(), "torch": torch.__version__, "device": device,
                       "config": str(args.config), "data_root": str(args.data_root), "prior_scale": 0.0},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
