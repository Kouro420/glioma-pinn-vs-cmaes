#!/usr/bin/env python
"""What the second tissue renormalization does to the objective.

A synthetic case's tissue maps were renormalized inside the SOURCE tumor by the generator
(``load_case`` on the source brain), and ``load_case`` renormalizes them again inside the
SYNTHETIC tumor when the case is fitted. The fitted model therefore does not see the
generator's diffusion field. For each of the 16 noise-free cases on the fit grid
(``outputs/synthetic_e1_both``) this script loads the case twice -- with the second
renormalization (what every fit did) and without it -- and in each setting scores the
objective, thresholds re-optimized as in ``scripts/objective_at_estimates.py``, at the truth
and at CMA-ES's estimate on these cases (``outputs/runs_e1_both``). It also reports the size
of the change of the diffusivity profile P = p_w + p_g / c inside the whole tumor. A
measurement; no outcome rule.

    python scripts/x4_renormalisation.py
"""

from __future__ import annotations

import argparse
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

from scripts.objective_at_estimates import evaluate, records  # noqa: E402
from scripts.profile_objective import ThresholdSearch, build, git_commit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmaes", default=str(REPO / "outputs" / "runs_e1_both"))
    ap.add_argument("--data-root", type=Path, default=REPO / "outputs" / "synthetic_e1_both")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "tables" / "x4_renormalisation.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    factor = config["model"]["factor"]
    C = records(args.cmaes, "cmaes")
    rows = []
    for k in sorted(C):
        started = time.perf_counter()
        m = C[k]["metrics"]
        row = {"case": k}
        profiles = {}
        for label, flag in (("twice", True), ("once", False)):
            case, scaling, u0, objective = build(k, args.data_root, config, 0.0, device,
                                                 renormalise_tumour=flag)
            search = ThresholdSearch(objective, torch.device(device))
            profiles[label] = (np.asarray(case.p_wm) + np.asarray(case.p_gm) / factor,
                               np.asarray(case.mask_wt, dtype=bool))
            row[label] = {
                "truth": evaluate(case, scaling, u0, objective, search, config,
                                  float(m["mu_d_true"]), float(m["mu_rho_true"]), device),
                "cmaes": evaluate(case, scaling, u0, objective, search, config,
                                  float(C[k]["estimate"]["mu_d"]),
                                  float(C[k]["estimate"]["mu_rho"]), device),
            }
            row[label]["truth_over_cmaes"] = row[label]["truth"]["total"] / row[label]["cmaes"]["total"]
        (p2, wt), (p1, _) = profiles["twice"], profiles["once"]
        inside = wt & (p1 > 0)
        row["profile_change_in_wt"] = {
            "mean_rel": float(np.mean(p2[inside] / p1[inside] - 1.0)) if inside.any() else None,
            "max_rel": float(np.max(p2[inside] / p1[inside] - 1.0)) if inside.any() else None,
        }
        rows.append(row)
        print(f"{k}: truth/cmaes twice {row['twice']['truth_over_cmaes']:.3f}  once "
              f"{row['once']['truth_over_cmaes']:.3f}  P change in WT mean "
              f"{row['profile_change_in_wt']['mean_rel']:+.3f}  ({time.perf_counter() - started:.0f} s)",
              flush=True)

    summary = {
        "n": len(rows),
        "truth_below_cmaes_twice": sum(r["twice"]["truth_over_cmaes"] < 1.0 for r in rows),
        "truth_below_cmaes_once": sum(r["once"]["truth_over_cmaes"] < 1.0 for r in rows),
        "median_truth_over_cmaes_twice": statistics.median(r["twice"]["truth_over_cmaes"] for r in rows),
        "median_truth_over_cmaes_once": statistics.median(r["once"]["truth_over_cmaes"] for r in rows),
        "median_profile_change_in_wt": statistics.median(
            r["profile_change_in_wt"]["mean_rel"] for r in rows
            if r["profile_change_in_wt"]["mean_rel"] is not None),
        "provenance": {"git_commit": git_commit(), "torch": torch.__version__, "device": device,
                       "config": str(args.config), "data_root": str(args.data_root),
                       "cmaes_records": str(args.cmaes), "prior_scale": 0.0},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
