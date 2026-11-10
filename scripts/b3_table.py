"""The PINN on the training settings printed in Zhang et al. (2025), summarized.

Reads the PINN records of this run (``outputs/runs_b3``, ``configs/paper_settings_b3.yaml``), the CMA-ES
records of the matched comparison (``outputs/runs_syn_np``) and the objective at both estimates
(``outputs/tables/b3_objective.json``, from ``scripts/objective_at_estimates.py``). Applies both parts of
the outcome rule set before the run, and reports per case
the direction of mu_D's movement in fine-tuning, its mean rate per step and whether it was still moving at
the end (|change over the last 500 steps| > STILL).

    python -m scripts.b3_table
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
B3 = REPO / "outputs" / "runs_b3"
CMAES = REPO / "outputs" / "runs_syn_np"
OBJECTIVE = REPO / "outputs" / "tables" / "b3_objective.json"
OUT = REPO / "outputs" / "tables" / "b3_paper_settings.json"
INV_START = 20000  # first fine-tuning step (pre-training has 20,000 steps)
STILL = 0.005      # mu_D change over the last 500 steps counted as "still moving"


def load(folder: Path, method: str) -> dict[str, dict]:
    out = {}
    for p in sorted(folder.glob(f"*__{method}.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["case_id"]] = r
    return out


def at_step(history: list[dict], step: int) -> float:
    return [h for h in history if h["step"] <= step][-1]["rD"]


def main() -> None:
    pinn, cmaes = load(B3, "pinn"), load(CMAES, "cmaes")
    if len(pinn) != 16:
        raise SystemExit(f"{B3}: {len(pinn)} cases, expected 16")
    rows = []
    for case, r in sorted(pinn.items()):
        h, last = r["history"], r["history"][-1]["step"]
        start, end = at_step(h, INV_START), r["estimate"]["mu_d"]
        tail = at_step(h, last - 500) - h[-1]["rD"]
        rows.append({"case": case, "mu_d_true": r["metrics"]["mu_d_true"], "mu_rho_true": r["metrics"]["mu_rho_true"],
                     "mu_d": end, "mu_rho": r["estimate"]["mu_rho"],
                     "err_mu_d": r["metrics"]["err_mu_d"], "err_mu_rho": r["metrics"]["err_mu_rho"],
                     "err_mu_d_cmaes": cmaes[case]["metrics"]["err_mu_d"],
                     "err_mu_rho_cmaes": cmaes[case]["metrics"]["err_mu_rho"],
                     "dice_wt": r["metrics"]["dice_wt"], "dice_tc": r["metrics"]["dice_tc"],
                     "dice_wt_cmaes": cmaes[case]["metrics"]["dice_wt"], "dice_tc_cmaes": cmaes[case]["metrics"]["dice_tc"],
                     "mu_d_finetune_start": start, "direction": "down" if end < start else "up",
                     "rate_per_step": (end - start) / (last - INV_START), "change_last_500": -tail,
                     "still_moving": abs(tail) > STILL, "wall_clock_s": r["cost"]["wall_clock_s"],
                     "stop_reason": r["cost"]["stop_reason"]})
    truth, est = [x["mu_d_true"] for x in rows], [x["mu_d"] for x in rows]
    r_mu_d = float(pearsonr(est, truth)[0])
    med_err, med_err_cmaes = st.median(x["err_mu_d"] for x in rows), st.median(x["err_mu_d_cmaes"] for x in rows)
    if r_mu_d >= 0.5 and med_err <= med_err_cmaes:
        part2 = "the PINN calibrates diffusion at least as well as CMA-ES"
    elif r_mu_d < 0.5:
        part2 = "the PINN's mu_D estimates do not follow the truth"
    else:
        part2 = "the PINN's mu_D estimates follow the truth (r >= 0.5), but less accurately than CMA-ES's"
    result = {"what": "The PINN on the training settings printed in Zhang et al. (2025), 16 synthetic cases, prior off",
              "r_mu_d_truth": r_mu_d, "median_abs_err_mu_d": med_err, "median_abs_err_mu_d_cmaes": med_err_cmaes,
              "median_abs_err_mu_rho": st.median(x["err_mu_rho"] for x in rows),
              "median_abs_err_mu_rho_cmaes": st.median(x["err_mu_rho_cmaes"] for x in rows),
              "mu_d_range": [min(est), max(est)], "verdict_parameters": part2,
              "directions": {d: sum(x["direction"] == d for x in rows) for d in ("down", "up")},
              "still_moving": sum(x["still_moving"] for x in rows), "still_threshold": STILL,
              "median_dice_wt": st.median(x["dice_wt"] for x in rows),
              "median_dice_tc": st.median(x["dice_tc"] for x in rows),
              "cmaes_dice_better_wt": sum(x["dice_wt_cmaes"] > x["dice_wt"] for x in rows),
              "cmaes_err_mu_d_better": sum(x["err_mu_d_cmaes"] < x["err_mu_d"] for x in rows),
              "cmaes_err_mu_rho_better": sum(x["err_mu_rho_cmaes"] < x["err_mu_rho"] for x in rows),
              "cases": rows}
    if OBJECTIVE.exists():
        obj = json.loads(OBJECTIVE.read_text(encoding="utf-8"))["summary"]
        result["objective"] = {k: obj[k] for k in ("n", "rule", "pinn_within", "rule_met", "ratio_min",
                                                   "ratio_median", "ratio_max", "truth_below_cmaes")}
        result["verdict_objective"] = ("the PINN reaches the CMA-ES objective value"
                                       if obj["rule_met"] else
                                       "the PINN stops short of the CMA-ES objective value")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for x in rows:
        print(f"{x['case']}: mu_D {x['mu_d']:.3f} (truth {x['mu_d_true']:.3f}) start {x['mu_d_finetune_start']:.3f} "
              f"{x['direction']} {x['rate_per_step']:+.2e}/step, last 500 {x['change_last_500']:+.4f}; "
              f"mu_rho {x['mu_rho']:.3f} (truth {x['mu_rho_true']:.3f}); Dice WT {x['dice_wt']:.3f} vs {x['dice_wt_cmaes']:.3f}")
    print({k: v for k, v in result.items() if k not in ("cases", "what")})
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
