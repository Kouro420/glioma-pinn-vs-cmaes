"""The no-data test of why the PINN's mu_D falls, summarized.

Reads the PINN records of the no-data run (``outputs/runs_nodata``: both mask weights 0 in fine-tuning,
pre-training unchanged, fine-tuning run to its 5,000-step budget) and the records of the same 4 cases
with the data, on the training schedule of Zhang et al. (2025) and prior off (``outputs/runs_syn_np_paper``).
Compares the fall of mu_D over fine-tuning, Delta = 1 - mu_D(end), and applies the outcome rule set before
the run. Also reports mu_D at the end of pre-training (step 20,000; mu_D does not move in pre-training)
and over the last 500 steps.

    python -m scripts.nodata_table
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NODATA = REPO / "outputs" / "runs_nodata"
E2 = REPO / "outputs" / "runs_syn_np_paper"
OUT = REPO / "outputs" / "tables" / "nodata_test.json"
NEAR = 0.05  # the rule's tolerance on Delta


def load(folder: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(folder.glob("*__pinn.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["case_id"]] = r
    return out


def at_step(history: list[dict], step: int) -> float:
    return [h for h in history if h["step"] <= step][-1]["rD"]


def verdict(rows: list[dict]) -> str:
    """The no-data outcome rule, set before the run."""
    if sum(abs(r["delta_nodata"] - r["delta_e2"]) <= NEAR for r in rows) >= 3:
        return "the fall does not come from the data; the residual term with the pre-trained network drives it"
    if sum(r["delta_nodata"] < NEAR for r in rows) >= 3:
        return "the data drives the fall"
    return "both the data and the residual term contribute to the fall"


def main() -> None:
    nodata, e2 = load(NODATA), load(E2)
    if len(nodata) != 4:
        raise SystemExit(f"{NODATA}: {len(nodata)} cases, expected 4")
    rows = []
    for case, r in nodata.items():
        if r["config"]["pinn"].get("seg_weight_scale") != 0.0:
            raise SystemExit(f"{case}: mask weights not 0")
        q = e2[case]
        hn, he = r["history"], q["history"]
        rows.append({
            "case": case,
            "mu_d_true": q["metrics"]["mu_d_true"],
            "mu_d_end_nodata": r["estimate"]["mu_d"], "mu_d_end_e2": q["estimate"]["mu_d"],
            "delta_nodata": 1.0 - r["estimate"]["mu_d"], "delta_e2": 1.0 - q["estimate"]["mu_d"],
            "mu_d_pretrain_end_nodata": at_step(hn, 20000), "mu_d_pretrain_end_e2": at_step(he, 20000),
            "last_step_nodata": hn[-1]["step"], "last_step_e2": he[-1]["step"],
            "fall_last_500_nodata": at_step(hn, hn[-1]["step"] - 500) - hn[-1]["rD"],
            "fall_last_500_e2": at_step(he, he[-1]["step"] - 500) - he[-1]["rD"],
            "mu_rho_end_nodata": r["estimate"]["mu_rho"], "mu_rho_end_e2": q["estimate"]["mu_rho"],
            "thresholds_end_nodata": [r["estimate"]["th_wt"], r["estimate"]["th_tc"]],
            "would_stop_nodata": r["early_stopping"]["stages"]["inv"].get("would_stop"),
        })
        rows[-1]["delta_diff"] = rows[-1]["delta_nodata"] - rows[-1]["delta_e2"]
    result = {"what": "No-data test: Delta = 1 - mu_D(end), mask weights 0 in fine-tuning, vs the same cases "
                      "with the mask terms on (paper schedule, prior off)",
              "near": NEAR,
              "within_near_of_e2": sum(abs(x["delta_diff"]) <= NEAR for x in rows),
              "delta_below_near": sum(x["delta_nodata"] < NEAR for x in rows),
              "verdict": verdict(rows), "cases": rows}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for x in rows:
        print(f"{x['case']}: Delta no-data {x['delta_nodata']:.3f}, with data {x['delta_e2']:.3f} (diff {x['delta_diff']:+.3f}); "
              f"mu_D after pre-training {x['mu_d_pretrain_end_nodata']:.3f} / {x['mu_d_pretrain_end_e2']:.3f}; "
              f"last-500 fall {x['fall_last_500_nodata']:.3f} / {x['fall_last_500_e2']:.3f}; "
              f"mu_rho {x['mu_rho_end_nodata']:.3f} / {x['mu_rho_end_e2']:.3f}; steps {x['last_step_nodata']}")
    print("within 0.05 of the run with data:", result["within_near_of_e2"], "of 4; Delta < 0.05:", result["delta_below_near"], "of 4")
    print("verdict:", result["verdict"])
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
