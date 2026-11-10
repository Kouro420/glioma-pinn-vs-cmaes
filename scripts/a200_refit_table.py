"""The refit with thresholds and tumor seed position fixed at the generator's values, at a sharper threshold
(a = 200), summarized.

Reads the CMA-ES records of the a = 200 refit (``outputs/runs_a200_on``, second tissue renormalization
inside the tumor on, the setting the outcome rule is judged on; ``outputs/runs_a200_off``, off) and, for comparison,
the same refit at a = 20 (``outputs/runs_x1_thfix_seed``). Reports E, the median signed relative
mu_D error, and the verdict of the outcome rule; the count below the truth; per-brain medians; stop
reasons and evaluation counts; the objective at the refit point against the objective at the truth,
both at a = 200. The truth's objective comes from ``outputs/tables/b1_b2_rescore.json`` (same cases,
thresholds and tumor seed position at the generator's values, prior off, a = 200, renormalization on and off), the
refit point's from the record's best score.

    python -m scripts.a200_refit_table
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETS = {"on": REPO / "outputs" / "runs_a200_on", "off": REPO / "outputs" / "runs_a200_off"}
X1 = REPO / "outputs" / "runs_x1_thfix_seed"
RESCORE = REPO / "outputs" / "tables" / "b1_b2_rescore.json"
OUT = REPO / "outputs" / "tables" / "a200_refit.json"
BUDGET = 600


def verdict(e: float) -> str:
    """The outcome rule, set before the run."""
    if abs(e) <= 0.05:
        return "the smoothed threshold accounts for the bias that remains with the thresholds and the tumor seed position fixed"
    if e >= -0.07:
        return "the smoothed threshold accounts for at least half of the bias that remains with the thresholds and the tumor seed position fixed"
    return "the smoothed threshold does not account for most of the bias that remains with the thresholds and the tumor seed position fixed"


def load(folder: Path) -> dict[str, dict]:
    return {json.loads(p.read_text(encoding="utf-8"))["case_id"]: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("*.json"))}


def rel(record: dict, key: str) -> float:
    m = record["metrics"]
    true = m["mu_d_true"] if key == "mu_d" else m["mu_rho_true"]
    return (record["estimate"][key] - true) / true


def summarise(records: dict[str, dict], truth_obj: dict[str, float] | None) -> dict:
    rows = []
    for case, r in records.items():
        best = min(h["score"] for h in r["history"])
        row = {
            "case": case,
            "brain": case.split("-")[1] + "-" + case.split("-")[2],
            "mu_d_true": r["metrics"]["mu_d_true"],
            "mu_d": r["estimate"]["mu_d"],
            "rel_err_mu_d": rel(r, "mu_d"),
            "rel_err_mu_rho": rel(r, "mu_rho"),
            "n_evaluations": r["cost"]["n_evaluations"],
            "stop_reason": r["cost"]["stop_reason"],
            "objective_refit": best,
            "dice_wt": r["metrics"]["dice_wt"],
            "dice_tc": r["metrics"]["dice_tc"],
        }
        if truth_obj is not None:
            row["objective_truth"] = truth_obj[case]
            row["ratio_truth_over_refit"] = truth_obj[case] / best
        rows.append(row)
    e = st.median(x["rel_err_mu_d"] for x in rows)
    out = {
        "n": len(rows),
        "E_median_signed_rel_err_mu_d": e,
        "median_abs_rel_err_mu_d": st.median(abs(x["rel_err_mu_d"]) for x in rows),
        "range_signed_rel_err_mu_d": [min(x["rel_err_mu_d"] for x in rows), max(x["rel_err_mu_d"] for x in rows)],
        "below_truth": sum(x["rel_err_mu_d"] < 0 for x in rows),
        "median_abs_rel_err_mu_rho": st.median(abs(x["rel_err_mu_rho"]) for x in rows),
        "per_brain_E": {b: st.median(x["rel_err_mu_d"] for x in rows if x["brain"] == b)
                        for b in sorted({x["brain"] for x in rows})},
        "evaluations_range": [min(x["n_evaluations"] for x in rows), max(x["n_evaluations"] for x in rows)],
        "stopped_on_budget": sum(x["n_evaluations"] >= BUDGET for x in rows),
        "stop_reasons": sorted({x["stop_reason"] for x in rows}),
        "median_dice_wt": st.median(x["dice_wt"] for x in rows),
        "median_dice_tc": st.median(x["dice_tc"] for x in rows),
        "cases": rows,
    }
    if truth_obj is not None:
        ratios = [x["ratio_truth_over_refit"] for x in rows]
        out["truth_at_least_as_good"] = sum(q <= 1.0 for q in ratios)
        out["median_ratio_truth_over_refit"] = st.median(ratios)
        out["range_ratio_truth_over_refit"] = [min(ratios), max(ratios)]
    return out


def main() -> None:
    rescore = json.loads(RESCORE.read_text(encoding="utf-8"))["cases"]
    result = {"what": "Refit at a = 200 (thresholds and the tumor seed position at the generator's values, prior off)",
              "rule_primary": "on"}
    for setting, folder in SETS.items():
        truth_obj = {c["case"]: c["settings"][setting]["a200"]["truth"] for c in rescore}
        s = summarise(load(folder), truth_obj)
        s["verdict"] = verdict(s["E_median_signed_rel_err_mu_d"])
        s["stall_note"] = s["stopped_on_budget"] > 4
        result[setting] = s
    result["x1_ii_a20"] = {k: v for k, v in summarise(load(X1), None).items() if k != "cases"}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key in ("on", "off", "x1_ii_a20"):
        s = result[key]
        print(key, "E", round(s["E_median_signed_rel_err_mu_d"], 4), "abs", round(s["median_abs_rel_err_mu_d"], 4),
              "range", [round(v, 3) for v in s["range_signed_rel_err_mu_d"]], "below", s["below_truth"],
              "per brain", {b: round(v, 4) for b, v in s["per_brain_E"].items()},
              "mu_rho", round(s["median_abs_rel_err_mu_rho"], 4), "evals", s["evaluations_range"],
              "budget", s["stopped_on_budget"], s["stop_reasons"],
              "dice", round(s["median_dice_wt"], 4), round(s["median_dice_tc"], 4))
        if "verdict" in s:
            print("   ratio truth/refit median", round(s["median_ratio_truth_over_refit"], 4), "range",
                  [round(v, 4) for v in s["range_ratio_truth_over_refit"]], "truth >= refit in",
                  s["truth_at_least_as_good"], "| verdict:", s["verdict"], "| stall note:", s["stall_note"])
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
