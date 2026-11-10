"""The refit with thresholds and tumor seed position fixed at the generator's values and the second
tissue renormalization off, summarized.

Reads the CMA-ES records of this refit (``outputs/runs_b1b``) and, for comparison, the same refit
with the renormalization on (``outputs/runs_x1_thfix_seed``). Reports E, the median signed relative
mu_D error, and the verdict of the outcome rule; the count below the truth; per-brain medians. Uses
``summarise`` of ``scripts/a200_refit_table.py`` (no truth objective here).

    python -m scripts.b1b_table
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.a200_refit_table import X1, load, summarise

REPO = Path(__file__).resolve().parents[1]
B1B = REPO / "outputs" / "runs_b1b"
OUT = REPO / "outputs" / "tables" / "b1b_refit.json"


def verdict(e: float) -> str:
    """The outcome rule, set before the run."""
    if abs(e) <= 0.05:
        return "the second renormalization accounts for the bias that remains with the thresholds and the tumor seed position fixed"
    if e >= -0.07:
        return "the second renormalization accounts for at least half of the bias that remains with the thresholds and the tumor seed position fixed"
    return "the second renormalization does not account for most of the bias that remains with the thresholds and the tumor seed position fixed"


def main() -> None:
    records = load(B1B)
    if len(records) != 16:
        raise SystemExit(f"{B1B}: {len(records)} cases, expected 16")
    for case, r in records.items():
        if r["config"]["data"].get("renormalise_tumour") is not False or r["config"]["cmaes"].get("fix_thresholds") != "truth":
            raise SystemExit(f"{case}: not the renormalization-off refit setting")
    s = summarise(records, None)
    s["verdict"] = verdict(s["E_median_signed_rel_err_mu_d"])
    result = {"what": "CMA-ES refit, thresholds and the tumor seed position at the generator's values, second "
                      "renormalization off, random seed 1, prior off", "b1b": s,
              "x1_ii_renorm_on": {k: v for k, v in summarise(load(X1), None).items() if k != "cases"}}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for key in ("b1b", "x1_ii_renorm_on"):
        x = result[key]
        print(key, "E", round(x["E_median_signed_rel_err_mu_d"], 4), "abs", round(x["median_abs_rel_err_mu_d"], 4),
              "range", [round(v, 3) for v in x["range_signed_rel_err_mu_d"]], "below", x["below_truth"],
              "per brain", {b: round(v, 4) for b, v in x["per_brain_E"].items()},
              "evals", x["evaluations_range"], "budget", x["stopped_on_budget"],
              "dice", round(x["median_dice_wt"], 4), round(x["median_dice_tc"], 4))
    print("verdict:", s["verdict"])
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
