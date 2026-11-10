"""Three CMA-ES checks of the tumor seed position, summarized, with the verdict of each outcome rule.

Records (CMA-ES random seed 1, prior off, 600 evaluations):
* (a) ``outputs/runs_d104a``: the 16 noise-free cases on the fit grid (``synthetic_e1_both``), the tumor
  seed position at the generator's, thresholds free;
* (b) ``outputs/runs_d104b``: the same 16 cases, the all-free baseline again, with CMA-ES random seed 1;
* (c) ``outputs/runs_d104c``: the 12 design-range cases (truths inside [0.75, 1.25]), the tumor seed
  position at the generator's.
For context: the all-free baseline on the same 16 cases (``outputs/runs_e1_both``, CMA-ES random seed 0,
for which pycma draws a seed from the clock), the refit with thresholds fixed (``runs_x1_thfix``), the
refit with thresholds and tumor seed position fixed (``runs_x1_thfix_seed``), and the design-range cases
with the tumor seed position at the mask centroid (``runs_env``).

M is the median absolute relative mu_D error. The outcome rules, set before the runs:
* (a) against (b): M(a) <= M(b)/2 -> "most"; M(a) >= M(b) - 0.05 -> "does not reduce"; otherwise "in part";
* (c) on the 5 RHUH-0001 cases: M <= 0.15 -> "tumor seed position"; M >= 0.50 -> "masks"; otherwise "in part".

    python -m scripts.d104_table
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs"
SETS = {
    "d104a_seed_fixed_thresholds_free": "runs_d104a",
    "d104b_baseline_seed1": "runs_d104b",
    "e1_both_baseline_seed0": "runs_e1_both",
    "x1_i_thresholds_fixed": "runs_x1_thfix",
    "x1_ii_thresholds_and_seed_fixed": "runs_x1_thfix_seed",
    "d104c_env_generator_seed": "runs_d104c",
    "env_centroid_seed": "runs_env",
}
OUT = RUNS / "tables" / "d104_seed_checks.json"


def load(folder: str) -> dict[str, dict]:
    out = {}
    for p in sorted((RUNS / folder).glob("*__cmaes*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["case_id"]] = r
    return out


def brain(case: str) -> str:
    return "-".join(case.split("-")[1:3])


def summarise(records: dict[str, dict]) -> dict:
    rows = []
    for case, r in sorted(records.items()):
        m = r["metrics"]
        rows.append({"case": case, "brain": brain(case), "mu_d_true": m["mu_d_true"], "mu_d": r["estimate"]["mu_d"],
                     "rel_err_mu_d": (r["estimate"]["mu_d"] - m["mu_d_true"]) / m["mu_d_true"],
                     "rel_err_mu_rho": (r["estimate"]["mu_rho"] - m["mu_rho_true"]) / m["mu_rho_true"],
                     "th_wt": r["estimate"]["th_wt"], "th_tc": r["estimate"]["th_tc"],
                     "dice_wt": m["dice_wt"], "dice_tc": m["dice_tc"],
                     "n_evaluations": r["cost"]["n_evaluations"], "stop_reason": r["cost"]["stop_reason"],
                     "best_score": min(h["score"] for h in r["history"]),
                     "x0_vox": r.get("initial_condition", {}).get("x0_vox") if isinstance(r.get("initial_condition"), dict) else None})

    def block(sub):
        return {"n": len(sub),
                "M_median_abs_rel_err_mu_d": st.median(abs(x["rel_err_mu_d"]) for x in sub),
                "E_median_signed_rel_err_mu_d": st.median(x["rel_err_mu_d"] for x in sub),
                "below_truth": sum(x["rel_err_mu_d"] < 0 for x in sub),
                "range_signed_rel_err_mu_d": [min(x["rel_err_mu_d"] for x in sub), max(x["rel_err_mu_d"] for x in sub)],
                "median_abs_rel_err_mu_rho": st.median(abs(x["rel_err_mu_rho"]) for x in sub),
                "median_dice_wt": st.median(x["dice_wt"] for x in sub),
                "median_dice_tc": st.median(x["dice_tc"] for x in sub),
                "evaluations_range": [min(x["n_evaluations"] for x in sub), max(x["n_evaluations"] for x in sub)],
                "stopped_on_budget": sum(x["n_evaluations"] >= 600 for x in sub)}

    out = block(rows)
    out["per_brain"] = {b: block([x for x in rows if x["brain"] == b]) for b in sorted({x["brain"] for x in rows})}
    out["cases"] = rows
    return out


def rule_a(m_a: float, m_b: float) -> str:
    if m_a <= m_b / 2:
        return "with the thresholds free, the tumor seed position alone accounts for most of the bias"
    if m_a >= m_b - 0.05:
        return "with the thresholds free, the tumor seed position alone does not reduce the bias"
    return "with the thresholds free, the tumor seed position alone accounts for part of the bias"


def rule_c(m: float) -> str:
    if m <= 0.15:
        return "the low mu_D on RHUH-0001 came from the tumor seed position, not from the masks"
    if m >= 0.50:
        return "the masks favor a low mu_D on RHUH-0001 even with the generator's tumor seed position"
    return "the low mu_D on RHUH-0001 came in part from the tumor seed position"


def main() -> None:
    result = {"what": "CMA-ES checks of the tumor seed position and the random seed (prior off, 600 evaluations)", "sets": {}}
    for name, folder in SETS.items():
        records = load(folder)
        expected = 12 if "env" in name else 16
        if len(records) != expected:
            raise SystemExit(f"{folder}: {len(records)} cases, expected {expected}")
        result["sets"][name] = summarise(records)
        result["sets"][name]["folder"] = f"outputs/{folder}"
    s = result["sets"]
    m_a = s["d104a_seed_fixed_thresholds_free"]["M_median_abs_rel_err_mu_d"]
    m_b = s["d104b_baseline_seed1"]["M_median_abs_rel_err_mu_d"]
    m_c = s["d104c_env_generator_seed"]["per_brain"]["RHUH-0001"]["M_median_abs_rel_err_mu_d"]
    result["rule_a"] = {"M_a": m_a, "M_b": m_b, "half_M_b": m_b / 2, "M_b_minus_0.05": m_b - 0.05, "verdict": rule_a(m_a, m_b)}
    result["rule_c"] = {"M_c_RHUH_0001": m_c, "verdict": rule_c(m_c)}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, x in s.items():
        pb = " | ".join(f"{b}: M {v['M_median_abs_rel_err_mu_d']:.3f} E {v['E_median_signed_rel_err_mu_d']:+.3f} "
                        f"below {v['below_truth']}/{v['n']}" for b, v in x["per_brain"].items())
        print(f"{name:34s} M {x['M_median_abs_rel_err_mu_d']:.3f} E {x['E_median_signed_rel_err_mu_d']:+.3f} "
              f"below {x['below_truth']}/{x['n']} Dice {x['median_dice_wt']:.3f}/{x['median_dice_tc']:.3f} "
              f"evals {x['evaluations_range']} budget {x['stopped_on_budget']} || {pb}")
    print("rule (a):", result["rule_a"])
    print("rule (c):", result["rule_c"])
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
