"""How far the PINN release of Zhang et al., at its own defaults, moves from its first guess.

Reads the 20 patient records of outputs/runs_patient_release_defaults (the release
with every Options.py default, separable network, prior on) and, for comparison,
the dense-network PINN records with the prior on (outputs/runs_patient). Writes
outputs/tables/e4_displacement.json: the median |mu - 1| per parameter, the outcome
rule on it, and the range of each estimate over the patients.
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def summarise(folder: str, pattern: str) -> dict:
    mu_d, mu_rho, cases = [], [], []
    for path in sorted((REPO / "outputs" / folder).glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        estimate = record.get("estimate") or {}
        if "mu_d" not in estimate:
            continue
        cases.append(record.get("case_id") or path.stem)
        mu_d.append(float(estimate["mu_d"]))
        mu_rho.append(float(estimate["mu_rho"]))
    return {
        "folder": f"outputs/{folder}",
        "n": len(cases),
        "median_abs_mu_d_minus_1": round(statistics.median(abs(v - 1) for v in mu_d), 4),
        "median_abs_mu_rho_minus_1": round(statistics.median(abs(v - 1) for v in mu_rho), 4),
        "mu_d_range": [round(min(mu_d), 3), round(max(mu_d), 3)],
        "mu_rho_range": [round(min(mu_rho), 3), round(max(mu_rho), 3)],
    }


def main() -> None:
    release = summarise("runs_patient_release_defaults", "*.json")
    dense = summarise("runs_patient", "*pinn*.json")
    rule_first_guess = (release["median_abs_mu_d_minus_1"] < 0.05
                        and release["median_abs_mu_rho_minus_1"] < 0.05)
    out = {
        "rule": "returns its first guess if median |mu_D - 1| < 0.05 and median |mu_rho - 1| < 0.05",
        "rule_first_guess": rule_first_guess,
        "release_defaults_separable_prior_on": release,
        "dense_prior_on": dense,
    }
    target = REPO / "outputs" / "tables" / "e4_displacement.json"
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
