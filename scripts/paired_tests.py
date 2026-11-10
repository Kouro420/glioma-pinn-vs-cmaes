#!/usr/bin/env python
"""Both paired tests for every PINN-versus-CMA-ES comparison.

Every comparison reports the exact sign test and the Wilcoxon signed-rank test side by side.

Per comparison and metric: n, wins of each arm, ties, the sign-test p and the
Wilcoxon p (scipy.stats.wilcoxon, two-sided, zero differences dropped, the same
convention as the sign test). "cmaes_better" counts cases where CMA-ES has the
higher Dice or the smaller absolute parameter error.

    python scripts/paired_tests.py                # every comparison whose folders exist
    python scripts/paired_tests.py --out outputs/tables/paired_tests.json

Caveat: the 16 synthetic cases are 8 per brain on 2 geometries, so the cases are
not independent and every p is optimistic.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from math import comb
from statistics import median
from pathlib import Path

import scipy
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs"
BRAIN = re.compile(r"^(?:SYN|ENV)-(RHUH-\d{4})-")  # synthetic case -> its source brain

# label: (PINN folder, PINN method tag, CMA-ES folder). Folders under outputs/.
# Missing folders are skipped and listed.
COMPARISONS = {
    "C1 synthetic, default schedule, prior off": ("runs_syn_np", "pinn", "runs_syn_np"),
    "C1 synthetic, default schedule, prior on": ("runs_syn", "pinn", "runs_syn"),
    "patients, default schedule, prior on": ("runs_patient", "pinn", "runs_patient"),
    "E2 synthetic, paper schedule, prior off": ("runs_syn_np_paper", "pinn", "runs_syn_np"),
    "E3 patients, paper schedule, prior off": ("runs_patient_np_paper", "pinn", "runs_patient_np"),
    "E4 patients, release defaults vs CMA-ES prior off": (
        "runs_patient_release_defaults", "pinn", "runs_patient_np"),
    "S2 synthetic, release defaults vs CMA-ES prior off": (
        "runs_syn_release_defaults", "pinn", "runs_syn_np"),
    "EXP-REACH synthetic, lr 1e-3, prior off": ("runs_reach", "pinn", "runs_syn_np"),
    "EXP-REACH synthetic, lr 3e-4, prior off": ("runs_reach_3e-4", "pinn", "runs_syn_np"),
    "EXP-REACH synthetic, lr 1e-3, prior on": ("runs_reach_prior", "pinn", "runs_syn"),
    "EXP-ENV synthetic, paper schedule, prior off": ("runs_env", "pinn", "runs_env"),
    "B3 synthetic, paper's printed settings, prior off": ("runs_b3", "pinn", "runs_syn_np"),
}


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p at 1/2, ties already dropped."""
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def load(folder: Path, method: str) -> dict:
    out = {}
    for f in glob.glob(str(folder / f"*__{method}.json")):
        out[os.path.basename(f).split("__")[0]] = json.loads(Path(f).read_text(encoding="utf-8"))
    return out


def test_pair(margins: list[float]) -> dict:
    """margins > 0: CMA-ES better. Both tests on the same margins."""
    better = sum(m > 0 for m in margins)
    worse = sum(m < 0 for m in margins)
    nonzero = [m for m in margins if m != 0]
    p_w = float(wilcoxon(nonzero, zero_method="wilcox").pvalue) if len(nonzero) >= 1 else 1.0
    return {"n": len(margins), "cmaes_better": better, "pinn_better": worse,
            "ties": len(margins) - better - worse, "p_sign": sign_test(better, worse),
            "p_wilcoxon": p_w, "median_margin": float(median(margins))}


def compare(pinn_dir: Path, tag: str, cmaes_dir: Path) -> dict:
    P, C = load(pinn_dir, tag), load(cmaes_dir, "cmaes")
    cases = sorted(set(P) & set(C))
    result = {"pinn_dir": str(pinn_dir.relative_to(REPO)).replace("\\", "/"),
              "cmaes_dir": str(cmaes_dir.relative_to(REPO)).replace("\\", "/"),
              "cases": len(cases), "missing_pinn": sorted(set(C) - set(P)),
              "missing_cmaes": sorted(set(P) - set(C)), "metrics": {}}
    for metric in ("dice_wt", "dice_tc"):
        margins = [C[k]["metrics"][metric] - P[k]["metrics"][metric] for k in cases]
        result["metrics"][metric] = test_pair(margins)
    for metric in ("err_mu_d", "err_mu_rho"):  # absolute error against the truth
        if cases and all(metric in P[k]["metrics"] and metric in C[k]["metrics"] for k in cases):
            margins = [P[k]["metrics"][metric] - C[k]["metrics"][metric] for k in cases]
            result["metrics"][metric] = test_pair(margins)
    # Synthetic cases split by source brain (one brain carries the mu_rho result).
    # Counts only; the per-brain n is too small for a test.
    brains = {}
    for k in cases:
        match = BRAIN.match(k)
        if match:
            brains.setdefault(match.group(1), []).append(k)
    if brains:
        result["per_brain"] = {}
        for brain, keys in sorted(brains.items()):
            counts = {}
            for metric in result["metrics"]:
                if metric.startswith("dice"):
                    margins = [C[k]["metrics"][metric] - P[k]["metrics"][metric] for k in keys]
                else:
                    margins = [P[k]["metrics"][metric] - C[k]["metrics"][metric] for k in keys]
                counts[metric] = {"cmaes_better": sum(m > 0 for m in margins),
                                  "pinn_better": sum(m < 0 for m in margins)}
            result["per_brain"][brain] = {"cases": len(keys), "metrics": counts}
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RUNS / "tables" / "paired_tests.json")
    args = ap.parse_args()
    table, skipped = {}, []
    for label, (pinn, tag, cmaes) in COMPARISONS.items():
        if not (RUNS / pinn).is_dir() or not (RUNS / cmaes).is_dir():
            skipped.append(label)
            continue
        table[label] = compare(RUNS / pinn, tag, RUNS / cmaes)
    record = {"what": "exact sign test and Wilcoxon signed-rank, two-sided, zero differences "
                      "dropped, no multiplicity correction; margins > 0 favor CMA-ES",
              "caveat": "16 synthetic cases = 8 per brain on 2 geometries: not independent; "
                        "every p is optimistic",
              "scipy": scipy.__version__, "comparisons": table, "skipped_no_records": skipped}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for label, entry in table.items():
        print(f"{label}  (n={entry['cases']})")
        for metric, t in entry["metrics"].items():
            print(f"  {metric:<11} CMA-ES better {t['cmaes_better']:>2}, PINN {t['pinn_better']:>2}, "
                  f"ties {t['ties']}  sign p={t['p_sign']:.3g}  Wilcoxon p={t['p_wilcoxon']:.3g}")
        for brain, b in entry.get("per_brain", {}).items():
            print(f"  {brain} (n={b['cases']}): " + ", ".join(
                f"{metric} {c['cmaes_better']}-{c['pinn_better']}" for metric, c in b["metrics"].items()))
    if skipped:
        print("skipped (records not found):", "; ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
