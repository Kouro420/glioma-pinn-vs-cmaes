#!/usr/bin/env python
"""Reproduce one synthetic case of our paper end to end, and check it against our run records.

The case is SYN-RHUH-0002-d1.2-r0.6 (Figure 1 of our paper, row 1). Four steps, each skipped if its output
already exists:

1. build the synthetic case from the public brain RHUH-0002 (truth on a 0.5 mm grid, noisy masks);
2. fit it with CMA-ES (configs/base.yaml, prior off, random seed 1);
3. fit it with the reference PINN on the training schedule of Zhang et al. (configs/paper_schedule.yaml, prior off);
4. compare truth, both estimates and both Dice values with repro/one_case/expected.json.

    python repro/one_case/run_one_case.py --data-root <PREDICT-GBM folder>           # about 6 h on one GPU
    python repro/one_case/run_one_case.py --data-root <PREDICT-GBM folder> --smoke   # minutes; checks the pipeline only

Outputs go to outputs/repro_one_case/. Needs scripts/fetch_reference_pinn.sh first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASE = "SYN-RHUH-0002-d1.2-r0.6"
SOURCE = "RHUH-0002"
# The generator gives case k of its list the noise random seed synthetic.seed + k (configs/base.yaml:
# 20260916). In our list RHUH-0001's eight cases come first, so RHUH-0002 alone starts at +8; this case is the
# seventh of RHUH-0002 (--n-cases 7). Its noise is then exactly that of our run.
SEED = 20260916 + 8
N_CASES = 7


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def check(name: str, got: float, want: float, tol: float) -> bool:
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<12} {got:.4f}  (our run {want:.4f}, tolerance {tol:g})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="folder holding the PREDICT-GBM cases")
    ap.add_argument("--smoke", action="store_true", help="short budgets: checks that every step runs, not the numbers")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not (REPO / "external" / "pinngbm-torch" / "Options.py").is_file():
        print("The reference PINN is missing: run scripts/fetch_reference_pinn.sh first.")
        return 1

    out = REPO / "outputs" / "repro_one_case"
    # The smoke run builds only the first RHUH-0002 case (one fine solve instead of seven) and fits that.
    case, n_cases = ("SYN-RHUH-0002-d0.2-r0.6", 1) if args.smoke else (CASE, N_CASES)
    synthetic = out / ("synthetic_smoke" if args.smoke else "synthetic")
    runs = out / ("runs_smoke" if args.smoke else "runs")
    py = sys.executable
    device = ["--device", args.device] if args.device else []

    # 1. the synthetic case
    if not (synthetic / case / "truth.json").is_file():
        run([py, "-m", "src.data.make_synthetic", "--case", SOURCE, "--data-root", str(args.data_root),
             "--out", str(synthetic), "--seed", str(SEED), "--n-cases", str(n_cases), *device])
    # 2. CMA-ES
    cmaes_record = runs / f"{case}__cmaes_seed1.json"
    if not cmaes_record.is_file():
        run([py, "-m", "experiments.run_cmaes", "--case", case, "--data-root", str(synthetic), "--synthetic",
             "--config", "configs/base.yaml", "--prior-scale", "0.0", "--seed", "1", "--out", str(runs), *device,
             *(["--time-budget-s", "120"] if args.smoke else [])])
    # 3. the PINN
    pinn_record = runs / f"{case}__pinn.json"
    if not pinn_record.is_file():
        # The smoke config's small batches fit an 8 GB GPU; the training schedule of Zhang et al. needs about 22 GB.
        config = "configs/smoke_one_case_small_gpu.yaml" if args.smoke else "configs/paper_schedule.yaml"
        run([py, "-m", "experiments.run_pinn", "--case", case, "--data-root", str(synthetic), "--synthetic",
             "--config", config, "--prior-scale", "0.0", "--out", str(runs), *device])

    # 4. compare
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    truth = json.loads((synthetic / case / "truth.json").read_text(encoding="utf-8"))
    cmaes = json.loads(cmaes_record.read_text(encoding="utf-8"))
    pinn = json.loads(pinn_record.read_text(encoding="utf-8"))
    if args.smoke:
        print("\nSmoke run finished: every step ran and wrote its record (numbers are not checked).")
        print(f"  CMA-ES mu_D {cmaes['estimate']['mu_d']:.3f}, PINN mu_D {pinn['estimate']['mu_d']:.3f}")
        return 0

    results = []
    syn, tol = expected["synthetic"], expected["synthetic"]["tolerance"]
    print(f"\n{CASE}: synthetic case")
    results.append(check("mu_d_true", truth["mu_d_true"], syn["mu_d_true"], tol["mu_d_true"]))
    results.append(check("mu_rho_true", truth["mu_rho_true"], syn["mu_rho_true"], tol["mu_rho_true"]))
    for key in ("voxels_wt", "voxels_tc"):
        results.append(check(key, truth[key], syn[key], tol["voxels_relative"] * syn[key]))
    for arm, record in (("cmaes", cmaes), ("pinn", pinn)):
        want, t = expected[arm], expected[arm]["tolerance"]
        print(f"{arm}")
        for key in ("mu_d", "mu_rho"):
            results.append(check(key, record["estimate"][key], want[key], t[key]))
        for key in ("dice_wt", "dice_tc"):
            results.append(check(key, record["metrics"][key], want[key], t[key]))
    print("the findings of our paper on this case")
    findings = [pinn["estimate"]["mu_d"] < cmaes["estimate"]["mu_d"] < truth["mu_d_true"],
                cmaes["metrics"]["dice_wt"] > pinn["metrics"]["dice_wt"]]
    for text, ok in zip(expected["findings"], findings):
        print(f"  {'ok  ' if ok else 'FAIL'} {text}")
    results += findings
    print(f"\n{sum(results)} of {len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
