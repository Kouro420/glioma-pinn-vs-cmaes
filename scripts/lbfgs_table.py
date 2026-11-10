"""The L-BFGS test of Additional file 1 (table tab:supp-lbfgs).

The PINN on the training schedule of Zhang et al., prior off, with the L-BFGS stage of their original
TensorFlow code added after Adam in pre-training (outputs/runs_lbfgs_full), against the same 4 profiled
cases without it (outputs/runs_syn_np_paper). Per case: the true mu_D, both estimates and their
differences in mu_D and mu_rho, and the L-BFGS iterations, evaluations and loss reduction in pre-training.
The rule fixed in advance: the final mu_D with L-BFGS lies within 0.05 of the value without it in all
four cases.

    python -m scripts.lbfgs_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LBFGS = REPO / "outputs" / "runs_lbfgs_full"
PAPER = REPO / "outputs" / "runs_syn_np_paper"
PROFILED = REPO / "experiments" / "cases_profile_zoom.txt"
TOLERANCE = 0.05  # the rule's bound on |difference in mu_D|


def load(folder: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(folder.glob("*__pinn.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record["case_id"]] = record
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "tables",
                        help="folder for lbfgs_full.json (default: outputs/tables)")
    arguments = parser.parse_args()

    profiled = PROFILED.read_text(encoding="utf-8").split()
    with_lbfgs, paper = load(LBFGS), load(PAPER)
    missing = [c for c in profiled if c not in with_lbfgs or c not in paper]
    if missing:
        raise ValueError(f"no record for {missing} in {LBFGS.name} or {PAPER.name}")

    rows = []
    for case in profiled:
        a, b = with_lbfgs[case], paper[case]
        if a["metrics"]["mu_d_true"] != b["metrics"]["mu_d_true"]:
            raise ValueError(f"{case}: the two records disagree on the truth")
        if a["provenance"]["prior_scale"] != b["provenance"]["prior_scale"]:
            raise ValueError(f"{case}: the two records differ in the prior")
        lbfgs = (a.get("early_stopping") or {}).get("stages", {}).get("init", {}).get("lbfgs") or {}
        row = {
            "case": case,
            "source_brain": case.split("-d")[0].replace("SYN-", ""),
            "mu_d_true": a["metrics"]["mu_d_true"],
            "mu_d_lbfgs": a["estimate"]["mu_d"],
            "mu_d_paper_schedule": b["estimate"]["mu_d"],
            "diff_mu_d": a["estimate"]["mu_d"] - b["estimate"]["mu_d"],
            "mu_rho_lbfgs": a["estimate"]["mu_rho"],
            "mu_rho_paper_schedule": b["estimate"]["mu_rho"],
            "diff_mu_rho": a["estimate"]["mu_rho"] - b["estimate"]["mu_rho"],
            "prior": a["provenance"]["prior_scale"],
        }
        if lbfgs:
            row.update({
                "lbfgs_iterations": lbfgs["iterations"],
                "lbfgs_function_evals": lbfgs["function_evals"],
                "pretraining_loss_start": lbfgs["loss_start"],
                "pretraining_loss_end": lbfgs["loss_end"],
                "loss_reduction": 1.0 - lbfgs["loss_end"] / lbfgs["loss_start"],
                "lbfgs_stop_reason": lbfgs["stop_reason"],
            })
        rows.append(row)

    largest = max(abs(r["diff_mu_d"]) for r in rows)
    doc = {
        "description": "PINN on the training schedule of Zhang et al. with L-BFGS after Adam in "
                       "pre-training, against the same schedule without it (paper_schedule), "
                       "4 profiled cases, prior off; differences are with minus without L-BFGS",
        "cases": rows,
        "max_abs_diff_mu_d": largest,
        "tolerance": TOLERANCE,
        "all_within_tolerance": largest <= TOLERANCE,
    }
    if all("lbfgs_iterations" in r for r in rows):
        for key in ("lbfgs_iterations", "lbfgs_function_evals", "loss_reduction"):
            doc[f"{key}_range"] = [min(r[key] for r in rows), max(r[key] for r in rows)]

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    out = arguments.out_dir / "lbfgs_full.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for r in rows:
        print(f"{r['case']}: true mu_D {r['mu_d_true']:.3f}, L-BFGS {r['mu_d_lbfgs']:.3f}, "
              f"paper schedule {r['mu_d_paper_schedule']:.3f}, difference {r['diff_mu_d']:+.3f} "
              f"(mu_rho {r['diff_mu_rho']:+.3f})")
    print(f"largest |difference in mu_D| {largest:.3f}; all within {TOLERANCE}: {doc['all_within_tolerance']}")
    if "lbfgs_iterations_range" in doc:
        it, ev, red = doc["lbfgs_iterations_range"], doc["lbfgs_function_evals_range"], doc["loss_reduction_range"]
        print(f"L-BFGS iterations {it[0]}-{it[1]}, evaluations {ev[0]}-{ev[1]}, "
              f"pre-training loss lowered by {100 * red[0]:.0f}-{100 * red[1]:.0f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
