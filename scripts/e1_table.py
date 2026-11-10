"""The noise-and-grid test of Additional file 1 (table tab:supp-e1).

CMA-ES refits, prior off, of the 16 synthetic cases in four variants: as built (outputs/runs_syn_np),
without noise (outputs/runs_e1_nonoise), on the fit grid (outputs/runs_e1_samegrid) and both
(outputs/runs_e1_both). Each fit is scored against its own variant's truth, read from the run record.
Writes outputs/tables/e1_primary.json (the 4 profiled cases) and outputs/tables/e1_all16.json.

    python -m scripts.e1_table
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# variant -> (CMA-ES run records, synthetic cases with their truth.json)
VARIANTS = {
    "baseline": ("outputs/runs_syn_np", "outputs/synthetic_local"),
    "nonoise": ("outputs/runs_e1_nonoise", "outputs/synthetic_e1_nonoise"),
    "samegrid": ("outputs/runs_e1_samegrid", "outputs/synthetic_e1_samegrid"),
    "both": ("outputs/runs_e1_both", "outputs/synthetic_e1_both"),
}
PROFILED = REPO / "experiments" / "cases_profile_zoom.txt"
LOW = -0.2  # an estimate counts as low when its mu_D is more than 20 % below the truth


def load(folder: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(folder.glob("*__cmaes.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record["case_id"]] = record
    return out


def file_truth(synthetic: Path, case: str, record_truth: float) -> float:
    """mu_D_true from the case's truth.json, which must equal the run record's.

    Without the synthetic cases on disk the record's value is used, and the script says so.
    """
    path = synthetic / case / "truth.json"
    if not path.is_file():
        return record_truth
    value = json.loads(path.read_text(encoding="utf-8"))["mu_d_true"]
    if value != record_truth:
        raise ValueError(f"{case}: mu_d_true {value} in {path}, {record_truth} in the run record")
    return value


def row(record: dict, synthetic: Path) -> dict:
    estimate, metrics = record["estimate"], record["metrics"]
    d_true, rho_true = metrics["mu_d_true"], metrics["mu_rho_true"]
    return {
        "case": record["case_id"],
        "mu_d": estimate["mu_d"],
        "mu_d_true": d_true,
        "truth_file": file_truth(synthetic, record["case_id"], d_true),
        "rel_err_mu_d": (estimate["mu_d"] - d_true) / d_true,
        "mu_rho": estimate["mu_rho"],
        "mu_rho_true": rho_true,
        "rel_err_mu_rho": (estimate["mu_rho"] - rho_true) / rho_true,
        "dice_wt": metrics["dice_wt"],
        "n_eval": record["cost"]["n_evaluations"],
        "prior": record["provenance"]["prior_scale"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "tables",
                        help="folder for e1_primary.json and e1_all16.json (default: outputs/tables)")
    parser.add_argument("--synthetic-root", type=Path, default=REPO,
                        help="folder that holds outputs/synthetic_*; used only to check the truth")
    arguments = parser.parse_args()

    profiled = PROFILED.read_text(encoding="utf-8").split()
    primary, all16 = {}, {}
    for variant, (runs, synthetic) in VARIANTS.items():
        records = load(REPO / runs)
        if len(records) != 16:
            raise ValueError(f"{runs}: {len(records)} CMA-ES records, expected 16")
        synthetic_dir = arguments.synthetic_root / synthetic
        if not synthetic_dir.is_dir():
            print(f"{variant}: {synthetic_dir} not found; truth_file is the run record's mu_d_true")
        rows = [row(records[case], synthetic_dir) for case in sorted(records)]

        picked = [r for r in rows if r["case"] in profiled]
        if len(picked) != len(profiled):
            raise ValueError(f"{runs}: not every case of {PROFILED.name} has a record")
        signed = [r["rel_err_mu_d"] for r in picked]
        primary[variant] = {
            "cases": picked,
            "low_by_20pct": sum(1 for x in signed if x < LOW),
            "median_abs_rel_err_mu_d": statistics.median(abs(x) for x in signed),
            "median_rel_err_mu_d": statistics.median(signed),
            "median_abs_rel_err_mu_rho": statistics.median(abs(r["rel_err_mu_rho"]) for r in picked),
        }

        signed = [r["rel_err_mu_d"] for r in rows]
        all16[variant] = {
            "n": len(rows),
            "low_by_20pct": sum(1 for x in signed if x < LOW),
            "median_rel_err_mu_d": statistics.median(signed),
            "median_abs_rel_err_mu_d": statistics.median(abs(x) for x in signed),
            "median_abs_rel_err_mu_rho": statistics.median(abs(r["rel_err_mu_rho"]) for r in rows),
            "median_dice_wt": statistics.median(r["dice_wt"] for r in rows),
            "priors": sorted({r["prior"] for r in rows}),
            "cases": [{key: r[key] for key in ("case", "rel_err_mu_d", "rel_err_mu_rho", "dice_wt", "prior")}
                      for r in rows],
        }

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    for name, doc in (("e1_primary.json", primary), ("e1_all16.json", all16)):
        (arguments.out_dir / name).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"wrote {arguments.out_dir / name}")
    for variant in VARIANTS:
        p, a = primary[variant], all16[variant]
        print(f"{variant:9s} 4 profiled: median |rel err mu_D| {p['median_abs_rel_err_mu_d']:.3f}, "
              f"low {p['low_by_20pct']}/4; all 16: {a['median_abs_rel_err_mu_d']:.3f}, low {a['low_by_20pct']}/16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
