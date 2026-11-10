"""Paired PINN vs CMA-ES summary for one pair of run folders.

    python scripts/compare_pinn_cmaes.py --pinn outputs/runs_patient_np_paper --cmaes outputs/runs_patient_np --out outputs/tables/e3_patients.json

Per case: estimates, Dice, stop reasons. Pooled: exact sign test on Dice (two-sided, ties dropped), the
spread of each arm's estimates, and the Pearson r between the arms' estimates. Truth errors are
added when the records carry a truth (synthetic cases).
"""

import argparse
import glob
import json
import os
import statistics
from math import comb


def sign_test(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def load(folder, method):
    out = {}
    for f in glob.glob(os.path.join(folder, f"*__{method}.json")):
        out[os.path.basename(f).split("__")[0]] = json.load(open(f, encoding="utf-8"))
    return out


def truth_of(record):
    m = record["metrics"]
    return m.get("mu_d_true"), m.get("mu_rho_true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", required=True)
    ap.add_argument("--cmaes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P, C = load(args.pinn, "pinn"), load(args.cmaes, "cmaes")
    cases = sorted(set(P) & set(C))
    rows = []
    for k in cases:
        p, c = P[k], C[k]
        stages = (p.get("early_stopping") or {}).get("stages") or {}
        rows.append({
            "case": k,
            "pinn": {**{m: p["estimate"][m] for m in ("mu_d", "mu_rho")},
                     **{m: p["metrics"][m] for m in ("dice_wt", "dice_tc")},
                     "stop": {s: v.get("stop_reason") for s, v in stages.items()},
                     "prior_scale": p["provenance"].get("prior_scale"),
                     "wall_clock_h": p["cost"]["wall_clock_s"] / 3600},
            "cmaes": {**{m: c["estimate"][m] for m in ("mu_d", "mu_rho")},
                      **{m: c["metrics"][m] for m in ("dice_wt", "dice_tc")},
                      "prior_scale": c["provenance"].get("prior_scale")},
            "truth": dict(zip(("mu_d", "mu_rho"), truth_of(c))),
        })
    summary = {"n": len(cases), "pinn_dir": args.pinn, "cmaes_dir": args.cmaes,
               "missing_pinn": sorted(set(C) - set(P)), "missing_cmaes": sorted(set(P) - set(C))}
    for m in ("dice_wt", "dice_tc"):
        w = sum(r["cmaes"][m] > r["pinn"][m] for r in rows)
        l = sum(r["cmaes"][m] < r["pinn"][m] for r in rows)
        summary[m] = {"cmaes_wins": w, "pinn_wins": l, "ties": len(rows) - w - l, "p_sign": sign_test(w, l),
                      "median_pinn": statistics.median(r["pinn"][m] for r in rows),
                      "median_cmaes": statistics.median(r["cmaes"][m] for r in rows)}
    for m in ("mu_d", "mu_rho"):
        xp = [r["pinn"][m] for r in rows]
        xc = [r["cmaes"][m] for r in rows]
        summary[m] = {"pinn_min": min(xp), "pinn_median": statistics.median(xp), "pinn_max": max(xp),
                      "cmaes_min": min(xc), "cmaes_median": statistics.median(xc), "cmaes_max": max(xc),
                      "r_pinn_vs_cmaes": pearson(xp, xc)}
        tr = [r["truth"][m] for r in rows]
        if all(t is not None for t in tr):
            summary[m]["r_pinn_vs_truth"] = pearson(xp, tr)
            summary[m]["r_cmaes_vs_truth"] = pearson(xc, tr)
            summary[m]["median_rel_err_pinn"] = statistics.median((a - t) / t for a, t in zip(xp, tr))
            summary[m]["median_rel_err_cmaes"] = statistics.median((a - t) / t for a, t in zip(xc, tr))
    stops = {}
    for r in rows:
        for s, v in r["pinn"]["stop"].items():
            stops.setdefault(s, {}).setdefault(v, 0)
            stops[s][v] += 1
    summary["pinn_stop_reasons"] = stops
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
