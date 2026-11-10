"""Where does the objective's own minimum sit, relative to the truth? No optimizer involved.

For each of the four profiled synthetic cases (prior off), the minimum of the objective on the
fine grid (step 0.02) is compared with the true multipliers. If the minimum misses mu_D by a
larger fraction than mu_rho, then even an optimizer that found the exact minimum would recover
diffusion worse than proliferation: the error floor is in the data and the objective, not in a
search. Four cases, one noise realization each -- an observation, not a tested claim.

Also writes, for the converged CMA-ES arm on the sixteen prior-off cases, the median error in each
multiplier, so that our paper can state the same asymmetry for the one arm that finished its search.

Writes outputs/tables/objective_minimum_vs_truth.json. Run from the repository root.
"""
import csv
import datetime
import glob
import io
import json
import statistics as st

cases = []
for jf in sorted(glob.glob('outputs/profiles/*__prior0__zoom0.02.json')):
    meta = json.load(open(jf))
    rows = list(csv.DictReader(open(jf.replace('.json', '.csv'))))
    best = min(rows, key=lambda r: float(r['total']))
    md, mr = float(best['mu_d']), float(best['mu_rho'])
    td, tr = meta['truth']['mu_d_true'], meta['truth']['mu_rho_true']
    cases.append({'case': meta['case_id'], 'prior_scale': meta['prior_scale'],
                  'minimum_mu_d': md, 'minimum_mu_rho': mr,
                  'truth_mu_d': round(td, 4), 'truth_mu_rho': round(tr, 4),
                  'miss_mu_d_pct': round(100 * (md - td) / td, 1),
                  'miss_mu_rho_pct': round(100 * (mr - tr) / tr, 1)})

cma = [json.load(open(f))['metrics'] for f in sorted(glob.glob('outputs/runs_syn_np/*__cmaes.json'))]
doc = {
    'generated': datetime.date.today().isoformat(),
    'what': __doc__.strip(),
    'profiles': cases,
    'n_profiles': len(cases),
    'mu_d_missed_by_larger_fraction_than_mu_rho': sum(
        1 for c in cases if abs(c['miss_mu_d_pct']) > abs(c['miss_mu_rho_pct'])),
    'miss_mu_d_pct_range': [min(c['miss_mu_d_pct'] for c in cases), max(c['miss_mu_d_pct'] for c in cases)],
    'miss_mu_rho_pct_range': [min(c['miss_mu_rho_pct'] for c in cases), max(c['miss_mu_rho_pct'] for c in cases)],
    'cmaes_condition_B': {
        'n': len(cma),
        'median_err_mu_d': round(st.median(m['err_mu_d'] for m in cma), 3),
        'median_err_mu_rho': round(st.median(m['err_mu_rho'] for m in cma), 3),
        'median_rel_err_d_eff': round(st.median(m['rel_err_d_eff'] for m in cma), 3),
        'median_rel_err_rho_eff': round(st.median(m['rel_err_rho_eff'] for m in cma), 3),
    },
}
io.open('outputs/tables/objective_minimum_vs_truth.json', 'w', encoding='utf-8').write(json.dumps(doc, indent=2))
for c in cases:
    print(c)
print({k: doc[k] for k in ('mu_d_missed_by_larger_fraction_than_mu_rho', 'miss_mu_d_pct_range',
                           'miss_mu_rho_pct_range', 'cmaes_condition_B')})
