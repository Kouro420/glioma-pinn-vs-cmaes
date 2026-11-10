"""Accuracy and bias of the estimates: the numbers our paper quotes.

Prior off, 16 synthetic cases, both arms. For each arm: median relative error of the
three reported quantities (D t_end, rho t_end, lambda); how often diffusion is worse than
proliferation; how often each estimate lies below the truth; rank agreement with the truth; the range
of the PINN's mu_D estimates. Relative error of mu_D equals relative error of D t_end, since
mu_D is proportional to D t_end.

Also: whether CMA-ES stopped on the objective's own minimum in the four profiled cases; how much the
pooled mask-fit/mu_rho correlation depends on the four RHUH-0001 r1.3 cases; and whether the
basin widths of the objective profiles (Figure S1) at twice the minimum reach the edge of the [0, 2] box.

Writes outputs/tables/c1_accuracy.json. Run from the repository root.
"""
import csv
import datetime
import glob
import io
import json
import statistics as st


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


R = {'cmaes': [], 'pinn': []}
for f in sorted(glob.glob('outputs/runs_syn_np/*.json')):
    r = json.load(open(f))
    m, e = r['metrics'], r['estimate']
    R[r['method']].append({
        'case': r['case_id'], 'mu_d': e['mu_d'], 'mu_rho': e['mu_rho'],
        'mu_d_true': m['mu_d_true'], 'mu_rho_true': m['mu_rho_true'],
        'rel_d': m['rel_err_d_eff'], 'rel_rho': m['rel_err_rho_eff'], 'rel_lambda': m['rel_err_lambda'],
        'lambda_est': m['lambda_estimate_mm'], 'lambda_true': m['lambda_truth_mm'],
        'err_mu_rho': m['err_mu_rho'], 'dice': (m['dice_wt'] + m['dice_tc']) / 2})

arms = {}
for arm, v in R.items():
    arms[arm] = {
        'n': len(v),
        'median_rel_err_D_t': round(st.median(x['rel_d'] for x in v), 3),
        'median_rel_err_rho_t': round(st.median(x['rel_rho'] for x in v), 3),
        'median_rel_err_lambda': round(st.median(x['rel_lambda'] for x in v), 3),
        'diffusion_worse_than_proliferation': sum(1 for x in v if x['rel_d'] > x['rel_rho']),
        'mu_d_below_truth': sum(1 for x in v if x['mu_d'] < x['mu_d_true']),
        'mu_rho_below_truth': sum(1 for x in v if x['mu_rho'] < x['mu_rho_true']),
        'lambda_below_truth': sum(1 for x in v if x['lambda_est'] < x['lambda_true']),
        'spearman_mu_d_vs_truth': round(spearman([x['mu_d'] for x in v], [x['mu_d_true'] for x in v]), 2),
        'spearman_mu_rho_vs_truth': round(spearman([x['mu_rho'] for x in v], [x['mu_rho_true'] for x in v]), 2),
        'pearson_mu_rho_vs_truth': round(pearson([x['mu_rho'] for x in v], [x['mu_rho_true'] for x in v]), 2),
        'mu_d_estimate_range': [round(min(x['mu_d'] for x in v), 3), round(max(x['mu_d'] for x in v), 3)],
        'mu_d_truth_range': [round(min(x['mu_d_true'] for x in v), 2), round(max(x['mu_d_true'] for x in v), 2)],
        'r_dice_vs_err_mu_rho': round(pearson([x['dice'] for x in v], [x['err_mu_rho'] for x in v]), 3),
    }
keep = [x for x in R['pinn'] if not ('RHUH-0001' in x['case'] and 'r1.3' in x['case'])]
arms['pinn']['r_dice_vs_err_mu_rho_without_RHUH0001_r1.3'] = round(
    pearson([x['dice'] for x in keep], [x['err_mu_rho'] for x in keep]), 3)
arms['pinn']['n_without_RHUH0001_r1.3'] = len(keep)

# CMA-ES final estimate against the fine-grid minimum, and F2 widths at 2x the minimum
cm = {x['case']: x for x in R['cmaes']}
profiles = []
for jf in sorted(glob.glob('outputs/profiles/*__prior0__zoom0.02.json')):
    cid = json.load(open(jf))['case_id']
    fine = list(csv.DictReader(open(jf.replace('.json', '.csv'))))
    b = min(fine, key=lambda r: float(r['total']))
    coarse = list(csv.DictReader(open(jf.replace('zoom0.02', 'grid21').replace('.json', '.csv'))))
    fmin = min(float(r['total']) for r in coarse)
    inside = [r for r in coarse if float(r['total']) <= 2.0 * fmin]
    d = [float(r['mu_d']) for r in inside]
    p = [float(r['mu_rho']) for r in inside]
    profiles.append({
        'case': cid,
        'cmaes_final': [round(cm[cid]['mu_d'], 3), round(cm[cid]['mu_rho'], 3)],
        'fine_grid_minimum': [float(b['mu_d']), float(b['mu_rho'])],
        'cmaes_within_one_fine_step': abs(cm[cid]['mu_d'] - float(b['mu_d'])) <= 0.02
        and abs(cm[cid]['mu_rho'] - float(b['mu_rho'])) <= 0.02,
        'coarse_2x_mu_d_extent': [min(d), max(d)], 'coarse_2x_mu_rho_extent': [min(p), max(p)],
        'coarse_2x_both_reach_box_edge': (max(d) >= 2.0 or min(d) <= 0.001 + 1e-9) and
                                         (max(p) >= 2.0 or min(p) <= 0.001 + 1e-9),
    })

# --- reference points, bootstrap interval, CMA-ES tail movement, fine-grid basin widths ---
import random
cma = R['cmaes']
# reference points for the relative errors: the starting value mu = 1 (no calibration)
arms['cmaes']['median_rel_err_mu_d_at_start_mu1'] = round(st.median(abs(1 - x['mu_d_true']) / x['mu_d_true'] for x in cma), 3)
arms['cmaes']['median_rel_err_mu_rho_at_start_mu1'] = round(st.median(abs(1 - x['mu_rho_true']) / x['mu_rho_true'] for x in cma), 3)
# mask fit against the relative mu_D error, CMA-ES, with a 4,000-sample bootstrap interval
dice = [x['dice'] for x in cma]; reld = [x['rel_d'] for x in cma]
rng = random.Random(20260924)
boots = []
for _ in range(4000):
    idx = [rng.randrange(len(cma)) for _ in cma]
    a_, b_ = [dice[i] for i in idx], [reld[i] for i in idx]
    if len(set(a_)) > 1 and len(set(b_)) > 1:
        boots.append(pearson(a_, b_))
boots.sort()
arms['cmaes']['r_dice_vs_rel_err_mu_d'] = round(pearson(dice, reld), 3)
arms['cmaes']['r_dice_vs_rel_err_mu_d_boot95'] = [round(boots[int(0.025 * len(boots))], 2), round(boots[int(0.975 * len(boots)) - 1], 2)]
# how far CMA-ES's mu_D moved over the last fifth of its 600 evaluations, per case
tail = []
for f in sorted(glob.glob('outputs/runs_syn_np/*__cmaes.json')):
    h = sorted(json.load(open(f))['history'], key=lambda e: e['evaluation'])
    best, bs, bm = [], float('inf'), h[0]['mu_d']
    j = 0
    for ev in range(601):
        while j < len(h) and h[j]['evaluation'] <= ev:
            if h[j]['score'] < bs:
                bs, bm = h[j]['score'], h[j]['mu_d']
            j += 1
        best.append(bm)
    tail.append(max(best[480:]) - min(best[480:]))
arms['cmaes']['max_mu_d_movement_last_fifth_any_case'] = round(max(tail), 4)
# 1.05 basin widths on the fine grid, absolute and relative to the minimum's own coordinate
for p_ in profiles:
    fine = list(csv.DictReader(open(glob.glob('outputs/profiles/' + p_['case'] + '__prior0__zoom0.02.csv')[0])))
    fmin = min(float(r['total']) for r in fine)
    ins = [r for r in fine if float(r['total']) <= 1.05 * fmin]
    wd = max(float(r['mu_d']) for r in ins) - min(float(r['mu_d']) for r in ins)
    wr = max(float(r['mu_rho']) for r in ins) - min(float(r['mu_rho']) for r in ins)
    md, mr = p_['fine_grid_minimum']
    p_['fine_1p05_width_abs'] = [round(wd, 3), round(wr, 3)]
    p_['fine_1p05_width_rel_to_minimum'] = [round(wd / md, 2), round(wr / mr, 2)]
    p_['coarse_2x_widths_determinable'] = not (p_['coarse_2x_mu_d_extent'][1] >= 2.0 or p_['coarse_2x_mu_d_extent'][0] <= 0.001 + 1e-9) or \
        not (p_['coarse_2x_mu_rho_extent'][1] >= 2.0 or p_['coarse_2x_mu_rho_extent'][0] <= 0.001 + 1e-9)
# -------------------------------------------------------------------------------------------

doc = {'generated': datetime.date.today().isoformat(), 'what': __doc__.strip(),
       'source': 'outputs/runs_syn_np; outputs/profiles', 'arms': arms, 'profiles': profiles,
       'cmaes_on_minimum_in': sum(1 for p in profiles if p['cmaes_within_one_fine_step']),
       'f2_2x_both_widths_reach_box_edge_in': sum(1 for p in profiles if p['coarse_2x_both_reach_box_edge'])}
io.open('outputs/tables/c1_accuracy.json', 'w', encoding='utf-8').write(json.dumps(doc, indent=2))
print(json.dumps({'arms': arms, 'cmaes_on_minimum_in': doc['cmaes_on_minimum_in'],
                  'f2_2x_both_widths_reach_box_edge_in': doc['f2_2x_both_widths_reach_box_edge_in'],
                  'extents': [(p['case'], p['coarse_2x_mu_d_extent'], p['coarse_2x_mu_rho_extent']) for p in profiles]},
                 indent=1))
