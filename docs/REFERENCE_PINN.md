# The reference PINN, and what this repository does around it

The PINN arm is the published method of Zhang et al. (2025), run from the authors' PyTorch release
<https://github.com/Rayzhangzirui/pinngbm-torch> at commit 99af20f (`scripts/fetch_reference_pinn.sh`).
No file of the release is edited. `experiments/run_pinn.py` imports it, sets options the release
offers, wraps its training loop, and scores the result with the same metrics as CMA-ES. Every change
below is recorded in each run record (`provenance`, `config`, `early_stopping`).

## Options set through the release's own interface

* The dense network (`separable: false`), not the release's separable default.
* The step budget of Zhang et al. (20,000 pre-training, 5,000 fine-tuning steps) in place of the
  release's 4,000 + 1,000, with learning rates 1e-3 and 1e-4 (`configs/paper_schedule.yaml`).
* The prior: a penalty in the release's objective that pulls the four parameters toward their start
  values; `--prior-scale 1` keeps it, as published, and `0` switches it off. The same option acts on
  the CMA-ES arm.
* Every other setting at the release default. The full list against Zhang et al. is
  [PINN_SETTINGS.md](PINN_SETTINGS.md).

## Additions (wrappers in `experiments/`)

| file | what it adds | used by |
|---|---|---|
| `pinn_early_stopping.py` | the original TensorFlow code's early stopping (patience 1,000, min_delta 1e-6, burn-in 1,000), which the PyTorch release lacks | every run on the schedule of Zhang et al. |
| `pinn_lbfgs.py` | the original code's L-BFGS stage after Adam, in pre-training | `runs_lbfgs_full`, `runs_b3` |
| `pinn_param_lr.py` | a separate Adam learning rate for μ_D and μ_ρ in fine-tuning; our modification, not part of Zhang et al.'s method | the reach test only (`runs_reach*`) |
| `pinn_paper_settings.py` | the training settings Zhang et al. print where the release differs (data weight, loss form, sampling, network, ansatz) | `runs_b3` only |
| `run_pinn.py` `seg_weight_scale` | both mask weights scaled (0 = no data in fine-tuning) | the no-data test (`runs_nodata`) |

## Runtime patches (applied in memory, never written to the release)

* `_patch_scalar_conversion`: the release passes 1 × 1 arrays to `float()`, which NumPy 2 rejects; the
  patch converts them first. Results are unchanged.
* `_patch_ball_threshold`: the collocation ball where the characteristic solution exceeds 0.001
  (Zhang et al., App. F) in place of the release's hard-coded 0.01. Only with `pinn_paper_settings`
  (`runs_b3`).

## What the release does not have, and the original TensorFlow code does

The numbers of Zhang et al. came from their original TensorFlow code
(<https://github.com/Rayzhangzirui/pinngbm>, commit 41e2eeb). Three of its features are not in the
PyTorch release, and our main runs do not add them:

* the box [0.75, 1.25] on μ_D and μ_ρ;
* L-BFGS-B after Adam (tested separately, see the table above);
* a trainable tumor seed position x_0, in its patient mode (`trainx0`).

Our paper states its conclusions for the release as run, not for PINNs in general.
