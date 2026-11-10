# PINN settings: Zhang et al. (2025), their code, and our runs

Sources: Zhang et al. (2025, pp. 3–6, 14–15); their PyTorch release `external/pinngbm-torch` at commit
99af20f (file and line given as `Options.py:82`); our configs `configs/paper_schedule.yaml` (main runs)
and `configs/paper_settings_b3.yaml` with `experiments/pinn_paper_settings.py` and
`experiments/run_pinn.py` (paper-settings run).

"Main runs" are the matched comparison, the design-range PINN and, with its one change, the reach test.
"Paper-settings run" is `runs_b3`, the PINN on the training settings that Zhang et al. print.

Symbols: WT is the whole tumor (FLAIR mask) and TC the tumor core (T1c mask); SEG is the mask data term;
u_c is the density threshold of a mask; ū is the characteristic solution, the finite-difference
solution at μ_D = μ_ρ = 1, and L_char the pre-training term that pulls the network toward ū; rD is the
release's name for μ_D.

## A. Training settings

The paper-settings run takes the value of Zhang et al. wherever they print one.

| # | setting | Zhang et al. (locator) | release default | our main runs | our paper-settings run |
|---|---|---|---|---|---|
| 1 | SEG weight | 1e-3, one term L_SEG (p. 6; App. E p. 14) | `seg1` 1e-5 + `seg2` 1e-5 (`Options.py:82`) | 1e-5 each | 1e-3 each (note 1) |
| 2 | SEG loss form | Eq. (15)–(16), p. 5: mean of (H(φu − u_c) − y)², φ inside H, summed over WT and TC, not normalized | φ weights the error, each term divided by mean(y²) (`GBMproblem.py:22–23, 279–298`) | release form | Zhang et al.'s form, `paper_seg_loss` (note 2) |
| 3 | Heaviside steepness a | 20 (Eq. 15) | 20, hard-coded (`GBMproblem.py:19`) | 20 | 20 |
| 4 | PDE residual weight | 1 (p. 6) | 1 | 1 | 1 |
| 5 | pre-training loss | L_PDE + L_char, each weight 1 (Eq. 14; p. 6) | `res` 1, `uchar_res` 10 (`Options.py:77`) | 1 and 10 | 1 and 1 |
| 6 | parameter penalties | box penalties only, weight 1 (Eq. 18–19, p. 5) | the prior, 1e-3 each, plus a threshold box `thrange` that is not in the loss list | prior off, no box term | prior off; box penalties, weight 1 (note 3) |
| 7 | residual in time | points sampled densely early: truncated exponential, λ = 0.5, t ∈ [0, 1] (App. F p. 15) | t uniform, residual weighted by exp(−10 t) (`texp_weight` 10, `GBMproblem.py:247`) | release | Zhang et al.'s sampling, weight off (`texp_weight` 0) |
| 8 | residual in space | resampled with weight 1/r² from the tumor center, in the ball where ū > 0.001 (App. F p. 15) | uniform over the ball where ū > 0.01 (`Engine.py:491`); `rpow_weight` 0 | release | 1/r² in the ball ū > 0.001 (note 4) |
| 9 | collocation points, pre-training | N_r = 20,000, one set (App. E p. 14) | a new batch of 200,000 every step | 200,000 per step | 20,000, fixed |
| 10 | collocation points, fine-tuning | N_r = N_dat = 5,000 (App. E p. 14) | new batches: 200,000 residual, 100,000 data | 200,000 / 100,000 per step | 5,000 / 5,000, fixed (note 5) |
| 11 | L_char points | the residual points (Eq. 13) | the release's space-time grid | release | release grid, 20,000 fixed (note 6) |
| 12 | network | fully connected, 4 hidden layers × 128, tanh (App. E p. 14) | modified MLP with a random-Fourier input layer (`arch` mmlp, `fourier` true) | release | plain MLP, tanh (note 7) |
| 13 | hard initial-condition ansatz | u = t·u_NN + u_0 (Sec. 2.3, p. 4) | the same times a ball factor (r²max − r²)/r²max (`force_bc` true, `GBMproblem.py:559–565`) | release | Zhang et al.'s ansatz (`force_bc` false) |
| 14 | optimizer | "ADAM" (App. E p. 14) | Adam with AMSGrad (`Options.py:89`) | AMSGrad | plain Adam |
| 15 | learning rates | 1e-3 pre-training, 1e-4 fine-tuning (App. E p. 14) | 1e-3 and 1e-3 | 1e-3, 1e-4 | 1e-3, 1e-4 |
| 16 | step budgets | 20,000 max pre-training; 5,000 fine-tuning (App. E p. 14) | 4,000 and 1,000 | 20,000, 5,000 | 20,000, 5,000 |
| 17 | early stopping | pre-training: stop if the training or testing loss does not decrease for 1,000 iterations (App. E p. 14); fine-tuning: not printed | none | TensorFlow rule in both stages (note 8) | as main runs (note 8) |
| 18 | L-BFGS-B after Adam, pre-training | maxcor 100, maxls 50, tol 2.3e-9 (App. E p. 14); maximum iterations not printed | none | none | torch L-BFGS (note 9) |
| 19 | optimizer state at the switch | not printed | carried over (`traintype` full) | carried over | carried over |
| 20 | phase field in the flux | φ_τ = φ + τ, τ > 0 small (Eq. 3, p. 3); τ not printed | φ (`GBMproblem.py:233`) | φ | φ (note 10) |
| 21 | residual form | φ ∂u/∂t (Eq. 3) | φ u_t (`GBMproblem.py:234`) | the same | the same |
| 22 | random seed (network initialization) | not printed | 0 | 0 | 0 |

Notes:

1. Zhang et al. print one weight, 1e-3, for one term L_SEG; we give each of the release's two terms
   1e-3. The split reading, 5e-4 each, was not run.
2. Without Zhang et al.'s form, a weight of 1e-3 on the release's normalized term would be 1/mean(y²)
   times theirs: about 30 times for WT, which covers about 3 % of the ball's voxels on
   SYN-RHUH-0002-d1.2-r0.6, and more for the smaller TC.
3. Box penalties on the problem's own boxes: `thrange` [0, 1] for the thresholds and `murange` [0, 2]
   for μ_D and μ_ρ (added in `pinn_paper_settings.py`). They act only at the box edges.
4. r is floored at half a voxel, which Zhang et al. do not print. The ball threshold comes from
   `run_pinn._patch_ball_threshold`.
5. The data points are uniform over the ball's voxels at t = 1 (p. 5).
6. The release stores ū at 100 time snapshots only, so ū is not known at the residual points.
7. `arch` mlp, `fourier` false; `DenseNet.py:30–88`: an input layer and 3 hidden layers, 4 tanh layers
   of 128 in all.
8. Patience 1,000, min_delta 1e-6, burn-in 1,000, and a stop when rD < 0.05. Zhang et al. print no rule
   for fine-tuning, so the TensorFlow rule is kept there. In the paper-settings run the training loss is
   watched on the fixed collocation set.
9. History 100, tolerance 2.3e-9, run to its own tolerance with a cap of 10,000 iterations and
   evaluations, as in `runs_lbfgs_full`. The torch line search takes at most 25 steps, fixed, in place
   of 50.
10. φ_τ cannot be set without a value for τ.

## B. Settings that would change the problem itself

Reported, not used in the paper-settings run.

| setting | Zhang et al. (locator) | ours, both arms | why not in the paper-settings run |
|---|---|---|---|
| box for μ_D, μ_ρ | [0.75, 1.25] (p. 5) | [0, 2] | changes the problem; the design-range test covers the band |
| threshold boxes | u_c^WT ∈ [0.2, 0.5], u_c^TC ∈ [0.5, 0.8] (p. 5) | [0, 1] each | changes the problem (the CMA-ES arm searches [0, 1]) |
| trainable tumor seed position x_0 | part of the trainable set Θ (p. 6) | fixed at the mask centroid | changes the problem |
| initial condition | u_0 = 0.1 exp(−0.1 ‖x − x_0‖²) (p. 3) | h_0 = 0.5, r_0 = 0.01 (`Options.py:38–39`) | shared with CMA-ES; changing it for one arm changes the model |
| start value of u_c^TC | 0.6 in the grid search (p. 4) | 0.70 (`Options.py:56`), also the start | the start is held equal across arms |
| geometry | registered atlas (p. 2) | the benchmark's maps | data, not a training setting |

## C. Checked and found as printed

Fine-tuning learning rate 1e-4 and 5,000 steps; pre-training rate 1e-3; Heaviside a = 20; the
residual written in non-divergence form with ∇(Pφ) from finite differences (Eq. 11, p. 4); data points
at t = 1 (p. 5); WM/GM ratio 10 (p. 2).

`tests/test_pinn_paper_settings.py` tests the additions of the paper-settings run.
