# One case, end to end

`run_one_case.py` reruns the whole pipeline of our paper on one synthetic case,
**SYN-RHUH-0002-d1.2-r0.6** (row 1 of Figure 1 in our paper), and checks the result against our run
records ([expected.json](expected.json)).

| step | what | time on one NVIDIA A100 |
|---|---|---|
| 1 | build the case from the public brain RHUH-0002: truth on a 0.5 mm grid, block-averaged to 1 mm, correlated noise, two masks. The noise of a case depends on its position in the generator's list, so the script builds the first seven RHUH-0002 cases, of which this case is the seventh | minutes |
| 2 | CMA-ES fit (`configs/base.yaml`, prior off, random seed 1, 600 evaluations) | about 30 min |
| 3 | PINN fit, the reference release on the schedule of Zhang et al. (`configs/paper_schedule.yaml`, prior off) | about 6 h |
| 4 | compare with our run records | seconds |

The prior is a penalty in the release's objective that pulls the four parameters toward their start
values; both fits switch it off (`--prior-scale 0.0`). CMA-ES uses random seed 1 because `pycma`
replaces random seed 0 with one from the clock.

```bash
bash scripts/fetch_reference_pinn.sh                # the reference PINN
python scripts/fetch_predict_gbm.py                 # the source brains RHUH-0001 and RHUH-0002 (docs/DATA.md)
python repro/one_case/run_one_case.py --data-root src/data/predict_gbm/predict_gbm --smoke   # minutes; pipeline only
python repro/one_case/run_one_case.py --data-root src/data/predict_gbm/predict_gbm           # about 6.5 h on one GPU
```

The `--smoke` run builds only the first RHUH-0002 case, SYN-RHUH-0002-d0.2-r0.6, with short budgets
and small batches (`configs/smoke_one_case_small_gpu.yaml`, which fits an 8 GB GPU). It checks that
every step runs and writes its record, not the numbers.

## What is checked

The full run checks 14 values and findings against [expected.json](expected.json):

* **The synthetic case.** The true μ_D and μ_ρ to within 1e-6, and the voxel counts of the two masks
  to within 1 %.
* **CMA-ES:** μ_D 1.180 and μ_ρ 0.677, each to within 0.01; Dice WT 0.828 and Dice TC 0.703, each to
  within 0.005.
* **PINN:** μ_D 0.655 and μ_ρ 0.866, each to within 0.03; Dice WT 0.817 to within 0.01 and Dice TC
  0.683 to within 0.015. The PINN's tolerances are wider because GPU arithmetic is not
  bit-reproducible.
* **The findings of our paper on this case:** the PINN's μ_D ends below CMA-ES's, both below the truth
  1.578, and CMA-ES fits the whole-tumor mask better.

WT is the whole tumor (FLAIR mask), TC the tumor core (T1c mask). Outputs go to
`outputs/repro_one_case/`. The script exits with 0 only if every check passes.

## A full run

A full run on one NVIDIA A100 took 6 h 34 min and passed all 14 checks: CMA-ES μ_D 1.1797,
μ_ρ 0.6770, Dice WT 0.8282, Dice TC 0.7027; PINN μ_D 0.6639, μ_ρ 0.8652, Dice WT 0.8179,
Dice TC 0.6848.
