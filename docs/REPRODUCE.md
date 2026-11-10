# Reproducing our paper

There are three levels, each needing more compute than the one before.

1. **Rebuild the figures and tables from our run records** (minutes; most need no GPU): fetch the
   records, then run the commands in section C.
2. **Rerun one case end to end** (about 6.5 h on one GPU, or minutes with `--smoke`):
   [repro/one_case/README.md](../repro/one_case/README.md).
3. **Rerun every experiment**: sections A and B, then C.

`python scripts/fetch_records.py` downloads the run records into `outputs/`. Or download the archive
from Zenodo (link in README.md once published) and run `python scripts/fetch_records.py --archive <file>`.

Every command runs from the repository root. `$DATA` is the folder that holds the PREDICT-GBM cases:
`src/data/predict_gbm/predict_gbm` after `python scripts/fetch_predict_gbm.py` ([DATA.md](DATA.md)).
Every run record stores its fully resolved config, so the record is the final authority on how it ran.

## A. Synthetic cases

| folder | what | command |
|---|---|---|
| `outputs/synthetic_local/` | the 16 base cases (2 brains × 8 parameter pairs) | `python -m src.data.make_synthetic --case RHUH-0001 --case RHUH-0002 --data-root $DATA --out outputs/synthetic_local` |
| `outputs/synthetic_e1_nonoise/` | the base 16 without noise | same, with `--config configs/e1_nonoise.yaml --out outputs/synthetic_e1_nonoise` |
| `outputs/synthetic_e1_samegrid/` | the base 16 with the truth solved on the 1 mm fit grid | same, with `--config configs/e1_samegrid.yaml --out outputs/synthetic_e1_samegrid` |
| `outputs/synthetic_e1_both/` | the base 16 without noise and on the fit grid | same, with `--config configs/e1_both.yaml --out outputs/synthetic_e1_both` |
| `outputs/synthetic_env/` | the 12 design-range cases (truths inside [0.75, 1.25]) | same, with `--config configs/env_cases.yaml --out outputs/synthetic_env` |
| `configs/x1_generator_seeds.json`, `configs/env_generator_seeds.json` | the generator's tumor seed position x_0 per case (shipped) | `python scripts/x1_generator_seeds.py`, `python scripts/env_generator_seeds.py` |

In the base cases the truth is solved on a 0.5 mm grid and block-averaged to the 1 mm grid both
methods use. Each case folder holds `truth.json` with the true parameters and thresholds, and each set
folder holds `cases_synthetic.txt`, its list of case IDs. The noise of case i uses random seed
`synthetic.seed` + i from the config, so every rebuild gives the same cases. Generation needs a GPU
(about 30 times the work of one calibration solve per case).

## B. Calibration runs

The prior: a penalty in the release's objective that pulls the four parameters toward their start
values; `--prior-scale 1` keeps it, as published, and `0` switches it off.

One case:

```bash
python -m experiments.run_cmaes --case <id> --data-root <data> [--synthetic] --prior-scale <1.0 or 0.0> --config <config> --out <folder>
python -m experiments.run_pinn  --case <id> --data-root <data> [--synthetic] --prior-scale <1.0 or 0.0> --config <config> --out <folder>
```

Many cases, as a Slurm array with one task per line of the case list:

```bash
export DATA_ROOT=<data> OUT_DIR=outputs/<record folder>
sbatch --array=0-<n-1> experiments/slurm/gpu_job.sbatch <cmaes|pinn> <case list> <the same options>
```

The case lists are `experiments/cases_rhuh.txt` (the 20 patients are its first 20 lines),
`experiments/cases_synthetic16.txt` (the base 16), `experiments/cases_profile_zoom.txt` (the 4 profiled
cases) and the `cases_synthetic.txt` of each synthetic set.

The run sets of our paper (configs in `configs/`):

| record folder | method | config | prior | cases |
|---|---|---|---|---|
| `runs_syn` | both | `base.yaml` (PINN: dense network, release step budget) | on | base 16 |
| `runs_syn_np` | both | `base.yaml` | off | base 16 |
| `runs_syn_np_paper` | PINN | `paper_schedule.yaml` | off | base 16 |
| `runs_syn_release_defaults` | PINN | `release_defaults.yaml` (the release as shipped, separable network) | on | base 16 |
| `runs_patient` | both | `base.yaml` | on | 20 patients |
| `runs_patient_np` | CMA-ES | `base.yaml` | off | 20 patients |
| `runs_patient_np_paper` | PINN | `paper_schedule.yaml` | off | 20 patients |
| `runs_patient_release_defaults` | PINN | `release_defaults.yaml` | on | 20 patients |
| `runs_reach`, `runs_reach_prior` | PINN | `paper_schedule_reach.yaml` | off / on | base 16 |
| `runs_reach_3e-4` | PINN | `paper_schedule_reach_3e-4.yaml` | off | 4 profiled cases |
| `runs_env` | CMA-ES / PINN | `env_cases.yaml` / `paper_schedule.yaml` | off | `synthetic_env` |
| `runs_e1_nonoise`, `runs_e1_samegrid`, `runs_e1_both` | CMA-ES | `base.yaml` | off | `synthetic_e1_nonoise`, `synthetic_e1_samegrid`, `synthetic_e1_both` |
| `runs_x1_thfix`, `runs_x1_thfix_seed` | CMA-ES | `x1_thresholds_fixed.yaml`, `x1_thresholds_seed_fixed.yaml` | off | `synthetic_e1_both` |
| `runs_b1b` | CMA-ES | `x1_thresholds_seed_fixed_renorm_off.yaml` | off | `synthetic_e1_both` |
| `runs_a200_on`, `runs_a200_off` | CMA-ES | `x1_thresholds_seed_fixed_a200.yaml`, `x1_thresholds_seed_fixed_renorm_off_a200.yaml` | off | `synthetic_e1_both` |
| `runs_d104a`, `runs_d104b` | CMA-ES | `x1_seed_fixed_thresholds_free.yaml`, `e1_both_seed1.yaml` | off | `synthetic_e1_both` |
| `runs_d104c` | CMA-ES | `env_generator_seed.yaml` | off | `synthetic_env` |
| `runs_nodata` | PINN | `paper_schedule_nodata.yaml` | off | 4 profiled cases |
| `runs_b3` | PINN | `paper_settings_b3.yaml` | off | base 16 |
| `runs_lbfgs_full` | PINN | `paper_schedule_lbfgs_full.yaml` | off | 4 profiled cases |

The objective profiles behind Figure S1 (`outputs/profiles/`) are part of the Zenodo archive. They
come from `python -m scripts.profile_objective --all` (the 21 × 21 grid over [0, 2]²) and
`python -m scripts.profile_objective --all --zoom configs/profile_zoom.yaml` (a fine window per case).

## Labels in file, folder and key names

Some file names, record folders and table keys carry short run labels. The scripts and the record
archive use them, so they stay. The second tissue renormalization, named in several of them: when a
synthetic case is fitted, the loader rescales the white- and gray-matter maps inside the synthetic
tumor, although the generator has already done so inside the source tumor.

| label | meaning |
|---|---|
| `e1` (`e1_nonoise`, `e1_samegrid`, `e1_both`) | the noise-and-grid test: the base cases rebuilt without noise, on the 1 mm fit grid, or both |
| `primary`, `all16` (`e1_primary`, `e1_all16`) | the 4 profiled cases; all 16 cases |
| `x1`, `thfix` | CMA-ES refits of the `e1_both` cases with the thresholds fixed (`thfix`) or the thresholds and x_0 fixed (`thfix_seed`) at the generator's values |
| `a200` | the smoothed threshold with steepness a = 200 in place of 20 |
| `b1`, `b2` (`b1_b2_rescore`) | the truth and the refit point re-scored at a = 20, a = 200 and with hard thresholds |
| `b1b`, `renorm_off` | the thresholds-and-x_0-fixed refit with the second tissue renormalization off |
| `x4` | the check of the second tissue renormalization |
| `b3`, `B3` | the PINN on the training settings printed in Zhang et al. (2025) |
| `d104a`, `d104b`, `d104c` | tumor seed position checks: (a) x_0 at the generator's position, thresholds free; (b) the all-free refit with CMA-ES random seed 1; (c) the design-range cases with x_0 at the generator's position |
| `env`, `ENV-`, `EXP-ENV` | the design-range cases: truths inside the box [0.75, 1.25] of Zhang et al. (2025) |
| `reach`, `EXP-REACH` | the PINN with a separate learning rate for μ_D and μ_ρ, so that they can move as far as CMA-ES's estimates |
| `nodata` | the PINN with no mask data in fine-tuning |
| `paper` (`paper_schedule`, `runs_*_paper`) | the PINN on the training schedule of Zhang et al. (2025) |
| `release_defaults` | the PINN release with every default, as shipped |
| `C1` | both arms on the base cases with `base.yaml` |
| `E2`, `E3` | the matched comparison on the base cases and on the 20 patients, PINN on the schedule of Zhang et al. |
| `E4`, `S2` | the release with its defaults on the 20 patients and on the base cases |
| `np`, `prior0`, `condition_B` | prior off |

## C. Figures and tables

The scripts name figures by short labels. Supplementary figures are in Additional file 1.

| our paper | label | content |
|---|---|---|
| Figure 1 | `F11` | predicted tumors of both methods against the masks |
| Figure 2 | `F7` | estimates against the truth |
| Figure 3 | `F8` | the PINN's μ_D during training |
| Figure S1 | `F2` | objective profiles |
| Figure S2 | `F1` | prior on against prior off |
| Figure S3 | `F4` | matched comparison, synthetic tumors |
| Figure S4 | `F6` | comparison on the 20 patients |

```bash
python -m scripts.make_figures --only F1 F2 F4 F6 F7 F8   # Figures 2, 3, S1-S4
python -m scripts.f11_predictions                         # Figure 1 data (GPU, patient scans)
python -m scripts.make_figures --only F11                 # Figure 1
```

`make_figures` writes to `outputs/figures/` and reads the run records, `outputs/profiles/` and
`outputs/tables/paired_tests.json`. `f11_predictions` writes `outputs/tables/f11_slices.npz` and
`outputs/tables/f11_predictions.json`; it reads the records, `outputs/synthetic_local/` and the patient
scans in `src/data/predict_gbm/predict_gbm`.

Every table is a JSON file in `outputs/tables/`. The shipped copies are the ones our paper prints, so a
rebuilt file can be compared with them directly. GPU marks the scripts that solve the forward model.

**Main text, the thresholds-fixed refit table**

| command | writes | reads |
|---|---|---|
| `python -m scripts.e1_table` | `e1_all16.json` | `runs_syn_np`, `runs_e1_*` |
| `python -m scripts.d104_table` | `d104_seed_checks.json` | `runs_d104*`, `runs_e1_both`, `runs_x1_*`, `runs_env` |
| `python -m scripts.b1b_table` | `b1b_refit.json` | `runs_b1b`, `runs_x1_thfix_seed` |
| `python -m scripts.a200_refit_table` | `a200_refit.json` | `runs_a200_*`, `runs_x1_thfix_seed`, `b1_b2_rescore.json` |

**Main text, the PINN tests table**

| command | writes | reads |
|---|---|---|
| `scripts/compare_pinn_cmaes.py`, below | `e2_synthetic.json`, `s2_synthetic_release_defaults.json`, `e3_patients.json` | run records |
| `scripts/objective_at_estimates.py` (GPU), below | `e2_objective.json`, `reach_objective.json`, `env_objective.json`, `b3_objective.json`, `e3_objective.json` | run records, synthetic cases or `$DATA` |
| `python -m scripts.b3_table` | `b3_paper_settings.json` | `runs_b3`, `runs_syn_np`, `b3_objective.json` |
| `python scripts/paired_tests.py` | `paired_tests.json` | run records |

```bash
python scripts/compare_pinn_cmaes.py --pinn outputs/runs_syn_np_paper --cmaes outputs/runs_syn_np --out outputs/tables/e2_synthetic.json
python scripts/compare_pinn_cmaes.py --pinn outputs/runs_syn_release_defaults --cmaes outputs/runs_syn_np --out outputs/tables/s2_synthetic_release_defaults.json
python scripts/compare_pinn_cmaes.py --pinn outputs/runs_patient_np_paper --cmaes outputs/runs_patient_np --out outputs/tables/e3_patients.json

python scripts/objective_at_estimates.py --pinn outputs/runs_syn_np_paper --cmaes outputs/runs_syn_np --out outputs/tables/e2_objective.json
python scripts/objective_at_estimates.py --pinn outputs/runs_reach --cmaes outputs/runs_syn_np --out outputs/tables/reach_objective.json
python scripts/objective_at_estimates.py --pinn outputs/runs_env --cmaes outputs/runs_env --data-root outputs/synthetic_env --out outputs/tables/env_objective.json
python scripts/objective_at_estimates.py --pinn outputs/runs_b3 --cmaes outputs/runs_syn_np --out outputs/tables/b3_objective.json
python scripts/objective_at_estimates.py --pinn outputs/runs_patient_np_paper --cmaes outputs/runs_patient_np --patients --data-root $DATA --out outputs/tables/e3_objective.json
```

`objective_at_estimates.py` reads `outputs/synthetic_local/` unless `--data-root` says otherwise. Run
`b3_table` after `b3_objective.json` exists.

**Additional file 1**

| command | writes | reads |
|---|---|---|
| `python -m scripts.e1_table` | `e1_primary.json`, `e1_all16.json` | `runs_syn_np`, `runs_e1_*` |
| `python scripts/b1_b2_rescore.py` (GPU) | `b1_b2_rescore.json` | `synthetic_e1_both`, `runs_x1_thfix_seed` |
| `python scripts/x4_renormalisation.py` (GPU) | `x4_renormalisation.json` | `synthetic_e1_both`, `runs_e1_both` |
| `python -m scripts.d104_table` | `d104_seed_checks.json` | as above |
| `python scripts/d104c_objective.py` (GPU) | `d104c_objective.json` | `synthetic_env`, `runs_d104c` |
| `python -m scripts.lbfgs_table` | `lbfgs_full.json` | `runs_lbfgs_full`, `runs_syn_np_paper` |
| none: final estimates in the records | the learning-rate table | `runs_reach`, `runs_reach_3e-4` (4 profiled cases) |

Run `b1_b2_rescore.py` before `a200_refit_table`.

**Numbers in the text**

| command | writes | reads |
|---|---|---|
| `python scripts/c1_accuracy.py` | `c1_accuracy.json` | `runs_syn_np`, `outputs/profiles/` |
| `python scripts/objective_minimum_vs_truth.py` | `objective_minimum_vs_truth.json` | `outputs/profiles/`, `runs_syn_np` |
| `python -m scripts.nodata_table` | `nodata_test.json` | `runs_nodata`, `runs_syn_np_paper` |
| `python scripts/e4_displacement.py` | `e4_displacement.json` | `runs_patient_release_defaults`, `runs_patient` |

## Not reproducible from this repository

* **The clinical pilot (17 scans from Evangelisches Klinikum Bethel, Bielefeld) and the clinical row of
  Figure 1.** They need patient images that are not public ([DATA.md](DATA.md)). Our paper reports the
  pilot as a summary only.
* **Bit-identical CMA-ES reruns of runs that used random seed 0.** In `pycma`, random seed 0 seeds from
  the clock. Reruns with a fixed random seed reproduce our results to the precision shown in the README.
