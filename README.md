# glioma-pinn-vs-cmaes

Code for the paper:

> K. Moini, M. Soleimani, S. Nagel, Y. Kalou, M. Simon, O. Nikoubashman, P. Junker, M. Esmaeilzadeh.
> *How much of a Fisher–Kolmogorov glioma model can one structural MRI calibrate? A matched comparison
> of a physics-informed neural network and CMA-ES.* Manuscript; the reference is updated on publication.

## Overview

Glioblastoma grows well beyond the edge that MRI shows, and a tumor growth model fitted to the patient
could, in principle, estimate where it has already spread. In practice the model has to be calibrated
from a single pre-operative scan. We asked how much of the Fisher–Kolmogorov model,

    ∂u/∂t = ∇·(D(x) ∇u) + ρ u (1 − u),

one structural MRI can actually determine, and whether a physics-informed neural network (PINN)
calibrates it better than a classical optimizer.

To keep the comparison fair, both methods solve the same forward model, fit the same two tumor masks
(T1c and FLAIR) with the same score, start from the same point and run on the same GPU type. One is
CMA-ES with a finite-difference solver; the other is the published PINN of
[Zhang et al. (2025)](https://doi.org/10.1016/j.media.2024.103423), run from the authors' own code. We
test both on synthetic tumors, where the true parameters are known, and on 20 patients of the public
[PREDICT-GBM](https://doi.org/10.1038/s41746-026-03194-0) benchmark.

## Contents

| included in this repository | not included, and how to get it |
|---|---|
| **All the code** that produces the numbers, tables and figures of our paper. | **The PINN of Zhang et al. (2025).** `scripts/fetch_reference_pinn.sh` downloads the exact version we used. |
| **The settings of each experiment**, one YAML file per experiment in `configs/`. | **The patient scans** from PREDICT-GBM (RHUH cohort, one folder per patient: RHUH-0001, RHUH-0002, …). `scripts/fetch_predict_gbm.py` downloads the patients you need ([docs/DATA.md](docs/DATA.md)). |
| **The result tables** reported in our paper, in `outputs/tables/`. | **The synthetic tumors**, grown with known parameters in the brains of patients RHUH-0001 and RHUH-0002. The code rebuilds them with fixed random seeds. |
| **Tests** for the solver and the settings, in `tests/`. | **The run records**, one result file per calibration run. They are archived on Zenodo ([doi:10.5281/zenodo.24107357](https://doi.org/10.5281/zenodo.24107357)); `scripts/fetch_records.py` downloads them. |
| **A worked example** on one synthetic tumor that runs the whole pipeline, in `repro/one_case/`. | **The clinical pilot** (17 patients). These are patient data and are not shared; our paper reports them as a summary. |

## Installation

You need Python 3.11 and an NVIDIA GPU with CUDA. A PINN fit needs about 22 GB of GPU memory, a CMA-ES
fit about 3 GB.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
bash scripts/fetch_reference_pinn.sh      # the PINN of Zhang et al., at the version we used
python -m pytest -q tests                 # some tests skip until the data are present, and say why
```

## Usage

**Rebuild the figures and tables from our run records.** This needs no new calibration runs.

```bash
python scripts/fetch_records.py
python -m scripts.make_figures
```

Figure 1 also needs the patient scans and a GPU ([docs/REPRODUCE.md](docs/REPRODUCE.md), section C).

**Run the worked example.** It grows one synthetic tumor with known parameters in the brain of patient
RHUH-0002, estimates the parameters back with CMA-ES and with the PINN, and checks both results against
the values in our paper. The first command fetches the two brains that all synthetic tumors are built
on ([docs/DATA.md](docs/DATA.md)).

```bash
python scripts/fetch_predict_gbm.py
python repro/one_case/run_one_case.py --data-root src/data/predict_gbm/predict_gbm --smoke
python repro/one_case/run_one_case.py --data-root src/data/predict_gbm/predict_gbm
```

The `--smoke` run takes a few minutes and only checks that every step works. The full run takes about
6.5 hours on one GPU and checks the numbers.

**Rerun every experiment.** [docs/REPRODUCE.md](docs/REPRODUCE.md) lists each experiment with its settings
and the command behind every figure and table. Our runs used NVIDIA A100 GPUs; one CMA-ES fit
took 20–60 minutes and one PINN fit 5–8 hours.

## Documentation

| file | what it covers |
|---|---|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | every experiment, figure and table, with its command |
| [docs/DATA.md](docs/DATA.md) | where the data come from and which patients each task needs |
| [docs/REFERENCE_PINN.md](docs/REFERENCE_PINN.md) | how we run the PINN of Zhang et al. |
| [docs/PINN_SETTINGS.md](docs/PINN_SETTINGS.md) | the PINN's training settings in Zhang et al.'s paper, in their code and in our runs |
| [repro/one_case/README.md](repro/one_case/README.md) | the worked example and what it checks |

## Reproducibility

* **Random seeds.** Every run record stores its random seeds and its full settings. Our main CMA-ES
  runs used random seed 0, which `pycma` replaces with a seed drawn from the clock, so they cannot be
  repeated bit for bit. Rerun with random seed 1, they give the same result (median absolute relative
  error of μ_D 0.421 against 0.420).
* **GPU arithmetic.** The PINN is not bit-reproducible on a GPU, because the order of summation varies,
  so a rerun can differ slightly from our run records.
* **The PINN code is not modified.** We set options its release provides, add the early stopping of the
  authors' original TensorFlow code, and apply two small runtime patches; all are listed in
  [docs/REFERENCE_PINN.md](docs/REFERENCE_PINN.md).

## Repository layout

```
src/              forward solver, objective, CMA-ES calibration, data loading, metrics, synthetic tumors
experiments/      run_cmaes.py, run_pinn.py, PINN add-ons, Slurm job templates, patient lists
configs/          settings of each experiment
scripts/          table and figure scripts, download scripts
outputs/tables/   result tables
tests/            tests
repro/one_case/   worked example
docs/             documentation
```

## License

MIT, see [LICENSE](LICENSE). The PINN code, the PREDICT-GBM data and `pycma` come with their own terms.

## Citation

If you use this code, please cite our paper ([CITATION.cff](CITATION.cff)), together with
[Zhang et al. (2025)](https://doi.org/10.1016/j.media.2024.103423) for the PINN and
[Zimmer et al. (2026)](https://doi.org/10.1038/s41746-026-03194-0) for the PREDICT-GBM benchmark.
The run records are cited as: Moini, K. (2026). Run records, version 1.0 [Dataset]. Zenodo.
<https://doi.org/10.5281/zenodo.24107357>
