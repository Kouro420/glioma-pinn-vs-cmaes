# Data

## PREDICT-GBM (public)

Our paper uses the RHUH cohort of the PREDICT-GBM benchmark (Zimmer et al. 2026, *npj Digital Medicine*
9, doi:10.1038/s41746-026-03194-0). Platform: <https://github.com/BrainLesion/PredictGBM>. Data:
<https://huggingface.co/datasets/LZimmer/PREDICT-GBM>. Use the data under the benchmark's own terms;
this repository does not redistribute it.

### Which patients you need

| task | patients | download |
|---|---|---|
| the worked example (`repro/one_case/`) and every synthetic tumor | RHUH-0001 and RHUH-0002, the two brains the synthetic tumors are grown in | about 10.7 GB |
| the comparison on patients | RHUH-0001 to RHUH-0020 (the first 20 lines of `experiments/cases_rhuh.txt`) | nearly the whole archive, 19.5 GB |

### Download

The benchmark is one archive, `predict_gbm.tar.gz` (all cohorts), and its cases are stored in no
particular order, so single cases cannot be downloaded on their own. `scripts/fetch_predict_gbm.py` reads
the archive while it downloads, keeps only the files the code uses for each requested case (listed
below), and stops as soon as the last requested case is complete. It needs only the Python standard
library.

```bash
python scripts/fetch_predict_gbm.py              # RHUH-0001 and RHUH-0002
python scripts/fetch_predict_gbm.py --patients   # RHUH-0001 to RHUH-0020
```

The cases land in `src/data/predict_gbm/predict_gbm/<case>/`, the folder the configs and
`scripts/make_figures.py` expect; pass `--data-root src/data/predict_gbm/predict_gbm` to the run scripts.
A stopped download starts again from the beginning; cases that are already complete are skipped.

**The whole benchmark.** To keep all cohorts, download the archive itself (with a browser or
`curl -L -O https://huggingface.co/datasets/LZimmer/PREDICT-GBM/resolve/main/predict_gbm.tar.gz`) and
either unpack it (`tar -xzf predict_gbm.tar.gz -C src/data/predict_gbm`) or let the script take
the cases from it: `python scripts/fetch_predict_gbm.py --archive predict_gbm.tar.gz`.

### What each case holds

`tumor_seg.nii.gz` (tumor labels 1–3), `wm_pbmap.nii.gz`, `gm_pbmap.nii.gz`, `csf_pbmap.nii.gz` (tissue
maps), `t1c_bet_mask.nii.gz` (brain mask), `recurrence_preop.nii.gz` (follow-up recurrence, patient
evaluation only), and `t1c_bet_normalized.nii.gz`, `flair_bet_normalized.nii.gz` (images, figures only),
as the benchmark releases them. The loader finds a case by a folder named after its ID anywhere below
`--data-root`, so another folder layout also works.

## Synthetic cases (generated)

Built from the two public source brains with fixed random seeds ([REPRODUCE.md](REPRODUCE.md),
section A). They are regenerated rather than shipped.

## Clinical pilot (not shared)

Our paper's pilot uses 17 pre-operative scans from Evangelisches Klinikum Bethel, Bielefeld, under the
approval of the Ethics Committee of the Ärztekammer Westfalen-Lippe and the Westfälische
Wilhelms-Universität Münster (Az 2025-906-f-N). These are patient data. No image, mask, record or case
identifier of the pilot is in this repository or in the Zenodo archive; our paper reports it as a
summary. Requests for the data go to the clinical co-authors, not to this repository.

## Run records (Zenodo)

The run records of our paper (one JSON file per method and case) and the objective profiles are one
archive on Zenodo, <https://doi.org/10.5281/zenodo.24107357>. `python scripts/fetch_records.py`
downloads them into `outputs/`. Or download the archive by hand and run
`python scripts/fetch_records.py --archive <file>`. The records hold numbers only, no images: every
record is a synthetic case or a public RHUH case.
