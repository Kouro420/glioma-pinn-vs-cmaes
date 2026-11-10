#!/bin/bash
#SBATCH --job-name=glioma-fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# One calibration per array task: task i fits the case on line i+1 of the case list.
# Adapt the #SBATCH lines to your cluster (a PINN fit on the training schedule of Zhang et al.
# needs about 22 GB of GPU memory and up to 8 h, so pass --time=08:00:00 for it).
#
#   export DATA_ROOT=src/data/predict_gbm/predict_gbm
#   sbatch --array=0-19 experiments/slurm/gpu_job.sbatch cmaes experiments/cases_rhuh.txt --prior-scale 0.0
#   sbatch --array=0-15 --time=08:00:00 experiments/slurm/gpu_job.sbatch pinn <synthetic folder>/cases_synthetic.txt \
#       --synthetic --prior-scale 0.0 --config configs/paper_schedule.yaml
#
# DATA_ROOT is the folder that holds the cases; OUT_DIR (default outputs/runs) receives the run records.
# Activate your Python environment before sbatch, or add the activation line below.

set -euo pipefail

METHOD="${1:?first argument: cmaes or pinn}"
CASE_LIST="${2:?second argument: a file with one case ID per line}"
shift 2

CASE_ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CASE_LIST}" | tr -d '\r')"
: "${DATA_ROOT:?set DATA_ROOT to the folder that holds the cases}"
OUT_DIR="${OUT_DIR:-outputs/runs}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs "${OUT_DIR}"
python -m "experiments.run_${METHOD}" --case "${CASE_ID}" --data-root "${DATA_ROOT}" \
  --config configs/base.yaml --out "${OUT_DIR}" --save-density "$@"
