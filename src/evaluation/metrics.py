#!/usr/bin/env python
"""Metrics that judge a calibrated model, for both methods alike.

Three families, matching the evaluation plan of our paper:

* **Parameter recovery** on the synthetic cases, where the truth is known. This
  is the primary outcome. Absolute D and rho are never compared, because a
  single imaging time point does not identify them; the comparison uses the
  products D*t_end and rho*t_end and the infiltration length lambda.
* **Fit to the pre-operative masks**, as Dice of the thresholded density. This
  is in-sample: the same masks drive the calibration, so it is a sanity check,
  not validation.
* **Recurrence coverage**, the radiotherapy-planning read-out of
  Zimmer et al. (2026). The personalized target is built iso-volumetric to the
  standard plan (the tumor core dilated by 15 mm inside the brain mask), so
  the two plans irradiate the same volume and only their shape differs.

Label conventions follow the PREDICT-GBM derivatives: 1 necrotic, 2 edema,
3 enhancing, and in the recurrence map 4 resection cavity. Whole tumor is
{1,2,3} and tumor core is {1,3}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = [
    "dice",
    "segmentation_dice",
    "standard_plan",
    "isovolumetric_plan",
    "coverage_metrics",
    "parameter_errors",
    "mu_errors",
    "ParameterTruth",
]

LABEL_NECROTIC = 1
LABEL_EDEMA = 2
LABEL_ENHANCING = 3
LABEL_CAVITY = 4


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two binary masks; 1.0 when both are empty."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    total = int(a.sum() + b.sum())
    if total == 0:
        return 1.0
    return float(2.0 * np.count_nonzero(a & b) / total)


def segmentation_dice(
    density: np.ndarray,
    segmentation: np.ndarray,
    *,
    th_wt: float,
    th_tc: float,
) -> dict[str, float]:
    """Dice of the thresholded density against the whole tumor and the core."""
    gt_wt = np.isin(segmentation, [LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING])
    gt_tc = np.isin(segmentation, [LABEL_NECROTIC, LABEL_ENHANCING])
    return {
        "dice_wt": dice(density > th_wt, gt_wt),
        "dice_tc": dice(density > th_tc, gt_tc),
    }


def standard_plan(
    segmentation: np.ndarray,
    brain_mask: np.ndarray,
    *,
    margin_mm: float = 15.0,
    voxel_mm: float = 1.0,
) -> np.ndarray:
    """Guideline target: the tumor core dilated by ``margin_mm``, inside the brain.

    The dilation is isotropic and uses a Euclidean distance transform, so the
    margin is measured in millimeters rather than in voxel neighborhoods.
    """
    if margin_mm < 0:
        raise ValueError(f"margin_mm must be non-negative, got {margin_mm}")
    core = np.isin(segmentation, [LABEL_NECROTIC, LABEL_ENHANCING])
    if not core.any():
        raise ValueError("the tumor core is empty, so no standard plan can be built")
    distance = distance_transform_edt(~core) * voxel_mm
    return ((distance <= margin_mm) & (brain_mask > 0)).astype(bool)


def _threshold_for_volume(
    density: np.ndarray,
    target_voxels: float,
    *,
    max_iter: int = 100,
    tol: float = 0.01,
) -> float:
    """Density threshold whose super-level set has ``target_voxels`` voxels."""
    lo, hi = 0.001, 1.0
    if np.count_nonzero(density > lo) < target_voxels:
        return lo
    if np.count_nonzero(density > hi) > target_voxels:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        volume = np.count_nonzero(density > mid)
        if abs(volume - target_voxels) / (target_voxels + 1e-6) < tol:
            return float(mid)
        if volume > target_voxels:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return float(0.5 * (lo + hi))


def isovolumetric_plan(
    density: np.ndarray,
    brain_mask: np.ndarray,
    target_voxels: float,
) -> tuple[np.ndarray, float]:
    """Personalized target of the same volume as the standard plan.

    Iso-volumetry is the whole point of the protocol: a plan that may grow is
    trivially better at covering a recurrence, so both plans are forced to the
    same size and the model can only redistribute (Zimmer et al. 2026, p. 11).
    Two details follow from that.

    The threshold is searched on the density *inside the brain*, because voxels
    outside it are removed afterwards and would otherwise be counted toward the
    target, leaving the plan short.

    If the density still yields too few voxels -- everything above the search
    floor fits inside the target -- the plan is grown outward instead of being
    left undersized. Zimmer et al. binarize the prediction, take a distance
    transform, and add voxels in order of increasing distance until the volume
    matches (p. 11); that is what happens here.

    Returns the binary plan and the threshold that produced it.
    """
    inside = np.where(brain_mask > 0, density, 0.0)
    threshold = _threshold_for_volume(inside, target_voxels)
    plan = (inside > threshold) & (brain_mask > 0)

    missing = int(round(target_voxels)) - int(plan.sum())
    if missing > 0 and plan.any():
        distance = distance_transform_edt(~plan)
        candidates = (~plan) & (brain_mask > 0)
        order = np.argsort(distance[candidates], kind="stable")[:missing]
        indices = np.flatnonzero(candidates)[order]
        grown = plan.reshape(-1).copy()
        grown[indices] = True
        plan = grown.reshape(plan.shape)

    return plan.astype(bool), threshold


def coverage_metrics(
    density: np.ndarray,
    segmentation: np.ndarray,
    recurrence: np.ndarray,
    brain_mask: np.ndarray,
    *,
    margin_mm: float = 15.0,
    voxel_mm: float = 1.0,
) -> dict[str, float]:
    """Coverage of the recurrence by the personalized and the standard plan.

    Two recurrence definitions are reported, as in Zimmer et al. (2026):
    the enhancing recurrence (label 3) and the recurrence core (labels 1 and 3).

    Returns:
        Fractions in [0, 1], the plan volumes in voxels, and the threshold.
    """
    std = standard_plan(segmentation, brain_mask, margin_mm=margin_mm, voxel_mm=voxel_mm)
    target_voxels = float(std.sum())
    personal, threshold = isovolumetric_plan(density, brain_mask, target_voxels)

    out: dict[str, float] = {
        "standard_plan_voxels": target_voxels,
        "personal_plan_voxels": float(personal.sum()),
        "plan_threshold": float(threshold),
    }

    targets = {
        "enhancing": np.isin(recurrence, [LABEL_ENHANCING]),
        "core": np.isin(recurrence, [LABEL_NECROTIC, LABEL_ENHANCING]),
    }
    for name, target in targets.items():
        volume = float(target.sum())
        if volume == 0.0:
            out[f"coverage_standard_{name}"] = float("nan")
            out[f"coverage_personal_{name}"] = float("nan")
            continue
        out[f"coverage_standard_{name}"] = float(np.count_nonzero(target & std) / volume)
        out[f"coverage_personal_{name}"] = float(np.count_nonzero(target & personal) / volume)
    return out


@dataclass
class ParameterTruth:
    """Ground-truth parameters of one synthetic case.

    ``d_eff`` and ``rho_eff`` are the identifiable products D*t_end (mm^2) and
    rho*t_end (dimensionless), not per-day rates.
    """

    d_eff: float
    rho_eff: float
    th_wt: float
    th_tc: float

    @property
    def lam(self) -> float:
        """Infiltration length ``sqrt(d_eff / rho_eff) = sqrt(D / rho)`` in mm."""
        return float(np.sqrt(self.d_eff / self.rho_eff))


def mu_errors(
    mu_d: float,
    mu_rho: float,
    mu_d_true: float,
    mu_rho_true: float,
) -> dict[str, float | bool]:
    """Compare the estimate with the truth *in the coordinates both methods search*.

    Every method starts from ``mu_D = mu_rho = 1``, the point the shared
    characteristic grid search produces, and the prior (a penalty that pulls the
    four parameters toward their start values) charges for leaving it. The relative errors on ``D t`` and ``rho t`` cannot separate a
    method that searched well from one that never moved, because both are read
    off the same starting point. The distance from the start is therefore
    reported next to the distance from the truth: if the first is near zero the
    run says nothing about the method, whatever its other numbers look like.
    """
    start_d = abs(1.0 - mu_d_true)
    start_rho = abs(1.0 - mu_rho_true)
    err_d = abs(mu_d - mu_d_true)
    err_rho = abs(mu_rho - mu_rho_true)
    return {
        "mu_d_true": float(mu_d_true),
        "mu_rho_true": float(mu_rho_true),
        "err_mu_d": float(err_d),
        "err_mu_rho": float(err_rho),
        "err_mu_d_at_start": float(start_d),
        "err_mu_rho_at_start": float(start_rho),
        "moved_mu_d": float(abs(mu_d - 1.0)),
        "moved_mu_rho": float(abs(mu_rho - 1.0)),
        "improved_on_start": bool(err_d + err_rho < start_d + start_rho),
    }

def parameter_errors(
    estimate: ParameterTruth,
    truth: ParameterTruth,
) -> dict[str, float]:
    """Relative errors of the identifiable parameters, plus threshold errors.

    The front travel ``2 sqrt(d_eff * rho_eff) = 2 sqrt(D t_end * rho t_end) =
    v t_end``, in mm, is reported as well (``rel_err_speed``), with ``v = 2 sqrt(D
    rho)`` the wave speed. Like lambda, it does not depend on the time scale.
    """

    def rel(a: float, b: float) -> float:
        return float(abs(a - b) / abs(b)) if b != 0 else float("nan")

    speed_est = 2.0 * np.sqrt(estimate.d_eff * estimate.rho_eff)
    speed_true = 2.0 * np.sqrt(truth.d_eff * truth.rho_eff)
    return {
        "rel_err_d_eff": rel(estimate.d_eff, truth.d_eff),
        "rel_err_rho_eff": rel(estimate.rho_eff, truth.rho_eff),
        "rel_err_lambda": rel(estimate.lam, truth.lam),
        "rel_err_speed": rel(float(speed_est), float(speed_true)),
        "abs_err_th_wt": float(abs(estimate.th_wt - truth.th_wt)),
        "abs_err_th_tc": float(abs(estimate.th_tc - truth.th_tc)),
        "lambda_estimate_mm": estimate.lam,
        "lambda_truth_mm": truth.lam,
    }
