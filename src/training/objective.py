#!/usr/bin/env python
"""The score both calibration methods minimize.

Fairness of the PINN-versus-CMA-ES comparison rests on this file. The PINN
scores a candidate density against the MRI masks with a smoothed Heaviside
misfit plus parameter penalties (Zhang et al. 2025, Eqs. 15-16, and
``get_seg1_loss``, ``get_seg2_loss`` and ``range_constraint`` in
``external/pinngbm-torch/GBMproblem.py``, lines 279-312). The CMA-ES baseline
uses the identical expression, so that any difference in the results comes from
how the forward problem is solved and how the parameters are searched, not from
a different target.

Ways in which the reference code differs from the equations of Zhang et al.,
kept here because the PINN runs from the code and the two methods must share
one score:

* the code divides each mask term by ``mean(y^2)``; Eq. 16 does not,
* the code weights the squared error by the phase field ``phi``,
* the smoothed Heaviside uses steepness 20 (hard-coded), while the unused
  option ``ksigmoid = 50`` exists in ``Options.py``.

One term of the PINN loss has no counterpart here: the PDE residual. A finite
difference solution satisfies the discretized equation by construction, so that
term is identically zero for the baseline. Two further differences in what the
two arms optimize: the release's residual has d(phi u)/dt where the solver
advances du/dt, and the main PINN runs used the release's data weight 1e-5, not
the 1e-3 of Zhang et al. Our paper states all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ObjectiveWeights", "ObjectiveTerms", "SegmentationObjective"]


@dataclass
class ObjectiveWeights:
    """Weights of the shared score, mirroring ``weights_inv`` in the code."""

    seg_wt: float = 1e-5
    seg_tc: float = 1e-5
    reg_mu_d: float = 1e-3
    reg_mu_rho: float = 1e-3
    reg_th_wt: float = 1e-3
    reg_th_tc: float = 1e-3
    range_penalty: float = 1.0


@dataclass
class ObjectiveTerms:
    """Every term of one evaluation, kept for the progress curves."""

    total: float
    seg_wt: float
    seg_tc: float
    reg_mu_d: float
    reg_mu_rho: float
    reg_th_wt: float
    reg_th_tc: float
    range_penalty: float
    # The two halves of the score, already multiplied by their weights. The raw
    # terms above cannot be compared with each other, because the segmentation
    # weights are 1e-5 and the prior weights are 1e-3, so every history entry
    # also stores the weighted data part and the weighted prior part.
    data_weighted: float = 0.0
    prior_weighted: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {
            "total": self.total,
            "seg_wt": self.seg_wt,
            "seg_tc": self.seg_tc,
            "reg_mu_d": self.reg_mu_d,
            "reg_mu_rho": self.reg_mu_rho,
            "reg_th_wt": self.reg_th_wt,
            "reg_th_tc": self.reg_th_tc,
            "range_penalty": self.range_penalty,
            "data_weighted": self.data_weighted,
            "prior_weighted": self.prior_weighted,
        }
        out.update(self.extras)
        return out


def _smooth_heaviside(u: np.ndarray, threshold: float, steepness: float) -> np.ndarray:
    """Sigmoid approximation of ``1{u > threshold}``."""
    return 1.0 / (1.0 + np.exp(-steepness * (u - threshold)))


def _relu_sq(x: float, lo: float, hi: float) -> float:
    """Squared distance outside the interval ``[lo, hi]``."""
    return max(x - hi, 0.0) ** 2 + max(lo - x, 0.0) ** 2


class SegmentationObjective:
    """Score of a predicted density against the two MRI masks.

    Args:
        mask_wt, mask_tc: binary whole-tumor and tumor-core masks.
        phi: phase field, used as the spatial weight of the misfit.
        prior_th_wt, prior_th_tc: start values of the two thresholds. The prior
            (a penalty that pulls the four parameters toward their start values)
            pulls toward these and toward mu = 1.
        weights: term weights; defaults reproduce the reference configuration.
        steepness: steepness of the smoothed Heaviside (20 in the reference).
        mu_range, th_range: admissible intervals for the four parameters.
        mu_rho_range, th_tc_range: separate intervals for mu_rho and th_tc (the
            box of Zhang et al. gives u_FLAIR and u_T1c different ranges). None, the
            default, uses mu_range and th_range.
    """

    def __init__(
        self,
        mask_wt: np.ndarray,
        mask_tc: np.ndarray,
        phi: np.ndarray,
        *,
        prior_th_wt: float,
        prior_th_tc: float,
        weights: ObjectiveWeights | None = None,
        steepness: float = 20.0,
        mu_range: tuple[float, float] = (0.0, 2.0),
        th_range: tuple[float, float] = (0.0, 1.0),
        mu_rho_range: tuple[float, float] | None = None,
        th_tc_range: tuple[float, float] | None = None,
    ) -> None:
        if not (mask_wt.shape == mask_tc.shape == phi.shape):
            raise ValueError("masks and phase field must have the same shape")

        self.mask_wt = mask_wt.astype(np.float32)
        self.mask_tc = mask_tc.astype(np.float32)
        self.phi = phi.astype(np.float32)
        self.prior_th_wt = float(prior_th_wt)
        self.prior_th_tc = float(prior_th_tc)
        self.weights = weights or ObjectiveWeights()
        self.steepness = float(steepness)
        self.mu_range = mu_range
        self.th_range = th_range
        self.mu_rho_range = mu_range if mu_rho_range is None else mu_rho_range
        self.th_tc_range = th_range if th_tc_range is None else th_tc_range

        self._norm_wt = float(np.mean(self.mask_wt**2))
        self._norm_tc = float(np.mean(self.mask_tc**2))
        if self._norm_wt <= 0.0 or self._norm_tc <= 0.0:
            raise ValueError("a mask is empty, so its normalization would divide by zero")

    def _mask_term(self, u: np.ndarray, mask: np.ndarray, threshold: float, norm: float) -> float:
        pred = _smooth_heaviside(u, threshold, self.steepness)
        return float(np.mean(((pred - mask) * self.phi) ** 2) / norm)

    def __call__(
        self,
        u_final: np.ndarray,
        *,
        mu_d: float,
        mu_rho: float,
        th_wt: float,
        th_tc: float,
    ) -> ObjectiveTerms:
        """Score one candidate.

        Args:
            u_final: predicted density at the imaging time.
            mu_d, mu_rho: dimensionless diffusion and proliferation factors.
            th_wt, th_tc: the two density thresholds.

        Returns:
            Every term, and their weighted sum in ``total``.
        """
        w = self.weights
        seg_wt = self._mask_term(u_final, self.mask_wt, th_wt, self._norm_wt)
        seg_tc = self._mask_term(u_final, self.mask_tc, th_tc, self._norm_tc)

        reg_mu_d = (mu_d - 1.0) ** 2
        reg_mu_rho = (mu_rho - 1.0) ** 2
        reg_th_wt = (th_wt - self.prior_th_wt) ** 2
        reg_th_tc = (th_tc - self.prior_th_tc) ** 2

        out_of_range = (
            _relu_sq(mu_d, *self.mu_range)
            + _relu_sq(mu_rho, *self.mu_rho_range)
            + _relu_sq(th_wt, *self.th_range)
            + _relu_sq(th_tc, *self.th_tc_range)
        )

        data_weighted = w.seg_wt * seg_wt + w.seg_tc * seg_tc
        prior_weighted = (
            w.reg_mu_d * reg_mu_d
            + w.reg_mu_rho * reg_mu_rho
            + w.reg_th_wt * reg_th_wt
            + w.reg_th_tc * reg_th_tc
        )
        total = data_weighted + prior_weighted + w.range_penalty * out_of_range

        return ObjectiveTerms(
            total=float(total),
            seg_wt=seg_wt,
            seg_tc=seg_tc,
            reg_mu_d=float(reg_mu_d),
            reg_mu_rho=float(reg_mu_rho),
            reg_th_wt=float(reg_th_wt),
            reg_th_tc=float(reg_th_tc),
            range_penalty=float(out_of_range),
            data_weighted=float(data_weighted),
            prior_weighted=float(prior_weighted),
        )
