"""The PINN on the training settings printed in Zhang et al. (2025) (the paper-settings run).

Why this exists
---------------
The main PINN runs of our paper use the PyTorch port's own settings where
Zhang et al. (2025) print others (``docs/PINN_SETTINGS.md``). This module puts
the printed settings back **without editing the reference code**, the same way
``experiments/pinn_early_stopping`` and ``experiments/pinn_param_lr`` do. The problem
(cases, scoring objective, boxes, starts, tumor seed position at the mask centroid,
and the prior, a penalty pulling the four parameters toward their start values,
switched off) is unchanged; only how the PINN trains changes.

What it does, at the start of each stage (called from the early-stopping loop)
-------------------------------------------------------------------------------
* **Fixed collocation sets** (App. E, p. 14: "We take N_r = 20,000"; fine-tuning
  "N_r = N_dat = 5000"): one set per stage, drawn once, used at every step, instead
  of the PyTorch port's fresh random batch of 200,000 / 100,000 points per step.
* **Residual points sampled as App. F (p. 15) prints**: time from the truncated
  exponential density lambda e^(-lambda t) on [0, 1] with lambda = 0.5; space drawn
  with weight 1 / r^2 from the tumor center inside the PyTorch port's ball. The
  port's exp(-10 t) residual weight is switched off in the options (Zhang et al.
  sample, they do not weight). r is floored at half a voxel, so the center voxel
  has a finite weight; Zhang et al. do not say how they treat r = 0.
* **Data points** (fine-tuning): N_dat points drawn uniformly from the voxels of
  the ball, at t = 1 ("sampled uniformly at pixel resolution", p. 5).
* **The segmentation loss of Eq. (15)-(16), p. 5**: (1/N_dat) sum of
  (H(phi u - u_c) - y)^2 per mask, with H = sigmoid(20 (.)), phi INSIDE the
  Heaviside and no normalization; the PyTorch port instead weights the error by phi
  and divides by mean(y^2). With the printed weight 1e-3 on each mask term
  (the literal reading of Zhang et al.), this is their w_SEG L_SEG.
* **Box penalties for mu_D and mu_rho** (Eq. 18, weight 1) on the problem's own box
  [0, 2], added as the loss ``murange``; the thresholds' box penalty is the
  port's ``thrange``. With the open box they only act at its edges.

What it cannot reproduce, and says so in the record
----------------------------------------------------
* in Zhang et al., L_char uses the residual points; the PyTorch port stores the
  characteristic solution only at its time snapshots, so L_char keeps the
  port's space-time grid points (N_r of them, fixed);
* scipy's L-BFGS-B line search of 50 steps (torch's is fixed at 25);
* phi_tau = phi + tau in the flux (tau is not printed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

STEEPNESS = 20.0  # Eq. (15), a = 20; also hard-coded in GBMproblem.sigmoid_binarize


@dataclass(frozen=True)
class PaperSettings:
    res_points_init: int = 20000   # App. E, p. 14
    res_points_inv: int = 5000     # App. E, p. 14
    dat_points_inv: int = 5000     # App. E, p. 14
    time_lambda: float = 0.5       # App. F, p. 15
    r_floor_voxels: float = 0.5    # not printed: floor of r for the 1/r^2 weight
    seed: int = 0                  # random seed of the point sets

    @classmethod
    def from_config(cls, block: dict, seed: int = 0) -> "PaperSettings":
        known = {k: block[k] for k in cls.__dataclass_fields__ if k in block}
        known.setdefault("seed", seed)
        return cls(**known)


def truncated_exponential(n: int, lam: float, generator: torch.Generator, device) -> torch.Tensor:
    """n samples of t on [0, 1] with density proportional to exp(-lam t), sorted."""
    u = torch.rand(n, 1, generator=generator, device=device)
    t = -torch.log1p(-u * (1.0 - math.exp(-lam))) / lam
    return torch.sort(t, dim=0)[0]


def inverse_r2_indices(r: torch.Tensor, n: int, floor: float, generator: torch.Generator) -> torch.Tensor:
    """n indices drawn with replacement, probability proportional to 1 / max(r, floor)^2."""
    weight = 1.0 / torch.clamp(r, min=floor) ** 2
    return torch.multinomial(weight, n, replacement=True, generator=generator)


def paper_seg_loss(mask_key: str, threshold_key: str):
    """Eq. (16) for one mask: mean of (H(phi u(x, 1) - u_c) - y)^2 over the data points."""

    def loss(net, pde=None):
        data = pde.dataset.batch["dat"]
        upred = net(data["X_dat_train"])
        pred = torch.sigmoid(STEEPNESS * (data["phi_dat_train"] * upred - net.all_params_dict[threshold_key]))
        return torch.mean((pred - data[mask_key]) ** 2)

    return loss


def mu_range_penalty(pde):
    """Eq. (18) for mu_D and mu_rho on the problem's box (``rD_range``, ``rRHO_range``)."""

    def loss(net):
        relu = torch.nn.functional.relu
        total = 0.0
        for name, (lo, hi) in (("rD", pde.rD_range), ("rRHO", pde.rRHO_range)):
            value = net.all_params_dict[name].squeeze()
            total = total + relu(value - hi) ** 2 + relu(lo - value) ** 2
        return total

    return loss


def install_for_stage(trainer, settings: PaperSettings) -> dict:
    """Fix this stage's collocation sets and the loss forms of Zhang et al.; return a record."""
    pde = trainer.pde
    ds = pde.dataset
    X_space = ds["X_space_flat"]
    device = X_space.device
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + (0 if trainer.stage == "init" else 1))
    L = float(torch.as_tensor(ds["L"]).reshape(-1)[0])
    r = torch.sqrt(torch.sum(X_space ** 2, dim=1))
    floor = settings.r_floor_voxels / L  # coordinates are scaled by L (mm per unit)

    n_res = settings.res_points_init if trainer.stage == "init" else settings.res_points_inv
    idx = inverse_r2_indices(r, n_res, floor, generator)
    t = truncated_exponential(n_res, settings.time_lambda, generator, device)
    batch = dict(ds.batch)
    batch["res"] = {
        "X_res_train": torch.cat([t, X_space[idx]], dim=1),
        "phi_res_train": ds["phi_flat"][idx],
        "P_res_train": ds["P_flat"][idx],
        "gradPphi_res_train": ds["gradPphi_flat"][idx],
    }
    # The record key "r_floor_mm" holds the floor in voxels, not mm.
    record = {"n_res": n_res, "res_unique_voxels": int(torch.unique(idx).numel()),
              "time_lambda": settings.time_lambda, "time_mean": float(t.mean()),
              "r_floor_mm": settings.r_floor_voxels, "space_weight": "1/r^2"}

    if trainer.stage == "init" and "X_st" in ds:
        n_st = int(ds["X_st"].shape[0])
        pick = torch.randint(0, n_st, (n_res,), generator=generator, device=device)
        st = {"X_st": ds["X_st"][pick], "phi_st": ds["phi_st"][pick]}
        if ds.get("uchar_st", None) is not None:
            st["uchar_st"] = ds["uchar_st"][pick]
        batch["st"] = st
        record["n_char"] = n_res
    if trainer.stage == "inv":
        n_space = int(X_space.shape[0])
        n_dat = min(settings.dat_points_inv, n_space)
        pick = torch.randperm(n_space, generator=generator, device=device)[:n_dat]
        batch["dat"] = {
            "X_dat_train": torch.cat([torch.ones(n_dat, 1, device=device), X_space[pick]], dim=1),
            "phi_dat_train": ds["phi_flat"][pick],
            "u1_dat_train": ds["u1_flat"][pick],
            "u2_dat_train": ds["u2_flat"][pick],
        }
        seg1, seg2 = paper_seg_loss("u1_dat_train", "th1"), paper_seg_loss("u2_dat_train", "th2")
        pde.loss_dict["seg1"] = lambda net: seg1(net, pde)
        pde.loss_dict["seg2"] = lambda net: seg2(net, pde)
        record.update({"n_dat": n_dat, "data_wt_fraction": float(ds["u1_flat"][pick].float().mean()),
                       "seg_form": "Eq. (15)-(16): mean (H(phi u - u_c) - y)^2, unnormalised"})
    pde.loss_dict["murange"] = mu_range_penalty(pde)

    ds.batch = batch
    ds.next_batch = lambda: None  # the set stays fixed for the whole stage
    record["fixed_sets"] = True
    return record
