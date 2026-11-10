"""The training settings printed in Zhang et al. (2025) (the paper-settings run), without the reference code."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.pinn_paper_settings import (  # noqa: E402
    STEEPNESS,
    inverse_r2_indices,
    mu_range_penalty,
    paper_seg_loss,
    truncated_exponential,
)


def test_truncated_exponential_is_on_unit_interval_with_the_right_mean():
    g = torch.Generator().manual_seed(0)
    t = truncated_exponential(200000, 0.5, g, "cpu")
    lam = 0.5
    expected = 1 / lam - math.exp(-lam) / (1 - math.exp(-lam))  # mean of the truncated density
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    assert abs(float(t.mean()) - expected) < 3e-3
    assert torch.all(t[1:] >= t[:-1])  # sorted, as the release's residual batches are


def test_inverse_r2_weight_prefers_the_centre():
    g = torch.Generator().manual_seed(0)
    r = torch.tensor([0.0, 1.0, 2.0])
    idx = inverse_r2_indices(r, 120000, floor=1.0, generator=g)
    counts = torch.bincount(idx, minlength=3).float() / idx.numel()
    # weights 1, 1, 1/4 -> probabilities 4/9, 4/9, 1/9
    assert abs(float(counts[2]) - 1 / 9) < 0.01
    assert abs(float(counts[0]) - 4 / 9) < 0.01


def test_seg_loss_is_eq_16_with_phi_inside_the_heaviside():
    # Eq. (16) of Zhang et al. (2025).
    X = torch.zeros(4, 4)
    data = {"X_dat_train": X, "phi_dat_train": torch.tensor([[1.0], [0.5], [1.0], [0.0]]),
            "u1_dat_train": torch.tensor([[1.0], [0.0], [0.0], [0.0]])}
    pde = SimpleNamespace(dataset=SimpleNamespace(batch={"dat": data}))
    net = lambda x: torch.full((x.shape[0], 1), 0.6)  # noqa: E731
    net.all_params_dict = {"th1": torch.tensor(0.35)}
    loss = paper_seg_loss("u1_dat_train", "th1")(net, pde)
    pred = torch.sigmoid(STEEPNESS * (data["phi_dat_train"] * 0.6 - 0.35))
    expected = torch.mean((pred - data["u1_dat_train"]) ** 2)  # no normalization by mean(y^2)
    assert torch.isclose(loss, expected)


def test_box_penalty_is_zero_inside_and_quadratic_outside():
    pde = SimpleNamespace(rD_range=[0.0, 2.0], rRHO_range=[0.0, 2.0])
    net = SimpleNamespace(all_params_dict={"rD": torch.tensor(2.5), "rRHO": torch.tensor(1.0)})
    assert math.isclose(float(mu_range_penalty(pde)(net)), 0.25)


@pytest.mark.skipif(not (REPO / "external" / "pinngbm-torch" / "Options.py").is_file(),
                    reason="the reference PINN clone is missing (scripts/fetch_reference_pinn.sh)")
def test_b3_config_maps_to_the_printed_settings():
    from experiments.run_pinn import _build_options  # imports the reference Options

    config = yaml.safe_load((REPO / "configs" / "paper_settings_b3.yaml").read_text(encoding="utf-8"))
    paths = SimpleNamespace(tumor_seg="s", wm="w", gm="g", csf="c", brain_mask=None, recurrence=None)
    opts = _build_options(config, paths, "cpu", prior_scale=0.0).opts
    w = opts["train_opts"]["weights_inv"]
    assert w["seg1"] == 1e-3 and w["seg2"] == 1e-3 and w["res"] == 1.0
    assert w["thrange"] == 1.0 and w["murange"] == 1.0
    assert all(w[k] == 0.0 for k in ("rD_reg", "rRHO_reg", "th1_reg", "th2_reg"))
    assert opts["train_opts"]["weights_init"] == {"res": 1.0, "uchar_res": 1.0}
    assert "murange" in opts["train_opts"]["loss_inv"]
    assert opts["train_opts"]["optim_opts"] == {"amsgrad": False}
    assert opts["nn_opts"]["arch"] == "mlp" and opts["nn_opts"]["fourier"] is False
    assert opts["pde_opts"]["texp_weight"] == 0.0 and opts["pde_opts"]["force_bc"] is False
    assert opts["train_opts"]["lr_init"] == 1e-3 and opts["train_opts"]["lr_inv"] == 1e-4
    assert opts["train_opts"]["iter_init"] == 20000 and opts["train_opts"]["iter_inv"] == 5000
