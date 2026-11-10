"""The L-BFGS stage (experiments/pinn_lbfgs.py) on a small problem with a known answer."""

from types import SimpleNamespace

import torch

from experiments.pinn_early_stopping import EarlyStopRule
from experiments.pinn_lbfgs import SCIPY_FTOL, LbfgsSettings, run_lbfgs


def toy_trainer(seed=0):
    """A linear least-squares 'PINN': the exact minimizer is known."""
    torch.manual_seed(seed)
    X = torch.randn(256, 3)
    y = X @ torch.tensor([[1.5], [-2.0], [0.5]]) + 0.3
    net = torch.nn.Linear(3, 1)
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)  # like rD in pre-training
    net.register_parameter("frozen", frozen)
    pde = SimpleNamespace(dataset=SimpleNamespace(batch={"res": {"X": X, "y": y}}))

    def wloss(keys, grad):
        batch = pde.dataset.batch["res"]
        loss = torch.mean((net(batch["X"]) - batch["y"]) ** 2)
        return loss, {"res": loss}, {"res": loss}

    trainer = SimpleNamespace(pde=pde, net=net, stage="init", loss=["res"],
                              get_wloss_sum_comp=wloss,
                              optimizer=SimpleNamespace(zero_grad=lambda: None))
    return trainer


def test_defaults_are_the_tensorflow_values():
    s = LbfgsSettings()
    assert (s.max_iter, s.max_fun, s.history_size, s.tolerance_grad) == (10000, 10000, 100, 0.0)
    assert s.tolerance_change == SCIPY_FTOL
    assert s.stages == ("init",)


def test_solves_least_squares_and_records_why_it_stopped():
    trainer = toy_trainer()
    batch = trainer.pde.dataset.batch
    live = {"res": dict(batch["res"])}
    trainer.pde.dataset.batch = live
    record = run_lbfgs(trainer, batch, LbfgsSettings(), EarlyStopRule(check_every=10))
    assert record["loss_end"] < 1e-8 < record["loss_start"]
    assert record["iterations"] > 0 and record["function_evals"] >= record["iterations"]
    assert record["stop_reason"]
    assert trainer.pde.dataset.batch is live  # the live batch is put back
    assert not trainer.net.frozen.requires_grad and float(trainer.net.frozen) == 1.0


def test_iteration_budget_is_respected():
    trainer = toy_trainer()
    batch = trainer.pde.dataset.batch
    record = run_lbfgs(trainer, batch, LbfgsSettings(max_iter=10, tolerance_change=0.0),
                       EarlyStopRule(check_every=10))
    assert record["iterations"] <= 10


def mlp_trainer(seed=0):
    """A small nonlinear fit whose line searches need many loss evaluations.

    The linear problem above converges in a handful of evaluations, so it cannot show
    whether the stage stops too early. This one can: with torch's default max_eval
    (12 evaluations per step() call), step() would return early and the stage would
    report convergence after about 20 iterations while the loss is still falling.
    """
    torch.manual_seed(seed)
    X = torch.linspace(-3, 3, 256).unsqueeze(1)
    y = torch.sin(2 * X) + 0.3 * X
    net = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    pde = SimpleNamespace(dataset=SimpleNamespace(batch={"res": {"X": X, "y": y}}))

    def wloss(keys, grad):
        batch = pde.dataset.batch["res"]
        loss = torch.mean((net(batch["X"]) - batch["y"]) ** 2)
        return loss, {"res": loss}, {"res": loss}

    return SimpleNamespace(pde=pde, net=net, stage="init", loss=["res"],
                           get_wloss_sum_comp=wloss,
                           optimizer=SimpleNamespace(zero_grad=lambda: None))


def test_does_not_stop_early_on_a_hard_problem():
    trainer = mlp_trainer()
    batch = trainer.pde.dataset.batch
    record = run_lbfgs(trainer, batch, LbfgsSettings(max_iter=300, apply_patience=False),
                       EarlyStopRule(check_every=10))
    # An early return of step() would end the stage near 20 iterations and report convergence.
    assert record["iterations"] >= 200, record
    assert record["loss_end"] < 1e-2 * record["loss_start"], record


def test_patience_is_logged_not_applied():
    trainer = mlp_trainer(seed=1)
    batch = trainer.pde.dataset.batch
    rule = EarlyStopRule(check_every=10, patience=20, min_delta=1.0)  # fires at once
    record = run_lbfgs(trainer, batch, LbfgsSettings(max_iter=100, apply_patience=False), rule)
    assert record["patience_would_stop_at"] is not None
    assert record["iterations"] >= 50
