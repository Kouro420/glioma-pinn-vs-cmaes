"""The early-stopping rule behaves like the TensorFlow original of Zhang et al. (2025)
(pinn.py, lines 59-100, in https://github.com/Rayzhangzirui/pinngbm)."""

from types import SimpleNamespace

import pytest
import torch

from experiments.pinn_early_stopping import EarlyStopRule, StageStopper, _monitor_value


class FakeProblem:
    """Stores its residual on itself, exactly as GBMproblem.resloss does."""

    def __init__(self, device, n):
        self.dataset = SimpleNamespace(batch={"res": {"X": torch.randn(n, 4, device=device)}})
        self.w = torch.randn(4, 256, device=device, requires_grad=True)
        self.res = torch.zeros(1, device=device)  # what training left behind

    def resloss(self):
        X = self.dataset.batch["res"]["X"]
        hidden = torch.tanh(X @ self.w)
        self.res = torch.tanh(hidden @ hidden.T[:, :64])  # keeps the graph alive
        self.extra = hidden  # a NEW attribute, also holding the graph
        return torch.mean(self.res ** 2)


def fake_trainer(device, n=512):
    pde = FakeProblem(device, n)
    trainer = SimpleNamespace(pde=pde, stage="init", loss=["res"],
                              optimizer=SimpleNamespace(zero_grad=lambda: None))
    def wloss(keys, grad):
        # Like Trainer.get_wloss_sum_comp: both dictionaries carry the graph too.
        loss = pde.resloss()
        return loss, {"res": 1.0 * loss}, {"res": loss}
    trainer.get_wloss_sum_comp = wloss
    return trainer


def test_check_restores_problem_state():
    trainer = fake_trainer("cpu")
    left_by_training = trainer.pde.res
    value = _monitor_value(trainer, trainer.pde.dataset.batch)
    assert value > 0
    assert trainer.pde.res is left_by_training
    assert not hasattr(trainer.pde, "extra")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_check_leaves_no_gpu_memory_behind():
    trainer = fake_trainer("cuda", n=4096)
    # Warm up first: the first matrix product allocates the cuBLAS workspace,
    # 4096 KiB x 2 + 16 KiB x 8 = 8,519,680 bytes, once per process. Training
    # does that before the first check, so it does not count against the check.
    _monitor_value(trainer, trainer.pde.dataset.batch)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _monitor_value(trainer, trainer.pde.dataset.batch)
    delta = torch.cuda.memory_allocated() - before
    assert abs(delta) < 1_000_000, f"check left {delta} bytes allocated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_without_restore_the_graph_stays():
    """Control for the test above: a residual evaluation that restores nothing leaves the graph,
    and so GPU memory, behind. This shows that the memory test can detect a leak."""
    trainer = fake_trainer("cuda", n=4096)
    _monitor_value(trainer, trainer.pde.dataset.batch)  # warm up
    before = torch.cuda.memory_allocated()
    trainer.pde.resloss()  # a residual evaluation with nothing restored
    assert torch.cuda.memory_allocated() - before > 1_000_000


def run(values, rule, mu_d=None):
    """Feed one value per step; return (steps_done at stop or None, stopper)."""
    stopper = StageStopper(rule)
    for steps_done, value in enumerate(values, start=1):
        if stopper.due(steps_done) and stopper.update(steps_done, value, mu_d):
            return steps_done, stopper
    return None, stopper


def test_no_check_during_burnin():
    rule = EarlyStopRule(patience=10, burnin=50, check_every=1)
    stop, stopper = run([1.0] * 49, rule)
    assert stop is None and stopper.checks == []


def test_flat_loss_stops_after_patience():
    rule = EarlyStopRule(patience=100, burnin=100, check_every=10)
    stop, stopper = run([1.0] * 1000, rule)
    # First due check (step 100) sets the best; 10 more checks of no progress = 100 steps.
    assert stop == 200
    assert stopper.best_step == 100
    assert "no improvement" in stopper.reason


def test_falling_loss_never_stops():
    rule = EarlyStopRule(patience=100, burnin=100, check_every=10)
    stop, _ = run([1.0 / k for k in range(1, 2001)], rule)
    assert stop is None


def test_improvement_smaller_than_min_delta_does_not_count():
    rule = EarlyStopRule(patience=100, min_delta=1e-3, burnin=10, check_every=10)
    values = [1.0 - 1e-5 * k for k in range(1, 1001)]  # falls 1e-4 per check
    stop, _ = run(values, rule)
    assert stop == 110


def test_mu_d_below_floor_stops_at_first_check():
    rule = EarlyStopRule(patience=100, burnin=20, check_every=10, min_mu_d=0.05)
    stop, stopper = run([1.0 / k for k in range(1, 200)], rule, mu_d=0.04)
    assert stop == 20
    assert "mu_D" in stopper.reason


def test_defaults_are_the_tensorflow_values():
    rule = EarlyStopRule()
    assert (rule.patience, rule.min_delta, rule.burnin, rule.min_mu_d) == (1000, 1e-6, 1000, 0.05)


def test_from_config_ignores_unknown_keys():
    rule = EarlyStopRule.from_config({"patience": 5, "comment": "x"})
    assert rule.patience == 5 and rule.burnin == 1000
