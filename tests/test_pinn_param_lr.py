"""Own Adam rate for rD and rRHO (experiments/pinn_param_lr.py; the reach test)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from experiments.pinn_early_stopping import EarlyStopRule, install
from experiments.pinn_param_lr import ParamLrSettings, split_param_group

REPO = Path(__file__).resolve().parents[1]


def model():
    """A network plus rD and rRHO, in one Adam (AMSGrad) group as in Trainer.py:34."""
    torch.manual_seed(0)
    net = torch.nn.Linear(3, 1)
    net.all_params_dict = torch.nn.ParameterDict(
        {"rD": torch.nn.Parameter(torch.tensor(1.0)), "rRHO": torch.nn.Parameter(torch.tensor(1.0))})
    net.trainable_param = ["rD", "rRHO"]
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    return net, optimizer, scheduler


def loss_of(net, X, y):
    # A constant-sign gradient on rD and rRHO, so Adam moves them by lr per step.
    return torch.mean((net(X) - y) ** 2) + 5.0 * net.all_params_dict["rD"] \
        + 5.0 * net.all_params_dict["rRHO"]


def pretrain(net, optimizer, scheduler, steps=20):
    """Pre-training: rD and rRHO frozen, so Adam keeps no state for them."""
    X, y = torch.randn(64, 3), torch.randn(64, 1)
    for p in net.all_params_dict.values():
        p.requires_grad = False
    for _ in range(steps):
        optimizer.zero_grad()
        torch.mean((net(X) - y) ** 2).backward()
        optimizer.step()
        scheduler.step()
    for p in net.all_params_dict.values():
        p.requires_grad = True
    for group in optimizer.param_groups:  # Engine.configure_stage -> set_lr(lr_inv)
        group["lr"] = 1e-4
    return X, y


def test_split_keeps_adam_state_and_settings():
    net, optimizer, scheduler = model()
    pretrain(net, optimizer, scheduler)
    before = optimizer.state[net.weight]["exp_avg"].clone()
    groups = split_param_group(optimizer, dict(net.all_params_dict.items()), 1e-3, scheduler)
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    assert groups[1]["params"] == ["rD", "rRHO"]
    assert all(g["amsgrad"] for g in optimizer.param_groups)
    assert optimizer.param_groups[0]["betas"] == optimizer.param_groups[1]["betas"]
    # The network's moments from pre-training are untouched.
    assert torch.equal(optimizer.state[net.weight]["exp_avg"], before)


@pytest.mark.parametrize("split", [False, True])
def test_parameter_travels_at_its_own_rate(split):
    net, optimizer, scheduler = model()
    X, y = pretrain(net, optimizer, scheduler)
    if split:
        split_param_group(optimizer, dict(net.all_params_dict.items()), 1e-3, scheduler)
    weight0 = net.weight.detach().clone()
    steps = 200
    for _ in range(steps):
        optimizer.zero_grad()
        loss_of(net, X, y).backward()
        optimizer.step()
        scheduler.step()
    rate = 1e-3 if split else 1e-4
    travel = 1.0 - float(net.all_params_dict["rD"].detach())
    assert 0.9 * steps * rate <= travel <= 1.01 * steps * rate, travel
    # The network keeps 1e-4 in both cases: no weight moves more than 200 x 1e-4.
    assert float((net.weight.detach() - weight0).abs().max()) <= 1.01 * steps * 1e-4
    # The scheduler (gamma 1.0) has not changed either rate.
    assert sorted(g["lr"] for g in optimizer.param_groups) == sorted({1e-4, rate})


def test_missing_parameter_raises():
    net, optimizer, _ = model()
    stranger = torch.nn.Parameter(torch.tensor(0.5))
    with pytest.raises(ValueError, match="not in the optimiser"):
        split_param_group(optimizer, {"th1": stranger}, 1e-3)


def test_settings_from_config():
    s = ParamLrSettings.from_config({"params": ["rD", "rRHO"], "lr": "3e-4"})
    assert s == ParamLrSettings(params=("rD", "rRHO"), lr=3e-4, stages=("inv",))


class FakeTrainer:
    """The attributes the replacement train_loop of pinn_early_stopping uses."""

    def __init__(self, stage, num_iter):
        self.net, self.optimizer, self.scheduler = model()
        self.X, self.y = pretrain(self.net, self.optimizer, self.scheduler)
        self.stage, self.start_step, self.num_iter = stage, 0, num_iter
        self.loss = ["seg1", "seg2"]
        self.pde = SimpleNamespace(dataset=SimpleNamespace(
            batch={"dat": {"X": self.X}}, next_batch=lambda: None))

    def get_wloss_sum_comp(self, keys, grad):
        total = loss_of(self.net, self.X, self.y)
        return total, {"seg1": total}, {"seg1": total, "seg2": total * 0}

    def log_stat(self, step, **kw):
        pass

    def validate(self, step):
        pass


def test_install_splits_in_fine_tuning_only_and_logs_it():
    log: dict = {}
    rule = EarlyStopRule(burnin=10**9)  # never checks: this test is about the rate
    install(FakeTrainer, rule, log, param_lr=ParamLrSettings(lr=1e-3))
    pre = FakeTrainer("init", 50)
    pre.train_loop()
    assert "param_lr" not in log["init"] and len(pre.optimizer.param_groups) == 1

    fine = FakeTrainer("inv", 100)
    fine.train_loop()
    entry = log["inv"]["param_lr"]
    assert entry["settings"]["lr"] == 1e-3
    assert [g["lr"] for g in entry["groups_at_end"]] == [1e-4, 1e-3]
    assert entry["start_values"]["rD"] == 1.0
    travel = entry["start_values"]["rD"] - entry["end_values"]["rD"]
    assert 0.09 <= travel <= 0.101, travel  # 100 steps x 1e-3, not x 1e-4


@pytest.mark.parametrize("name, lr", [("paper_schedule_reach.yaml", 1e-3),
                                      ("paper_schedule_reach_3e-4.yaml", 3e-4)])
def test_reach_configs_differ_from_paper_schedule_only_in_param_lr(name, lr):
    base = yaml.safe_load((REPO / "configs" / "paper_schedule.yaml").read_text(encoding="utf-8"))
    reach = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
    block = reach["pinn"].pop("param_lr")
    assert reach == base
    assert ParamLrSettings.from_config(block) == ParamLrSettings(lr=lr)
