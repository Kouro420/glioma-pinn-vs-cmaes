"""Own Adam learning rate for the PDE parameters rD and rRHO. MODIFIED BY THE AUTHORS
OF THIS REPOSITORY: not the published method.

Why this exists (the reach test)
--------------------------------
The reference code puts the network weights, rD, rRHO and the two thresholds
in ONE Adam (AMSGrad) parameter group (``Trainer.py:34``), so all of them share
the fine-tuning rate ``lr_inv``. Adam moves a parameter by at most about ``lr``
per step, whatever the size of its gradient, and in practice rD moves by close
to ``lr`` per step. On the training schedule of Zhang et al.
(``configs/paper_schedule.yaml``: ``lr_inv`` 1e-4, 5,000 steps) that bounds the
reach at about 1e-4 x 5,000 = 0.5 from the start of 1, while the truths lie at
0.21-1.58. The reach test gives rD and rRHO their own group at 1e-3 (the PyTorch
port's own ``lr_inv`` default, ``Options.py:79``) and leaves the network and the
thresholds at 1e-4.

What it does, and what it keeps
-------------------------------
``split_param_group`` moves the named parameters out of their group into one
new group with the given rate. Every other setting of the new group (AMSGrad,
betas, eps, weight decay) is copied from the group they left, so only the rate
differs. Adam's moments are stored per parameter object, not per group, so the
network's state carried over from pre-training (``traintype: full``) is kept
exactly. The split is done at the start of the stage's training loop, i.e.
AFTER ``Engine.configure_stage`` has called ``Trainer.set_lr(lr_inv)``, which
sets every group and would otherwise overwrite the new rate.

Nothing in ``external/pinngbm-torch`` is edited; ``experiments/pinn_early_stopping``
calls this from its replacement training loop, so it is used only together with
the training schedule of Zhang et al.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamLrSettings:
    """Which reference parameters get their own rate, the rate, and in which stages."""

    params: tuple[str, ...] = ("rD", "rRHO")
    lr: float = 1e-3
    stages: tuple[str, ...] = ("inv",)

    @classmethod
    def from_config(cls, block: dict) -> "ParamLrSettings":
        known = {k: block[k] for k in cls.__dataclass_fields__ if k in block}
        for key in ("params", "stages"):
            if key in known:
                known[key] = tuple(known[key])
        if "lr" in known:
            known["lr"] = float(known["lr"])
        return cls(**known)


def split_param_group(optimizer: torch.optim.Optimizer, named: dict[str, torch.nn.Parameter],
                      lr: float, scheduler=None) -> dict:
    """Move ``named`` parameters into one new optimizer group with rate ``lr``.

    Returns a record of the groups after the split: for each, the rate and
    either the parameter names (new group) or the parameter count.
    Raises ``ValueError`` if a parameter is not in the optimizer.
    """
    targets = list(named.values())
    target_ids = {id(p) for p in targets}
    template = None
    found: set[int] = set()
    for group in optimizer.param_groups:
        kept = [p for p in group["params"] if id(p) not in target_ids]
        if len(kept) != len(group["params"]):
            found |= {id(p) for p in group["params"] if id(p) in target_ids}
            template = template or group
        group["params"] = kept
    missing = [name for name, p in named.items() if id(p) not in found]
    if missing:
        raise ValueError(f"not in the optimiser: {missing}")

    new_group = {k: v for k, v in template.items() if k != "params"}
    new_group.update(params=targets, lr=float(lr), initial_lr=float(lr))
    optimizer.add_param_group(new_group)
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        # ExponentialLR (gamma 1.0 in the reference) scales each group's own
        # current rate, so the new group keeps lr; base_lrs is kept aligned for
        # any closed-form use.
        scheduler.base_lrs.append(float(lr))
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]
    return group_record(optimizer, named)


def group_record(optimizer: torch.optim.Optimizer, named: dict[str, torch.nn.Parameter]) -> list:
    """The rate of every group, with the names of the named parameters it holds."""
    names = {id(p): name for name, p in named.items()}
    record = []
    for group in optimizer.param_groups:
        held = [names[id(p)] for p in group["params"] if id(p) in names]
        record.append({"lr": float(group["lr"]),
                       "params": held if held else f"{len(group['params'])} other tensors"})
    return record


def install_for_stage(trainer, settings: ParamLrSettings) -> dict | None:
    """Split the trainer's optimizer if this stage is listed; return the record."""
    if trainer.stage not in settings.stages:
        return None
    named = {name: trainer.net.all_params_dict[name] for name in settings.params}
    groups = split_param_group(trainer.optimizer, named, settings.lr,
                               getattr(trainer, "scheduler", None))
    return {"settings": {"params": list(settings.params), "lr": settings.lr,
                         "stages": list(settings.stages)},
            "groups_at_start": groups,
            "start_values": {name: float(p.detach()) for name, p in named.items()}}


def finish_stage(trainer, settings: ParamLrSettings, entry: dict) -> dict:
    """Add the rates and values at the end of the stage to ``entry``.

    ``groups_at_end`` shows that nothing reset the new group's rate during
    the stage; the values show how far each parameter travelled.
    """
    named = {name: trainer.net.all_params_dict[name] for name in settings.params}
    entry["groups_at_end"] = group_record(trainer.optimizer, named)
    entry["end_values"] = {name: float(p.detach()) for name, p in named.items()}
    return entry
