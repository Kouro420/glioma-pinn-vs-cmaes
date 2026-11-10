"""The early-stopping rule of the original TensorFlow code, added to the PyTorch port.

Why this exists
---------------
Zhang et al. (2025) train with early stopping (App. E, p. 14: pre-training
runs "20 000 max" steps and stops after 1000 without progress). The original
TensorFlow code (https://github.com/Rayzhangzirui/pinngbm, commit 41e2eeb) does
it in ``pinn.py:59-100``: patience 1000, ``min_delta`` 1e-6, burn-in 1000,
checked after every step, and an extra stop when ``rD`` falls below 0.05. The PyTorch port we run
(``external/pinngbm-torch``, commit 99af20f) dropped it and trains a fixed
number of steps (``Trainer.py:82-102``).

This module puts the rule back **without editing the reference code**: it
replaces ``Trainer.train_loop`` with a copy of the same loop plus a check, the
same way ``experiments/run_pinn.py`` wraps the reference solver.

What is the same as the TensorFlow rule
---------------------------------------
* patience, ``min_delta``, burn-in and the ``rD < 0.05`` stop, with the
  TensorFlow values as defaults;
* one stopper per stage, reset between stages (``callback_train_end``);
* burn-in counted from the start of each stage, as in TensorFlow, where
  pre-training and fine-tuning are separate runs;
* what is watched: in pre-training the **total** loss (TensorFlow ``solvechar``
  watches ``total``), in fine-tuning the **unweighted sum of the two mask
  losses** (TensorFlow ``patient`` and ``segonly`` watch ``pdattest``, the sum of
  the data losses, ``options.py:239, 262``, ``pinn.py:440-445``);
* the network kept at the end is the last one, not the best one, as in
  TensorFlow.

What is different, and why
--------------------------
* **The watched loss is computed on one fixed batch drawn at the start of the
  stage.** TensorFlow trains on the full point set, so its loss is the same
  function from step to step. The PyTorch port draws a new random batch every
  step, so its training loss is noisy, and a noisy loss sets a new "best" by
  chance and would almost never trigger a patience rule. A fixed batch removes
  that noise. It is **not held out**: the PyTorch port has no test split, and in
  fine-tuning TensorFlow's ``pdattest`` is on held-out data points.
* **The check runs every ``check_every`` steps, not every step**, because each
  check costs one extra forward and backward pass. The waiting count grows by
  ``check_every`` per check, so patience is still counted in training steps.
  A stop can therefore come up to ``check_every - 1`` steps later than in
  TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class EarlyStopRule:
    """Settings of the rule. Defaults are the TensorFlow values (``options.py:26``)."""

    patience: int = 1000
    min_delta: float = 1e-6
    burnin: int = 1000
    check_every: int = 10
    min_mu_d: float = 0.05  # ``pinn.py:81``: stop if rD < 0.05
    # Stages in which the rule may END training. In any other stage it is checked and
    # logged ("would_stop"), not applied: the no-data test runs fine-tuning for its
    # full budget so that mu_D can be compared step by step with the runs of
    # configs/paper_schedule.yaml.
    stop_stages: tuple = ("init", "inv")

    @classmethod
    def from_config(cls, block: dict) -> "EarlyStopRule":
        known = {k: block[k] for k in cls.__dataclass_fields__ if k in block}
        if "stop_stages" in known:
            known["stop_stages"] = tuple(known["stop_stages"])
        return cls(**known)


@dataclass
class StageStopper:
    """Patience bookkeeping for one training stage.

    ``update`` is called after ``steps_done`` steps of the stage. It returns
    True when training should stop and then sets ``reason``.
    """

    rule: EarlyStopRule
    best: float = float("inf")
    best_step: int | None = None
    wait: int = 0
    reason: str = ""
    checks: list[tuple[int, float]] = field(default_factory=list)

    def due(self, steps_done: int) -> bool:
        return steps_done % self.rule.check_every == 0 and steps_done >= self.rule.burnin

    def update(self, steps_done: int, value: float, mu_d: float | None = None) -> bool:
        self.checks.append((steps_done, value))
        if mu_d is not None and mu_d < self.rule.min_mu_d:
            self.reason = f"mu_D {mu_d:.4g} below {self.rule.min_mu_d}"
            return True
        if value < self.best - self.rule.min_delta:
            self.best, self.best_step, self.wait = value, steps_done, 0
            return False
        # The first due check always improves on inf, so wait counts checks after it.
        self.wait += self.rule.check_every
        if self.wait >= self.rule.patience:
            self.reason = f"no improvement over {self.rule.patience} steps"
            return True
        return False


# What each stage watches. Pre-training: the weighted total of its own losses.
# Fine-tuning: the unweighted sum of the two mask losses.
MONITOR = {"init": "total", "inv": "seg1+seg2"}


def _cuda_allocated() -> int | None:
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else None


def _monitor_value(trainer, fixed_batch: dict) -> float:
    """The watched loss on the fixed batch, leaving no graph behind.

    The reference residual loss stores its output on the problem object:
    ``GBMproblem.resloss`` and ``get_res_pred`` set ``self.res`` and
    ``self.upred_res`` (``GBMproblem.py:260, 274-275``). In training,
    ``backward()`` frees the graph behind them. This check never calls
    ``backward()``, so the stored tensors would keep a second-order autograd graph
    alive into the next training step. Every attribute the check adds or replaces
    is therefore put back to what training left there. The caller measures
    allocated GPU memory before and after this call.
    """
    pde = trainer.pde
    dataset = pde.dataset
    pde_state = dict(vars(pde))
    saved = dataset.batch
    dataset.batch = fixed_batch
    try:
        # Gradients stay on: the residual loss differentiates u with respect to x and t.
        if trainer.stage == "inv":
            result = trainer.get_wloss_sum_comp(["seg1", "seg2"], True)
            value = sum(float(v.detach()) for v in result[2].values())
        else:
            result = trainer.get_wloss_sum_comp(trainer.loss, True)
            value = float(result[0].detach())
        # Every tensor of the call hangs off `result`: drop it before returning.
        del result
    finally:
        dataset.batch = saved
        for name in list(vars(pde)):
            if name not in pde_state:
                delattr(pde, name)
            elif getattr(pde, name) is not pde_state[name]:
                setattr(pde, name, pde_state[name])
        del pde_state
    trainer.optimizer.zero_grad()
    return value


def install(trainer_cls, rule: EarlyStopRule, log: dict, lbfgs=None, param_lr=None,
            paper=None) -> None:
    """Replace ``trainer_cls.train_loop`` with the reference loop plus the rule.

    ``log`` receives one entry per stage (``"init"``, ``"inv"``) with the step
    budget, the steps actually run, why it stopped, and every checked value.

    ``lbfgs`` (an ``experiments.pinn_lbfgs.LbfgsSettings``) adds the original
    TensorFlow code's L-BFGS stage after Adam in the stages it lists; its record
    goes under the stage's ``"lbfgs"`` key.

    ``param_lr`` (an ``experiments.pinn_param_lr.ParamLrSettings``) gives rD and
    rRHO their own Adam rate in the stages it lists (the reach test; our
    modification, not part of the published method); its record goes under the stage's
    ``"param_lr"`` key.

    ``paper`` (an ``experiments.pinn_paper_settings.PaperSettings``) fixes each stage's
    collocation sets and the loss forms of Zhang et al. before the first step (the
    paper-settings run); its record goes under the stage's ``"paper_settings"`` key.
    """

    def train_loop(self):
        paper_entry = None
        if paper is not None:
            # First, so that the early-stopping check and L-BFGS use the same fixed sets.
            from experiments.pinn_paper_settings import install_for_stage as install_paper

            paper_entry = install_paper(self, paper)
        param_lr_entry = None
        if param_lr is not None:
            # Here, not earlier: Engine.configure_stage has just called
            # set_lr(lr_inv), which sets the rate of every group.
            from experiments.pinn_param_lr import install_for_stage

            param_lr_entry = install_for_stage(self, param_lr)
        stopper = StageStopper(rule)
        fixed_batch = {name: dict(batch) for name, batch in self.pde.dataset.batch.items()}
        trains_mu_d = "rD" in getattr(self.net, "trainable_param", []) and self.stage == "inv"
        steps_done = 0
        stop_step = None
        would_stop = None
        memory_deltas: list[int] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # The body below is Trainer.train_loop (Trainer.py:82-102), unchanged.
        for step in range(self.start_step, self.start_step + self.num_iter):
            self.optimizer.zero_grad()
            wtotal, wloss_comp, uwloss_comp = self.get_wloss_sum_comp(self.loss, True)
            self.log_stat(step, **uwloss_comp, total=wtotal)
            self.validate(step)
            wtotal.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.pde.dataset.next_batch()
            # End of the reference body.

            steps_done += 1
            if stopper.due(steps_done):
                mu_d = float(self.net.all_params_dict["rD"].detach()) if trains_mu_d else None
                before = _cuda_allocated()
                value = _monitor_value(self, fixed_batch)
                after = _cuda_allocated()
                if before is not None:
                    memory_deltas.append(after - before)
                if stopper.update(steps_done, value, mu_d):
                    if self.stage in rule.stop_stages:
                        stop_step = step
                        print(f"[early stop] stage={self.stage} step={step} "
                              f"after {steps_done} of {self.num_iter}: {stopper.reason}")
                        break
                    if would_stop is None:
                        would_stop = {"steps_done": steps_done, "reason": stopper.reason}

        log[self.stage] = {
            "max_steps": int(self.num_iter),
            "steps_run": steps_done,
            "stopped_early": stop_step is not None,
            "stop_step": stop_step,
            "stop_reason": stopper.reason if stop_step is not None else "step budget reached",
            "rule_applied": self.stage in rule.stop_stages,
            "would_stop": would_stop,
            "monitor": MONITOR.get(self.stage, "total"),
            "best_value": None if stopper.best_step is None else stopper.best,
            "best_at_steps_done": stopper.best_step,
            "checks": stopper.checks,
            # Must stay near zero: a check that left memory behind would keep a
            # second autograd graph alive (see _monitor_value).
            "check_memory_delta_max_bytes": max(memory_deltas) if memory_deltas else None,
            "peak_allocated_bytes": (torch.cuda.max_memory_allocated()
                                     if torch.cuda.is_available() else None),
        }
        if paper_entry is not None:
            log[self.stage]["paper_settings"] = paper_entry
        if param_lr_entry is not None:
            from experiments.pinn_param_lr import finish_stage

            log[self.stage]["param_lr"] = finish_stage(self, param_lr, param_lr_entry)
        if lbfgs is not None and self.stage in lbfgs.stages:
            # Imported here: pinn_lbfgs imports this module.
            from experiments.pinn_lbfgs import run_lbfgs

            log[self.stage]["lbfgs"] = run_lbfgs(self, fixed_batch, lbfgs, rule)
            log[self.stage]["peak_allocated_bytes_with_lbfgs"] = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)

    trainer_cls.train_loop = train_loop
