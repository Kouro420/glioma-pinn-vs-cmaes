"""The L-BFGS stage of the original TensorFlow code, added after Adam in pre-training.

Why this exists
---------------
Zhang et al. (2025), App. E (p. 14), pre-train with Adam **and then
L-BFGS-B**. The original TensorFlow code (https://github.com/Rayzhangzirui/pinngbm,
commit 41e2eeb) runs ``scipy.optimize.minimize(method='L-BFGS-B')`` after Adam in its
``solve()`` (``glioma.py:403-408``), with ``lbfgs_opts`` from ``options.py:7``,
and its early-stopping callback applies to L-BFGS as well (``pinn.py:425-450``).
The PyTorch port we run (``external/pinngbm-torch``, commit 99af20f) has no
L-BFGS at all: its optimizer table holds only Adam and AdamW (``Trainer.py:16-19``).
It is built here as its own configuration, beside the training schedule of
Zhang et al. (``configs/paper_schedule.yaml``) and not inside it, so that the
two changes can be told apart.

What is the same as the TensorFlow stage
-----------------------------------------
* the settings: history 100 (``maxcor``), loss tolerance 2.22e-9 (``ftol``),
  gradient tolerance 0 (``gtol``), at most 10,000 iterations (``maxiter``) and
  10,000 loss evaluations (``maxfun``);
* **full batch**: one fixed set of points for the whole stage, so the loss is one
  deterministic function, which a quasi-Newton method needs;
* the early-stopping rule of ``pinn_early_stopping`` keeps running during
  L-BFGS, with its best value reset at the switch, as TensorFlow's
  ``callback_train_end`` resets it; the burn-in has already passed (TensorFlow's
  iteration counter continues from Adam);
* only the network weights move; in pre-training the model parameters are frozen.

What is different, and why
---------------------------
* ``torch.optim.LBFGS`` with the strong-Wolfe line search, not scipy's L-BFGS-B.
  The problem is unbounded here, so the "-B" (bounds) part does nothing in
  TensorFlow either. torch's line search allows at most 25 evaluations per
  search (hard-coded), against scipy's ``maxls`` 50.
* torch's loss tolerance is absolute; scipy's ``ftol`` is relative to
  ``max(|f|, 1)``. The pre-training losses are below 1, so the two coincide.
* The fixed batch has the PyTorch port's own batch size (100,000 data and 200,000
  residual points), not TensorFlow's 50,000.
* Only the stages listed in the config run it. Default: pre-training only, as
  App. E describes. Whether Zhang et al. also used it in fine-tuning is not
  published.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from experiments.pinn_early_stopping import EarlyStopRule, StageStopper, _monitor_value

# scipy's default ftol, which TensorFlow options.py:7 copies: np.finfo(float).eps * 1e7.
SCIPY_FTOL = 2.2204460492503131e-09


@dataclass(frozen=True)
class LbfgsSettings:
    """Defaults are TensorFlow ``options.py:7``."""

    max_iter: int = 10000
    max_fun: int = 10000
    history_size: int = 100
    tolerance_change: float = SCIPY_FTOL
    tolerance_grad: float = 0.0
    stages: tuple[str, ...] = ("init",)
    # True: the early-stopping patience rule stops L-BFGS, as TensorFlow's callback
    # does. False: the rule is only logged ("would_stop_at"), and L-BFGS runs to
    # its own limits, so the record shows how far it gets.
    apply_patience: bool = True

    @classmethod
    def from_config(cls, block: dict) -> "LbfgsSettings":
        known = {k: block[k] for k in cls.__dataclass_fields__ if k in block}
        if "stages" in known:
            known["stages"] = tuple(known["stages"])
        return cls(**known)


def run_lbfgs(trainer, batch: dict, settings: LbfgsSettings, rule: EarlyStopRule) -> dict:
    """Run L-BFGS on ``trainer``'s current stage loss over one fixed ``batch``.

    Returns a record: iterations, loss evaluations, the loss before and after,
    why it stopped, and every checked value.
    """
    dataset = trainer.pde.dataset
    params = [p for p in trainer.net.parameters() if p.requires_grad]
    optimiser = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=rule.check_every,  # one step() call = one check interval
        # Without max_eval torch caps each step() at max_iter * 5 // 4 = 12 loss
        # evaluations, and a line search that needs more returns early, which the
        # loop below cannot tell from convergence. So max_eval is set to the whole
        # stage's evaluation budget, and the loop below enforces that budget.
        max_eval=settings.max_fun,
        history_size=settings.history_size,
        tolerance_grad=settings.tolerance_grad,
        tolerance_change=settings.tolerance_change,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimiser.zero_grad()
        loss = trainer.get_wloss_sum_comp(trainer.loss, True)[0]
        loss.backward()
        return loss

    # The burn-in has passed (TensorFlow's counter runs on from Adam); the best
    # value starts afresh, as TensorFlow's callback_train_end resets it.
    stopper = StageStopper(replace(rule, burnin=0))
    loss_start = _monitor_value(trainer, batch)
    iterations = evaluations = 0
    reason = ""
    would_stop_at = None  # the patience rule's verdict when it is only logged
    saved = dataset.batch
    dataset.batch = batch
    try:
        while True:
            optimiser.step(closure)
            state = optimiser.state[params[0]]
            progressed = int(state["n_iter"]) - iterations
            iterations = int(state["n_iter"])
            evaluations = int(state["func_evals"])
            value = _monitor_value(trainer, batch)
            if not math.isfinite(value):
                reason = "non-finite loss"
                break
            if stopper.update(iterations, value):
                if settings.apply_patience:
                    reason = f"early stop: {stopper.reason}"
                    break
                if would_stop_at is None:
                    would_stop_at = {"iteration": iterations, "reason": stopper.reason}
            if progressed < rule.check_every:
                # max_eval is the stage budget, so an early return is torch's own
                # tolerance test (loss change or step size), or the budget itself.
                if evaluations >= settings.max_fun:
                    reason = "L-BFGS evaluation budget reached"
                else:
                    reason = "L-BFGS converged (loss-change or step-size tolerance)"
                break
            if iterations >= settings.max_iter:
                reason = "L-BFGS iteration budget reached"
                break
            if evaluations >= settings.max_fun:
                reason = "L-BFGS evaluation budget reached"
                break
    finally:
        dataset.batch = saved
        optimiser.state.clear()  # the curvature history is ~100 x (network size)
    loss_end = stopper.checks[-1][1] if stopper.checks else loss_start
    print(f"[lbfgs] stage={trainer.stage} iterations={iterations} evaluations={evaluations} "
          f"loss {loss_start:.4e} -> {loss_end:.4e}: {reason}")
    return {
        "settings": {**settings.__dict__, "stages": list(settings.stages),
                     "line_search": "strong_wolfe (torch, at most 25 evaluations per search)"},
        "iterations": iterations,
        "function_evals": evaluations,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "stop_reason": reason,
        "patience_would_stop_at": would_stop_at,
        "checks": stopper.checks,
    }
