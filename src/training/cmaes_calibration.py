#!/usr/bin/env python
"""Derivative-free calibration of the forward model with CMA-ES.

CMA-ES (Hansen and Ostermeier 2001) samples candidate parameter sets from a
Gaussian, keeps the better ones, and adapts the sampling shape. It needs no
gradients, which suits a score containing a thresholding step.

This is the baseline the PINN is judged against. To keep the comparison fair,
every choice here mirrors the PINN arm:

* the same four unknowns, mu_D, mu_rho and the two thresholds,
* the same score (``src/training/objective.py``),
* the same starting point, mu = 1 from the shared characteristic grid search
  and the start values of the two thresholds,
* the same hardware, one GPU per run,
* a progress curve in wall-clock seconds, so quality at an equal time budget
  can be read off afterwards rather than assumed.

Balcerak et al. (2025) time a comparable CMA-ES at 80 +/- 55 min per patient on
16 CPU cores; here the forward solves run on the GPU instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cma
import numpy as np

from src.models.characteristic import CharacteristicScaling
from src.models.fisher_kpp import solve_fisher_kpp
from src.training.objective import SegmentationObjective

__all__ = ["CalibrationSettings", "CalibrationResult", "calibrate"]

PARAMETER_NAMES = ("mu_d", "mu_rho", "th_wt", "th_tc")


@dataclass
class CalibrationSettings:
    """Search settings, all read from ``configs/base.yaml``."""

    lower: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    upper: Sequence[float] = (2.0, 2.0, 1.0, 1.0)
    initial: Sequence[float] = (1.0, 1.0, 0.35, 0.70)
    sigma0: float = 0.25
    popsize: int = 8
    max_evaluations: int = 600
    tol_fun: float = 1e-12
    seed: int = 0  # random seed of pycma; 0 draws a clock seed, so the run cannot be repeated exactly
    time_budget_s: float | None = None
    # Parameters held at a given value and removed from the search (for example
    # the thresholds fixed at the truth). Empty: all four are searched, as in
    # the main runs.
    fixed: dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Outcome of one calibration, including the progress curve."""

    best_parameters: dict[str, float]
    best_score: float
    n_evaluations: int
    wall_clock_s: float
    history: list[dict] = field(default_factory=list)
    stop_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "n_evaluations": self.n_evaluations,
            "wall_clock_s": self.wall_clock_s,
            "stop_reason": self.stop_reason,
            "history": self.history,
        }


def _to_unit(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (x - lower) / (upper - lower)


def _from_unit(z: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower + np.clip(z, 0.0, 1.0) * (upper - lower)


def calibrate(
    *,
    p_wm: np.ndarray,
    p_gm: np.ndarray,
    phi: np.ndarray,
    u0: np.ndarray,
    objective: SegmentationObjective,
    scaling: CharacteristicScaling,
    settings: CalibrationSettings | None = None,
    factor: float = 10.0,
    voxel_mm: float = 1.0,
    cfl_safety: float = 0.99,
    device: str | None = None,
    on_improvement: Callable[[dict], None] | None = None,
) -> CalibrationResult:
    """Search the four parameters that best explain the two MRI masks.

    Args:
        p_wm, p_gm: tissue probability maps.
        phi: phase field of the brain domain.
        u0: fixed initial condition, a Gaussian centered on the tumor seed position x0.
        objective: the shared score.
        scaling: characteristic scaling mapping mu_D, mu_rho to solver inputs.
        settings: search settings; defaults match ``configs/base.yaml``.
        on_improvement: optional callback fired whenever the best score drops.

    Returns:
        The best parameters, the score, and the full progress curve. Every
        history entry records elapsed seconds, so cost curves need no re-run.
    """
    settings = settings or CalibrationSettings()
    lower = np.asarray(settings.lower, dtype=float)
    upper = np.asarray(settings.upper, dtype=float)
    initial = np.array(settings.initial, dtype=float)  # a copy: fixed values are written in

    if not (len(lower) == len(upper) == len(initial) == len(PARAMETER_NAMES)):
        raise ValueError("bounds and the initial point must have four entries")

    # Fixed parameters are written into the starting point and never sampled:
    # CMA-ES searches only the free coordinates, and every evaluated point carries
    # the fixed values. With nothing fixed this is exactly the four-parameter search.
    unknown = set(settings.fixed) - set(PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"cannot fix unknown parameters: {sorted(unknown)}")
    for name, value in settings.fixed.items():
        i = PARAMETER_NAMES.index(name)
        if not lower[i] <= value <= upper[i]:
            raise ValueError(f"fixed {name}={value} lies outside [{lower[i]}, {upper[i]}]")
        initial[i] = value
    free = [i for i, name in enumerate(PARAMETER_NAMES) if name not in settings.fixed]
    if len(free) < 2:
        raise ValueError("pycma needs at least two free parameters")

    def full_point(z: np.ndarray) -> np.ndarray:
        values = initial.copy()
        values[free] = _from_unit(np.asarray(z), lower[free], upper[free])
        return values

    strategy = cma.CMAEvolutionStrategy(
        _to_unit(initial[free], lower[free], upper[free]).tolist(),
        settings.sigma0,
        {
            "bounds": [0.0, 1.0],
            "popsize": settings.popsize,
            "maxfevals": settings.max_evaluations,
            "tolfun": settings.tol_fun,
            "seed": settings.seed,
            "verbose": -9,
        },
    )

    started = time.perf_counter()
    history: list[dict] = []
    best_score = np.inf
    best_parameters = dict(zip(PARAMETER_NAMES, initial.tolist()))
    n_evaluations = 0
    stop_reason = ""

    def evaluate(values: np.ndarray) -> float:
        """Solve the forward model once, and record the point if it is the best.

        Pulled out of the loop so that the starting point can be scored by the
        same code path as every sampled candidate, with the same solver settings
        and the same bookkeeping.
        """
        nonlocal best_score, best_parameters, n_evaluations

        mu_d, mu_rho, th_wt, th_tc = (float(v) for v in values)
        d_eff, rho_eff = scaling.to_physical(mu_d, mu_rho)

        solution = solve_fisher_kpp(
            p_wm,
            p_gm,
            phi,
            u0,
            d_w=d_eff,
            rho=rho_eff,
            t_final=1.0,
            factor=factor,
            n_save=2,
            h=voxel_mm,
            cfl_safety=cfl_safety,
            device=device,
        )
        terms = objective(
            solution.final,
            mu_d=mu_d,
            mu_rho=mu_rho,
            th_wt=th_wt,
            th_tc=th_tc,
        )

        index = n_evaluations
        n_evaluations += 1

        if terms.total < best_score:
            best_score = terms.total
            best_parameters = {
                "mu_d": mu_d,
                "mu_rho": mu_rho,
                "th_wt": th_wt,
                "th_tc": th_tc,
            }
            entry = {
                "evaluation": index,
                "elapsed_s": time.perf_counter() - started,
                "score": terms.total,
                "d_eff": d_eff,
                "rho_eff": rho_eff,
                "lambda_mm": scaling.infiltration_length(mu_d, mu_rho),
                **best_parameters,
                **terms.as_dict(),
            }
            history.append(entry)
            if on_improvement is not None:
                on_improvement(entry)

        return terms.total

    # Evaluation 0 is the starting point itself: mu_D = mu_rho = 1 from the
    # characteristic grid search, with the two thresholds at their start values.
    # CMA-ES samples a population before it scores anything, so without this the
    # cost curve would begin at whatever the first population found, and the
    # question our paper asks (how much does the search add on top of the shared
    # grid search?) would have no baseline. The PINN arm starts from the same
    # point, so both curves share an origin.
    evaluate(initial)

    while not strategy.stop():
        candidates = strategy.ask()
        scores = []
        for candidate in candidates:
            scores.append(evaluate(full_point(candidate)))

            if settings.time_budget_s is not None and (
                time.perf_counter() - started > settings.time_budget_s
            ):
                stop_reason = "time budget reached"
                break

        if stop_reason:
            break
        strategy.tell(candidates, scores)

    if not stop_reason:
        stop_reason = "; ".join(f"{k}={v}" for k, v in strategy.stop().items()) or "converged"

    return CalibrationResult(
        best_parameters=best_parameters,
        best_score=float(best_score),
        n_evaluations=n_evaluations,
        wall_clock_s=time.perf_counter() - started,
        history=history,
        stop_reason=stop_reason,
    )
