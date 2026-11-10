#!/usr/bin/env python
"""Patient-specific characteristic parameters, and the scaling they define.

Both calibration methods start from the same point, so that the
comparison measures the search, not the initialization. That common starting
point is the characteristic scaling of Zhang et al. (2025), Sec. 2.1.3 and 2.2:

1. Take the tumor-core and whole-tumor radii from the MRI masks.
2. Grid-search a 1D spherically symmetric Fisher-Kolmogorov solution for the
   pair (Dbar, rhobar) whose threshold radii match those two numbers.
3. Use ``L = ceil(r_wt)`` as the length scale and ``T = L / sqrt(Dbar rhobar)``
   as the time scale.

Why this matters. From one imaging time point the triple
(D, rho, t_end) is not identifiable: doubling the time while halving D and rho
leaves the solution unchanged (Zhang et al. 2025, p. 3). Only the products
D*t_end and rho*t_end are constrained. This module therefore reports the
dimensionless products and the infiltration length

    lambda = sqrt(D / rho)   [mm],

never an absolute D in mm^2/day or rho in 1/day.

The reference implementation is ``external/pinngbm-torch/gbmutil.py``; this is
an independent implementation with the same grid and the same radius rules, so
that CMA-ES and the PINN are initialized identically.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

__all__ = [
    "CharacteristicScaling",
    "solve_radial",
    "volume_radius",
    "max_radius",
    "estimate_characteristic",
]

RadiusMethod = Literal["rvol", "rmax"]


@dataclass
class CharacteristicScaling:
    """The patient-specific scaling and the map back to physical quantities.

    Attributes:
        d_char: characteristic diffusivity Dbar, mm^2/day in the surrogate.
        rho_char: characteristic proliferation rate rhobar, 1/day.
        length: characteristic length L, mm.
        time: characteristic time T = L / sqrt(Dbar rhobar), days.
        r_wt_data, r_tc_data: radii measured from the masks, mm.
        r_wt_sim, r_tc_sim: radii of the matched surrogate solution, mm.
        score: relative L1 mismatch of the two radii, lower is better.
    """

    d_char: float
    rho_char: float
    length: float
    time: float
    r_wt_data: float
    r_tc_data: float
    r_wt_sim: float
    r_tc_sim: float
    score: float

    def to_physical(self, mu_d: float, mu_rho: float) -> tuple[float, float]:
        """Map the dimensionless parameters to solver inputs at ``t_final = 1``.

        The non-dimensional equation (Zhang et al. 2025, Eq. 7) is solved on the
        millimeter voxel grid over the unit time interval, so the solver takes

            d_w  = mu_d   * Dbar   * T      [mm^2 per unit time]
            rho  = mu_rho * rhobar * T      [1 per unit time]

        These are exactly the products D*t_end and rho*t_end discussed above.
        """
        return mu_d * self.d_char * self.time, mu_rho * self.rho_char * self.time

    def infiltration_length(self, mu_d: float, mu_rho: float) -> float:
        """Infiltration length ``lambda = sqrt(D / rho)`` in mm.

        This combination does not depend on the time scale: t_end cancels in
        the ratio of the two products.
        """
        d_eff, rho_eff = self.to_physical(mu_d, mu_rho)
        if rho_eff <= 0:
            return float("nan")
        return float(np.sqrt(d_eff / rho_eff))

    def as_dict(self) -> dict:
        return asdict(self)


def solve_radial(
    *,
    d: float,
    rho: float,
    t_final: float,
    h_init: float,
    r_init: float,
    domain: float = 180.0,
    dx: float = 1.0,
    xdim: int = 3,
    cfl_safety: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the spherically symmetric Fisher-Kolmogorov equation.

    Solves ``u_t = d (u_rr + (xdim-1)/r u_r) + rho u (1-u)`` on ``[0, domain]``
    with no-flux conditions at both ends and the Gaussian initial condition
    ``u(r, 0) = h_init exp(-r_init r^2)``.

    Returns:
        The radial grid and the density at ``t_final``.
    """
    if xdim not in (2, 3):
        raise ValueError(f"xdim must be 2 or 3, got {xdim}")

    n = int(np.round(domain / dx)) + 1
    r = np.linspace(0.0, domain, n)
    dx = float(r[1] - r[0])
    u = h_init * np.exp(-r_init * r**2)

    dt = cfl_safety * dx**2 / (2.0 * xdim * d)
    inv_dx2 = 1.0 / dx**2
    inv_2dx = 1.0 / (2.0 * dx)

    t = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        while t < t_final:
            step = min(dt, t_final - t)
            u_ip1 = np.roll(u, -1)
            u_im1 = np.roll(u, 1)
            lap = (u_ip1 - 2.0 * u + u_im1) * inv_dx2 + ((xdim - 1.0) / r) * (u_ip1 - u_im1) * inv_2dx
            lap[0] = xdim * 2.0 * (u[1] - u[0]) * inv_dx2
            lap[-1] = 2.0 * (u[-2] - u[-1]) * inv_dx2
            u = u + step * (d * lap + rho * u * (1.0 - u))
            t += step
    return r, u


def volume_radius(mask: np.ndarray, xdim: int = 3) -> float:
    """Radius of the sphere (or disc) with the same volume as ``mask``, in mm."""
    volume = float(np.count_nonzero(mask))
    if volume == 0.0:
        return 0.0
    if xdim == 2:
        return float(np.sqrt(volume / np.pi))
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def max_radius(mask: np.ndarray, centre: np.ndarray | None = None) -> float:
    """Largest distance from ``centre`` to any voxel of ``mask``, in mm."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0
    centre = coords.mean(axis=0) if centre is None else np.asarray(centre, dtype=float)
    return float(np.linalg.norm(coords - centre, axis=1).max())


def _threshold_radius(r: np.ndarray, u: np.ndarray, threshold: float) -> float:
    """First radius at which the density drops below ``threshold``."""
    below = np.flatnonzero(u < threshold)
    return float(r[below[0]]) if below.size else float("nan")


def estimate_characteristic(
    mask_wt: np.ndarray,
    mask_tc: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    th_wt: float,
    th_tc: float,
    h_init: float,
    r_init: float,
    xdim: int = 3,
    radius_method: RadiusMethod = "rvol",
    d_grid: np.ndarray | None = None,
    rho_grid: np.ndarray | None = None,
    domain: float = 180.0,
) -> CharacteristicScaling:
    """Grid-search the characteristic scaling for one case.

    Args:
        mask_wt, mask_tc: binary whole-tumor and tumor-core masks.
        x0: tumor-core centroid in voxel coordinates; used by ``rmax`` only.
        th_wt, th_tc: start values of the two mask thresholds (0.35 and 0.7 here).
        h_init, r_init: amplitude and decay of the Gaussian initial condition.
        radius_method: ``rvol`` (equivalent-volume radius) or ``rmax``.
        d_grid, rho_grid: search grids; default to the 10x10 grid of the
            reference implementation, Dbar in [0.1, 1] and rhobar in [0.01, 0.1].

    Returns:
        The matched :class:`CharacteristicScaling`.

    Raises:
        ValueError: if a mask is empty.
        RuntimeError: if no grid point produces both threshold radii.
    """
    if not np.any(mask_wt) or not np.any(mask_tc):
        raise ValueError("both the whole-tumor and tumor-core masks must be non-empty")

    if radius_method == "rvol":
        r_wt_data = volume_radius(mask_wt, xdim)
        r_tc_data = volume_radius(mask_tc, xdim)
    else:
        r_wt_data = max_radius(mask_wt, x0)
        r_tc_data = max_radius(mask_tc, x0)

    length = float(np.ceil(r_wt_data))
    d_grid = np.linspace(0.1, 1.0, 10) if d_grid is None else np.asarray(d_grid, dtype=float)
    rho_grid = np.linspace(0.01, 0.1, 10) if rho_grid is None else np.asarray(rho_grid, dtype=float)

    best: CharacteristicScaling | None = None
    best_score = np.inf

    for d in d_grid:
        for rho in rho_grid:
            t_char = length / np.sqrt(d * rho)
            r, u = solve_radial(
                d=float(d),
                rho=float(rho),
                t_final=float(t_char),
                h_init=h_init,
                r_init=r_init,
                domain=domain,
                xdim=xdim,
            )
            r_wt_sim = _threshold_radius(r, u, th_wt)
            r_tc_sim = _threshold_radius(r, u, th_tc)
            if np.isnan(r_wt_sim) or np.isnan(r_tc_sim):
                continue

            score = abs(r_wt_data - r_wt_sim) / (r_wt_data + 1e-3) + abs(r_tc_data - r_tc_sim) / (
                r_tc_data + 1e-3
            )
            if score < best_score:
                best_score = float(score)
                best = CharacteristicScaling(
                    d_char=float(d),
                    rho_char=float(rho),
                    length=length,
                    time=float(t_char),
                    r_wt_data=float(r_wt_data),
                    r_tc_data=float(r_tc_data),
                    r_wt_sim=float(r_wt_sim),
                    r_tc_sim=float(r_tc_sim),
                    score=float(score),
                )

    if best is None:
        raise RuntimeError("no grid point reproduced both threshold radii; widen the search grid")
    return best
