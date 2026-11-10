#!/usr/bin/env python
"""Fisher-Kolmogorov forward solver on the voxel grid, in PyTorch.

This is the shared forward model of this repository. It is used by:

* the CMA-ES baseline, which evaluates it once per candidate parameter set,
* the synthetic-truth generator, which runs it at half the voxel size (see
  ``src/data/make_synthetic.py``) so that the test data are not produced by the
  exact discretization the baseline later inverts.

Equation solved, the diffuse-domain form of Zhang et al. (2025), Eq. (3), without the
phi on the time derivative (their Eq. (3) and the release's PINN residual have d(phi u)/dt):

    du/dt = div( D(x) phi grad u ) + rho phi u (1 - u)
          = D_phi lap(u) + grad(D_phi) . grad(u) + rho phi u (1 - u)

with the tissue-weighted diffusivity

    D(x) = d_w ( P_wm(x) + P_gm(x) / factor ),    factor = D_w / D_g = 10,

and the phase field ``phi`` carrying the brain boundary, so that no explicit
no-flux condition on the irregular brain surface is needed. The computational
box uses homogeneous Dirichlet edges, which is admissible because the box
boundary is far from ``phi > 0`` (Zhang et al. 2025, Sec. 2.1.2).

Units. ``h`` is the voxel size in mm; ``d_w`` is in mm^2 per unit of ``t_final``
and ``rho`` in 1 per unit of ``t_final``. When the non-dimensional convention of
``src/models/characteristic.py`` is used, ``t_final = 1`` and the two parameters
are the products D*t_end and rho*t_end. From a single imaging time point those
products, not D and rho separately, are what the data constrain.

Time stepping is explicit Euler with a CFL-limited step, matching the reference
implementation in ``external/pinngbm-torch/solve_fisher_kpp_torch.py``;
``tests/test_solver_agreement.py`` checks that both agree at h = 1 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "SolverResult",
    "diffusion_field",
    "phase_field",
    "gaussian_seed",
    "solve_fisher_kpp",
]


@dataclass
class SolverResult:
    """Solution snapshots of one forward solve.

    Attributes:
        u: array of shape ``(*grid, n_save)`` with the density at ``t``.
        t: the save times, ``t[0] = 0`` and ``t[-1] = t_final``.
        dt: the CFL time step actually used.
        n_steps: number of explicit Euler steps taken.
    """

    u: np.ndarray
    t: np.ndarray
    dt: float
    n_steps: int

    @property
    def final(self) -> np.ndarray:
        """Density at ``t_final``."""
        return self.u[..., -1]


def _as_tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def diffusion_field(
    p_wm,
    p_gm,
    d_w: float,
    factor: float = 10.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Tissue-weighted diffusivity ``D = d_w (P_wm + P_gm / factor)``.

    ``factor`` is the white-to-gray ratio D_w / D_g, fixed at 10 in every run.
    CSF carries no tissue probability, so ``D`` vanishes there by construction.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor!r}")
    device = device or torch.device("cpu")
    wm = _as_tensor(p_wm, device, dtype)
    gm = _as_tensor(p_gm, device, dtype)
    return d_w * (wm + gm / factor)


def phase_field(
    p_wm,
    p_gm,
    p_csf,
    *,
    n_iter: int = 100,
    epsilon: float = 3.0,
    h: float = 1.0,
    tau: float = 1e-3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Smooth indicator of the brain domain, by Cahn-Hilliard relaxation.

    The sharp set ``{P_wm + P_gm > max(0.1, P_csf)}`` is relaxed for a short
    time so that its boundary becomes a transition layer of width ``epsilon``
    (3 mm in Zhang et al. 2025, Sec. 2.1.2). The level set ``phi = 0.5``
    approximates the brain surface.
    """
    device = device or torch.device("cpu")
    wm = _as_tensor(p_wm, device, dtype)
    gm = _as_tensor(p_gm, device, dtype)
    csf = _as_tensor(p_csf, device, dtype)

    dim = wm.ndim
    if dim not in (2, 3):
        raise ValueError(f"expected 2D or 3D tissue maps, got {dim}D")

    floor = torch.full_like(csf, 0.1)
    phi = ((wm + gm) > torch.maximum(floor, csf)).to(dtype)

    dt = h**4 / (16.0 * epsilon)
    eps_sq = epsilon**2
    inv_h2 = 1.0 / (h * h)
    inv_2h = 0.5 / h

    for _ in range(n_iter):
        lap = (-2.0 * dim) * phi
        for axis in range(dim):
            lap = lap + torch.roll(phi, -1, dims=axis) + torch.roll(phi, 1, dims=axis)
        lap = lap * inv_h2

        mu = 0.5 * phi * (1.0 - phi) * (1.0 - 2.0 * phi) - eps_sq * lap
        mobility = phi * (1.0 - phi) + tau

        dphi = torch.zeros_like(phi)
        for axis in range(dim):
            grad_mu = (torch.roll(mu, -1, dims=axis) - torch.roll(mu, 1, dims=axis)) * inv_2h
            flux = mobility * grad_mu
            dphi = dphi + (torch.roll(flux, -1, dims=axis) - torch.roll(flux, 1, dims=axis)) * inv_2h

        phi = torch.clamp(phi + dt * dphi, 0.0, 1.0)

    return phi.cpu().numpy()


def gaussian_seed(
    shape: Sequence[int],
    x0: Sequence[float],
    *,
    h_init: float,
    r_init: float,
    h: float = 1.0,
) -> np.ndarray:
    """Initial condition ``u_0(x) = h_init exp(-r_init |x - x0|^2)``.

    ``x0`` is given in voxel index coordinates of the *1 mm* grid and ``h`` is
    the voxel size of the grid being built, so the same initial condition can be
    placed on a refined grid. Distances enter in mm, as in Zhang et al. (2025), p. 3.
    """
    if len(shape) != len(x0):
        raise ValueError(f"x0 has {len(x0)} entries but the grid is {len(shape)}D")
    axes = [np.arange(n, dtype=np.float64) * h for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    dist_sq = np.zeros(tuple(shape), dtype=np.float64)
    for coord, centre in zip(mesh, x0):
        dist_sq += (coord - float(centre)) ** 2
    return (h_init * np.exp(-r_init * dist_sq)).astype(np.float32)


def _apply_dirichlet(u: torch.Tensor) -> torch.Tensor:
    """Zero the outermost layer of the computational box."""
    if u.ndim == 2:
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
    else:
        u[0, :, :] = 0.0
        u[-1, :, :] = 0.0
        u[:, 0, :] = 0.0
        u[:, -1, :] = 0.0
        u[:, :, 0] = 0.0
        u[:, :, -1] = 0.0
    return u


def _central_gradient(field: torch.Tensor, inv_2h: float) -> list[torch.Tensor]:
    """Central differences of ``field`` along every axis, zero at the edges."""
    grads = []
    for axis in range(field.ndim):
        g = torch.zeros_like(field)
        upper = [slice(None)] * field.ndim
        lower = [slice(None)] * field.ndim
        inner = [slice(None)] * field.ndim
        upper[axis] = slice(2, None)
        lower[axis] = slice(None, -2)
        inner[axis] = slice(1, -1)
        g[tuple(inner)] = (field[tuple(upper)] - field[tuple(lower)]) * inv_2h
        grads.append(g)
    return grads


def solve_fisher_kpp(
    p_wm,
    p_gm,
    phi,
    u0,
    *,
    d_w: float,
    rho: float,
    t_final: float,
    factor: float = 10.0,
    n_save: int = 2,
    h: float = 1.0,
    cfl_safety: float = 0.99,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    keep_on_gpu: bool = False,
) -> SolverResult:
    """Integrate the Fisher-Kolmogorov model to ``t_final``.

    Args:
        p_wm, p_gm: white- and gray-matter probability maps, 2D or 3D.
        phi: phase field of the brain domain, same shape.
        u0: initial density, same shape, values in [0, 1].
        d_w: white-matter diffusivity, mm^2 per unit time.
        rho: proliferation rate, 1 per unit time.
        t_final: end time, in the same time unit as ``d_w`` and ``rho``.
        factor: white-to-gray diffusivity ratio, 10 in every run.
        n_save: number of snapshots returned, including ``t = 0``; use 2 when
            only the final state is needed, which is the CMA-ES case.
        h: voxel size in mm.
        cfl_safety: fraction of the stability limit used for the time step.
        device: torch device; defaults to CUDA when available.
        keep_on_gpu: keep snapshots on the GPU instead of copying to host.

    Returns:
        A :class:`SolverResult`. ``u`` is clamped to [0, 1] at every step.

    Raises:
        ValueError: if the shapes disagree or ``n_save`` is below 2.
    """
    if n_save < 2:
        raise ValueError(f"n_save must be at least 2, got {n_save}")
    if t_final <= 0:
        raise ValueError(f"t_final must be positive, got {t_final}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    wm = _as_tensor(p_wm, device, dtype)
    gm = _as_tensor(p_gm, device, dtype)
    phi_t = _as_tensor(phi, device, dtype)
    u = _as_tensor(u0, device, dtype).clone()

    if not (wm.shape == gm.shape == phi_t.shape == u.shape):
        raise ValueError(
            "shape mismatch: "
            f"p_wm {tuple(wm.shape)}, p_gm {tuple(gm.shape)}, "
            f"phi {tuple(phi_t.shape)}, u0 {tuple(u.shape)}"
        )
    dim = u.ndim
    if dim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D grid, got {dim}D")

    inv_h2 = 1.0 / (h * h)
    inv_2h = 0.5 / h

    d_phi = (d_w * (wm + gm / factor)) * phi_t
    grad_d_phi = _central_gradient(d_phi, inv_2h)

    max_diff = float(torch.max(d_phi).item())
    if max_diff <= 0.0:
        raise ValueError("the diffusion field is zero everywhere; check the tissue maps")
    dt = cfl_safety * (h * h) / (2.0 * dim * max_diff)

    t_save = np.linspace(0.0, t_final, n_save)
    store_device = device if keep_on_gpu else torch.device("cpu")
    u_all = torch.zeros(tuple(u.shape) + (n_save,), dtype=dtype, device=store_device)

    u = _apply_dirichlet(u)
    u_all[..., 0] = u.to(store_device)

    interior = (slice(1, -1),) * dim
    t = 0.0
    save_idx = 1
    n_steps = 0

    while save_idx < n_save:
        target = float(t_save[save_idx])
        step = min(dt, target - t)

        u_in = u[interior]
        lap = -2.0 * dim * u_in
        for axis in range(dim):
            upper = list(interior)
            lower = list(interior)
            upper[axis] = slice(2, None)
            lower[axis] = slice(None, -2)
            lap = lap + u[tuple(upper)] + u[tuple(lower)]
        lap = lap * inv_h2

        advection = torch.zeros_like(u_in)
        for axis in range(dim):
            upper = list(interior)
            lower = list(interior)
            upper[axis] = slice(2, None)
            lower[axis] = slice(None, -2)
            grad_u = (u[tuple(upper)] - u[tuple(lower)]) * inv_2h
            advection = advection + grad_d_phi[axis][interior] * grad_u

        reaction = rho * phi_t[interior] * u_in * (1.0 - u_in)
        u[interior] = u_in + step * (d_phi[interior] * lap + advection + reaction)
        u.clamp_(0.0, 1.0)

        t += step
        n_steps += 1
        if t >= target - 1e-12:
            t = target
            u_all[..., save_idx] = u.to(store_device)
            save_idx += 1

    u_out = u_all if keep_on_gpu else u_all.numpy()
    return SolverResult(u=u_out, t=t_save, dt=dt, n_steps=n_steps)
