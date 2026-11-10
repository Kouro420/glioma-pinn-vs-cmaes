#!/usr/bin/env python
"""Check that our forward solver reproduces the reference implementation.

The CMA-ES baseline must invert the same forward model the PINN was built
around. Our solver (``src/models/fisher_kpp.py``) is an independent
implementation, so this test pins it against
``external/pinngbm-torch/solve_fisher_kpp_torch.py`` on a small phantom: same
tissue maps, same phase field, same tumor seed position, same end time, 1 mm voxels.

Agreement is required to 1e-4 in the maximum absolute difference. The tolerance
is not zero because the two implementations sum the same float32 terms in a
different order; it is far below the differences the calibration resolves.

The agreement is checked at five points, not one: the four corners of the
admissible ``(mu_d, mu_rho)`` box and the shared starting point. A search that
can reach a corner needs the two implementations to agree there too. The second
test measures the 1 mm versus 0.5 mm discretization gap, which the synthetic
design depends on and would otherwise only assume.

The result of this test is the evidence that the two arms share one forward
model. Run it with::

    pytest tests/test_solver_agreement.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "external" / "pinngbm-torch"

sys.path.insert(0, str(REPO_ROOT))

from src.models.fisher_kpp import gaussian_seed, phase_field, solve_fisher_kpp  # noqa: E402

TOLERANCE = 1e-4


def _phantom(shape=(48, 48, 40), seed: int = 0):
    """A small brain-like phantom: white core, grey rim, CSF outside."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    centre = (np.array(shape) - 1) / 2.0
    radius = np.linalg.norm(grid - centre, axis=-1)

    inner = np.clip(1.0 - radius / (0.55 * min(shape)), 0.0, 1.0)
    p_wm = (inner**2).astype(np.float32)
    p_gm = (inner - inner**2).astype(np.float32)
    p_csf = (1.0 - inner).astype(np.float32)

    jitter = rng.uniform(0.0, 0.02, size=shape).astype(np.float32)
    return p_wm + jitter, p_gm, p_csf


# The four corners of the calibrated box, plus its center. Both arms search
# mu_d, mu_rho in [0, 2] as multipliers on a characteristic (d_w, rho), so
# agreement at the starting point alone says nothing about agreement where the
# search actually goes -- and a CMA-ES population at sigma0 = 0.25 reaches these
# corners within a few generations. 0.1 rather than 0.0, because mu = 0 removes
# the term entirely and tests a degenerate problem neither arm ever solves.
BASE_D_W, BASE_RHO = 0.3, 0.05
BOX_CORNERS = [
    pytest.param(0.1, 0.1, id="mu_d=0.1,mu_rho=0.1"),
    pytest.param(0.1, 2.0, id="mu_d=0.1,mu_rho=2.0"),
    pytest.param(2.0, 0.1, id="mu_d=2.0,mu_rho=0.1"),
    pytest.param(2.0, 2.0, id="mu_d=2.0,mu_rho=2.0"),
    pytest.param(1.0, 1.0, id="mu_d=1.0,mu_rho=1.0 (the shared start)"),
]


@pytest.mark.skipif(
    not (REFERENCE_DIR / "solve_fisher_kpp_torch.py").is_file(),
    reason="the reference PINN is missing; run scripts/fetch_reference_pinn.sh",
)
@pytest.mark.parametrize("mu_d, mu_rho", BOX_CORNERS)
def test_matches_reference_solver(mu_d: float, mu_rho: float) -> None:
    """Our solution and the reference solution agree across the search box.

    One case per corner of the admissible (mu_d, mu_rho) region and one at the
    shared starting point, all at 1 mm voxels. Our paper claims that both
    arms solve the same forward problem; that claim has to hold
    wherever the search can go, not only where it begins.
    """
    sys.path.insert(0, str(REFERENCE_DIR))
    from solve_fisher_kpp_torch import solve_fisher_kpp_torch  # noqa: E402

    p_wm, p_gm, p_csf = _phantom()
    phi = phase_field(p_wm, p_gm, p_csf, n_iter=20, epsilon=3.0, device="cpu")
    u0 = gaussian_seed(p_wm.shape, (np.array(p_wm.shape) - 1) / 2.0, h_init=0.5, r_init=0.01)

    d_w, rho, t_final = BASE_D_W * mu_d, BASE_RHO * mu_rho, 25.0

    ours = solve_fisher_kpp(
        p_wm,
        p_gm,
        phi,
        u0,
        d_w=d_w,
        rho=rho,
        t_final=t_final,
        factor=10.0,
        n_save=2,
        h=1.0,
        device="cpu",
    )
    reference, _ = solve_fisher_kpp_torch(
        p_wm,
        p_gm,
        dw=d_w,
        factor=10.0,
        phi=phi,
        rho=rho,
        tfinal=t_final,
        u0=u0,
        Nt=2,
        device="cpu",
    )

    difference = np.max(np.abs(ours.final - reference[..., -1]))
    print(f"mu_d={mu_d} mu_rho={mu_rho}: max |difference| = {difference:.3e}")
    assert difference < TOLERANCE, (
        f"max |difference| = {difference:.3e}, tolerance {TOLERANCE:.0e} "
        f"at mu_d={mu_d}, mu_rho={mu_rho}"
    )


def _upsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor refinement, matching src.data.make_synthetic."""
    for axis in range(volume.ndim):
        volume = np.repeat(volume, factor, axis=axis)
    return volume


def _block_mean(volume: np.ndarray, factor: int) -> np.ndarray:
    """Average a fine grid back onto the coarse one, as the synthetic truth is."""
    shape = []
    for size in volume.shape:
        shape.extend([size // factor, factor])
    return volume.reshape(shape).mean(axis=tuple(range(1, 2 * volume.ndim, 2)))


def test_refinement_gap_is_measured() -> None:
    """Measure the 1 mm discretization error against a 0.5 mm solve.

    The synthetic truth is generated at ``fine_voxel_mm: 0.5`` and averaged to
    1 mm, so the calibrated solver never sees the grid the truth was made on.
    This test exists to put a number on that gap, because the alternative is
    to assume it is negligible.

    The tolerance here is deliberately loose: the point is to report
    the gap, not to require it to be small. A failure means the two grids
    disagree so badly that the synthetic study's premise is in question.
    """
    factor = 2
    p_wm, p_gm, p_csf = _phantom(shape=(32, 32, 28))
    centre = (np.array(p_wm.shape) - 1) / 2.0

    phi = phase_field(p_wm, p_gm, p_csf, n_iter=20, epsilon=3.0, h=1.0, device="cpu")
    u0 = gaussian_seed(p_wm.shape, centre, h_init=0.5, r_init=0.01, h=1.0)
    coarse = solve_fisher_kpp(
        p_wm, p_gm, phi, u0, d_w=0.3, rho=0.05, t_final=20.0, h=1.0, device="cpu"
    )
    assert np.isfinite(coarse.final).all()
    assert 0.0 <= coarse.final.min() <= coarse.final.max() <= 1.0
    assert coarse.n_steps > 0

    # The fine solve keeps epsilon and the tumor seed in millimeters, and passes the
    # physical spacing as h, exactly as make_synthetic does.
    fine_wm = _upsample(p_wm, factor)
    fine_gm = _upsample(p_gm, factor)
    fine_csf = _upsample(p_csf, factor)
    h_fine = 1.0 / factor
    fine_phi = phase_field(
        fine_wm, fine_gm, fine_csf, n_iter=20, epsilon=3.0, h=h_fine, device="cpu"
    )
    fine_u0 = gaussian_seed(fine_wm.shape, centre, h_init=0.5, r_init=0.01, h=h_fine)
    fine = solve_fisher_kpp(
        fine_wm, fine_gm, fine_phi, fine_u0,
        d_w=0.3, rho=0.05, t_final=20.0, h=h_fine, device="cpu",
    )
    assert np.isfinite(fine.final).all()

    averaged = _block_mean(fine.final, factor)
    assert averaged.shape == coarse.final.shape

    worst = float(np.max(np.abs(averaged - coarse.final)))
    mean = float(np.mean(np.abs(averaged - coarse.final)))
    print(f"discretisation gap, 0.5 mm averaged vs 1 mm: max {worst:.4f}, mean {mean:.4f}")

    # Loose on purpose: this reports, it does not police. 0.25 in u would mean
    # the two grids disagree about the mask a threshold would produce.
    assert worst < 0.25, (
        f"1 mm and 0.5 mm solutions differ by {worst:.3f} in u, which is large "
        "enough to affect a thresholded mask -- the synthetic design assumes "
        "this gap is small, so investigate before reporting any synthetic result"
    )
