"""The three mask terms of the re-score (a = 20, a = 200 and hard thresholds)."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.b1_b2_rescore import mask_term  # noqa: E402
from src.training.objective import SegmentationObjective, ObjectiveWeights  # noqa: E402


def test_a20_term_equals_the_shared_objective():
    rng = np.random.default_rng(0)
    u = rng.random((6, 6, 6)).astype(np.float32)
    mw = (rng.random((6, 6, 6)) > 0.6).astype(np.float32)
    mt = (rng.random((6, 6, 6)) > 0.8).astype(np.float32)
    phi = rng.random((6, 6, 6)).astype(np.float32)
    obj = SegmentationObjective(mw, mt, phi, prior_th_wt=0.35, prior_th_tc=0.7,
                                weights=ObjectiveWeights(reg_mu_d=0, reg_mu_rho=0, reg_th_wt=0, reg_th_tc=0))
    terms = obj(u, mu_d=1.0, mu_rho=1.0, th_wt=0.3, th_tc=0.6)
    assert np.isclose(mask_term(u, mw, phi, 0.3, float(np.mean(mw ** 2)), "a20"), terms.seg_wt)
    assert np.isclose(mask_term(u, mt, phi, 0.6, float(np.mean(mt ** 2)), "a20"), terms.seg_tc)


def test_hard_term_counts_mismatched_voxels():
    u = np.array([0.1, 0.5, 0.9, 0.9], dtype=np.float32)
    mask = np.array([0, 1, 1, 0], dtype=np.float32)
    phi = np.ones(4, dtype=np.float32)
    # prediction 1{u > 0.4} = [0, 1, 1, 1]: one mismatch of four, normalized by mean(mask^2) = 0.5
    assert np.isclose(mask_term(u, mask, phi, 0.4, 0.5, "hard"), 0.25 / 0.5)


def test_a200_is_closer_to_hard_than_a20():
    u = np.linspace(0, 1, 101, dtype=np.float32)
    mask = (u > 0.5).astype(np.float32)
    phi = np.ones_like(u)
    terms = {m: mask_term(u, mask, phi, 0.5, float(np.mean(mask ** 2)), m) for m in ("a20", "a200", "hard")}
    assert terms["hard"] < terms["a200"] < terms["a20"]
