"""Both paired tests of scripts/paired_tests.py: the exact sign test and the Wilcoxon signed-rank test."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paired_tests as pt  # noqa: E402


def test_sign_test_matches_the_reported_values():
    assert pt.sign_test(12, 4) == pytest.approx(0.0768, abs=5e-5)   # matched comparison, |mu_rho error|, Figure S3
    assert pt.sign_test(15, 5) == pytest.approx(0.0414, abs=5e-5)   # Dice WT, 20 cases
    assert pt.sign_test(16, 0) == pytest.approx(3.05e-5, rel=1e-2)
    assert pt.sign_test(0, 0) == 1.0


def test_ties_are_dropped_in_both_tests():
    t = pt.test_pair([0.1, 0.2, 0.3, 0.0, -0.05, 0.4])
    assert (t["n"], t["cmaes_better"], t["pinn_better"], t["ties"]) == (6, 4, 1, 1)
    assert t["p_sign"] == pt.sign_test(4, 1)
    assert 0 < t["p_wilcoxon"] <= 1


@pytest.mark.skipif(not (REPO / "outputs" / "runs_syn_np_paper").is_dir(),
                    reason="needs the run records; get them with python scripts/fetch_records.py")
def test_matched_comparison_mu_rho_sign_and_wilcoxon_disagree():
    entry = pt.compare(REPO / "outputs" / "runs_syn_np_paper", "pinn", REPO / "outputs" / "runs_syn_np")
    rho = entry["metrics"]["err_mu_rho"]
    assert (rho["cmaes_better"], rho["pinn_better"]) == (12, 4)
    assert rho["p_sign"] == pytest.approx(0.0768, abs=5e-5)
    assert rho["p_wilcoxon"] == pytest.approx(0.018, abs=5e-4)
