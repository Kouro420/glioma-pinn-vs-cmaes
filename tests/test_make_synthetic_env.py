"""Design-range case options of src/data/make_synthetic.py, and the base set built without them."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.data.make_synthetic import default_truths

REPO = Path(__file__).resolve().parents[1]
SCALINGS = {s: SimpleNamespace(d_char=0.9, rho_char=0.04, time=121.2) for s in ("RHUH-0001", "RHUH-0002")}


def test_base_set_ids_and_targets_unchanged():
    truths = default_truths(SCALINGS)
    assert [t.case_id for t in truths[:3]] == [
        "SYN-RHUH-0001-d0.2-r0.6", "SYN-RHUH-0001-d0.2-r1.3", "SYN-RHUH-0001-d0.4-r0.6"]
    assert len(truths) == 16
    assert {(t.th_wt, t.th_tc) for t in truths if t.mu_rho_target == 0.6} == {(0.25, 0.60)}


def test_env_config_builds_16_cases_inside_the_design_range():
    config = yaml.safe_load((REPO / "configs" / "env_cases.yaml").read_text(encoding="utf-8"))
    syn = config["synthetic"]
    truths = default_truths(SCALINGS, mu_ds=tuple(syn["mu_ds"]), mu_rhos=tuple(syn["mu_rhos"]),
                            case_prefix=syn["case_prefix"], noise_sigma=syn["noise_sigma"],
                            fine_voxel_mm=syn["fine_voxel_mm"])
    assert len(truths) == 16
    assert all(t.case_id.startswith("ENV-RHUH-000") for t in truths)
    lo, hi = syn["keep_truth_box"]
    assert (lo, hi) == (0.75, 1.25)
    assert all(lo < t.mu_d_target < hi and lo < t.mu_rho_target < hi for t in truths)
    # Same generator settings as the base set: noise, grid, thresholds tied to mu_rho.
    assert (syn["noise_sigma"], syn["fine_voxel_mm"]) == (0.05, 0.5)
    assert {(t.th_wt, t.th_tc) for t in truths} == {(0.25, 0.60), (0.45, 0.80)}
    assert config["cmaes"]["seed"] >= 1
