"""The CMA-ES configs of the seed checks (tumor seed position, CMA-ES random seed) differ from their
baselines only where intended."""

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))


def diff(x, y, prefix=""):
    out = []
    for key in sorted(set(x) | set(y)):
        if isinstance(x.get(key), dict) and isinstance(y.get(key), dict):
            out += diff(x[key], y[key], prefix + key + ".")
        elif x.get(key) != y.get(key):
            out.append(prefix + key)
    return out


def test_seed_check_configs_differ_from_their_baselines_only_where_intended():
    # Compared with the refit with the thresholds and the tumor seed position fixed. Against the
    # baseline's own run record only the random seed (cmaes.seed) differs; configs/e1_both.yaml
    # is the generation config and also differs in noise.
    assert diff(load("e1_both_seed1.yaml"), load("x1_thresholds_seed_fixed.yaml")) == [
        "cmaes.fix_thresholds", "initial_condition.x0_file"]
    assert diff(load("x1_seed_fixed_thresholds_free.yaml"),
                load("x1_thresholds_seed_fixed.yaml")) == ["cmaes.fix_thresholds"]
    assert diff(load("env_generator_seed.yaml"), load("env_cases.yaml")) == ["initial_condition.x0_file"]


@pytest.mark.skipif(not (REPO / "outputs" / "synthetic_env" / "cases_synthetic.txt").is_file(),
                    reason="the design-range cases are not generated yet (docs/REPRODUCE.md, section A)")
def test_env_seed_file_covers_every_kept_case():
    seeds = json.loads((REPO / "configs" / "env_generator_seeds.json").read_text(encoding="utf-8"))["cases"]
    kept = (REPO / "outputs" / "synthetic_env" / "cases_synthetic.txt").read_text(encoding="utf-8").split()
    assert len(kept) == 12 and all(case in seeds for case in kept)
