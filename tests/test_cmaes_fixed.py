"""CMA-ES with parameters held fixed, on a fake forward model.

The solver and the objective are replaced by a quadratic, so the test checks the
bookkeeping of src/training/cmaes_calibration.py, not the physics.
"""

from types import SimpleNamespace

import cma
import numpy as np
import pytest

import src.training.cmaes_calibration as cc

OPTIMUM = {"mu_d": 0.6, "mu_rho": 1.3, "th_wt": 0.25, "th_tc": 0.6}


@pytest.fixture
def fake_model(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(cc, "solve_fisher_kpp", lambda *a, **k: SimpleNamespace(final=None))

    def objective(final, **params):
        seen.append(params)
        total = sum((params[k] - OPTIMUM[k]) ** 2 for k in OPTIMUM)
        return SimpleNamespace(total=total, as_dict=lambda: {})

    scaling = SimpleNamespace(to_physical=lambda d, r: (d, r), infiltration_length=lambda d, r: 1.0)
    return objective, scaling, seen


def run(objective, scaling, **settings):
    return cc.calibrate(p_wm=None, p_gm=None, phi=None, u0=None, objective=objective,
                        scaling=scaling, settings=cc.CalibrationSettings(seed=1, **settings))


def test_fixed_thresholds_are_never_sampled(fake_model):
    objective, scaling, seen = fake_model
    fixed = {"th_wt": 0.25, "th_tc": 0.6}
    result = run(objective, scaling, fixed=fixed)
    assert len(seen) == result.n_evaluations > 100
    assert all(p["th_wt"] == 0.25 and p["th_tc"] == 0.6 for p in seen)
    assert result.best_parameters["th_wt"] == 0.25
    assert abs(result.best_parameters["mu_d"] - 0.6) < 1e-3
    assert abs(result.best_parameters["mu_rho"] - 1.3) < 1e-3
    # Evaluation 0 is the start point with the fixed values.
    assert seen[0] == {"mu_d": 1.0, "mu_rho": 1.0, "th_wt": 0.25, "th_tc": 0.6}


def test_nothing_fixed_is_the_plain_four_parameter_search(fake_model):
    """Same random seed, same options: the points match a plain pycma run point for point."""
    objective, scaling, seen = fake_model
    run(objective, scaling, max_evaluations=40)
    s = cc.CalibrationSettings(seed=1, max_evaluations=40)
    lower, upper = np.asarray(s.lower), np.asarray(s.upper)
    es = cma.CMAEvolutionStrategy(cc._to_unit(np.asarray(s.initial), lower, upper).tolist(),
                                  s.sigma0, {"bounds": [0.0, 1.0], "popsize": s.popsize,
                                             "maxfevals": s.max_evaluations, "tolfun": s.tol_fun,
                                             "seed": s.seed, "verbose": -9})
    expected = [dict(zip(cc.PARAMETER_NAMES, s.initial))]
    while not es.stop():
        candidates = es.ask()
        points = [cc._from_unit(np.asarray(c), lower, upper) for c in candidates]
        expected += [dict(zip(cc.PARAMETER_NAMES, map(float, p))) for p in points]
        es.tell(candidates, [sum((p[i] - OPTIMUM[n]) ** 2 for i, n in enumerate(cc.PARAMETER_NAMES))
                             for p in points])
    assert seen == expected


def test_bad_fixed_values_raise(fake_model):
    objective, scaling, _ = fake_model
    with pytest.raises(ValueError, match="outside"):
        run(objective, scaling, fixed={"th_wt": 1.5})
    with pytest.raises(ValueError, match="unknown"):
        run(objective, scaling, fixed={"x0": 1.0})
    with pytest.raises(ValueError, match="two free"):
        run(objective, scaling, fixed={"th_wt": 0.3, "th_tc": 0.6, "mu_rho": 1.0})


def test_x1_configs_differ_from_base_only_where_intended():
    import json
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    load = lambda name: yaml.safe_load((repo / "configs" / name).read_text(encoding="utf-8"))
    base = load("base.yaml")
    for name, x0_file in [("x1_thresholds_fixed.yaml", None),
                          ("x1_thresholds_seed_fixed.yaml", "configs/x1_generator_seeds.json")]:
        cfg = load(name)
        assert cfg["cmaes"].pop("fix_thresholds") == "truth"
        assert cfg["cmaes"]["seed"] == 1
        cfg["cmaes"]["seed"] = base["cmaes"]["seed"]
        assert cfg["initial_condition"].pop("x0_file", None) == x0_file
        assert cfg == base, name
    seeds = json.loads((repo / "configs" / "x1_generator_seeds.json").read_text(encoding="utf-8"))
    assert len(seeds["cases"]) == 16
