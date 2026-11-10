"""The no-data test: mask weights to zero, early stopping logged but not applied in fine-tuning."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.pinn_early_stopping import EarlyStopRule
from experiments.run_pinn import _scale_seg_weights

REPO = Path(__file__).resolve().parents[1]
NO_REFERENCE = not (REPO / "external" / "pinngbm-torch" / "Options.py").is_file()


def test_seg_scale_touches_only_the_two_mask_weights():
    w = "res,1.0,seg1,1e-05,seg2,1e-05,rD_reg,0.001,th1_reg,0.001"
    assert _scale_seg_weights(w, 0.0) == "res,1.0,seg1,0.0,seg2,0.0,rD_reg,0.001,th1_reg,0.001"


def test_stop_stages_default_and_from_config():
    assert EarlyStopRule().stop_stages == ("init", "inv")
    assert EarlyStopRule.from_config({"patience": 1000, "stop_stages": ["init"]}).stop_stages == ("init",)


@pytest.mark.skipif(NO_REFERENCE, reason="the reference PINN clone is missing (scripts/fetch_reference_pinn.sh)")
def test_nodata_config_differs_from_the_paper_schedule_only_where_intended():
    from experiments.run_pinn import _build_options  # imports the reference Options

    def load(name):
        return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))

    base, nodata = load("paper_schedule.yaml"), load("paper_schedule_nodata.yaml")
    assert nodata["pinn"].pop("seg_weight_scale") == 0.0
    assert nodata["pinn"]["early_stopping"].pop("stop_stages") == ["init"]
    assert nodata == base
    paths = SimpleNamespace(tumor_seg="s", wm="w", gm="g", csf="c", brain_mask=None, recurrence=None)
    w = _build_options(load("paper_schedule_nodata.yaml"), paths, "cpu", prior_scale=0.0).opts[
        "train_opts"]["weights_inv"]
    assert w["seg1"] == 0.0 and w["seg2"] == 0.0 and w["res"] == 1.0
