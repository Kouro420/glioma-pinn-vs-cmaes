#!/usr/bin/env python
"""Run the reference PINN on one case and write the same result record.

The PINN arm is the published method of Zhang et al. (2025), executed from the
authors' code in ``external/pinngbm-torch`` (commit 99af20f). Nothing of their
method is reimplemented here; this script only

* pins the settings that the code leaves at values differing from Zhang et al.,
* times the whole pipeline, including the grid search and the pre-training,
* scores the resulting density with the same metrics the baseline uses,
* records the same provenance fields, so the two arms pool cleanly.

Pinned deviations, all recorded in the output file: the dense network is used
(``separable: false``) rather than the separable default, the start value of the
core threshold is the code's 0.7 rather than the 0.6 of Zhang et al., and the
optimizer is Adam alone rather than Adam followed by L-BFGS-B.

With ``configs/paper_schedule.yaml`` the step budget of Zhang et al. and the
original TensorFlow code's early stopping are used instead of the fixed
4000 + 1000 steps; see ``experiments/pinn_early_stopping.py``.
With ``configs/paper_schedule_lbfgs_full.yaml`` the original code's L-BFGS stage
follows Adam in pre-training; see ``experiments/pinn_lbfgs.py``.
With ``configs/paper_schedule_reach.yaml`` (the reach test) rD and rRHO get
their own Adam rate in fine-tuning; see ``experiments/pinn_param_lr.py``. That
is our modification, not part of the published method, and the record says so
under ``provenance.param_lr``.

Usage::

    python -m experiments.run_pinn --case RHUH-0001 \
        --data-root src/data/predict_gbm/predict_gbm --out outputs/runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from experiments.run_cmaes import device_facts, provenance, synthetic_paths
from src.data.predict_gbm import find_case, load_case
from src.evaluation.metrics import (
    ParameterTruth,
    coverage_metrics,
    mu_errors,
    parameter_errors,
    segmentation_dice,
)

METHOD = "pinn"
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "external" / "pinngbm-torch"

STEP_PATTERN = re.compile(r"^\[Step (\d+)\]")
LOSS_PATTERN = re.compile(r"(\w+)=([-\d.eE+]+)")


class StepRecorder:
    """Stamp every printed training step with elapsed seconds.

    The reference trainer prints lines such as::

        [Step 250] res=0.0004 seg1=0.42 ... rD=0.954 rRHO=0.975 th1=0.35 th2=0.70

    but with no time, so the PINN arm would have no cost curve while the CMA-ES
    arm has one at every improvement. This wrapper sits in front of stdout,
    keeps the text untouched for the log, and stores one record per printed
    step.
    """

    def __init__(self, stream, started: float) -> None:
        self._stream = stream
        self._started = started
        self._buffer = ""
        self.history: list[dict] = []

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._capture(line)
        return len(text)

    def _capture(self, line: str) -> None:
        match = STEP_PATTERN.match(line.strip())
        if match is None:
            return
        entry = {
            "step": int(match.group(1)),
            "elapsed_s": time.perf_counter() - self._started,
        }
        for key, value in LOSS_PATTERN.findall(line):
            if key == "Step":
                continue
            try:
                entry[key] = float(value)
            except ValueError:
                continue
        self.history.append(entry)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):  # delegate isatty, encoding, fileno, ...
        return getattr(self._stream, name)


def _patch_scalar_conversion(engine_module) -> None:
    """Make the reference forward solver accept 1-element arrays.

    At the final prediction the reference passes the inferred parameters into
    ``solve_fisher_kpp_torch`` still wrapped as 1x1 arrays, which that function
    converts with ``float(dw)``. NumPy 1.x allowed this; NumPy 2.x raises

        TypeError: only 0-dimensional arrays can be converted to Python scalars

    so the run would fail after training. The solver is therefore wrapped here,
    in our own script: the numeric arguments are flattened to Python floats and
    the original function is called unchanged. Nothing in ``external/pinngbm-torch`` is edited, and the
    substitution is recorded in the result file.
    """

    original = engine_module.solve_fisher_kpp_torch

    def as_scalar(value):
        if isinstance(value, (int, float)):
            return float(value)
        return float(np.asarray(value).reshape(-1)[0])

    def patched(Pwm, Pgm, dw, factor, phi, rho, tfinal, u0, Nt, device=None):
        return original(
            Pwm,
            Pgm,
            as_scalar(dw),
            as_scalar(factor),
            phi,
            as_scalar(rho),
            as_scalar(tfinal),
            u0,
            Nt,
            device=device,
        )

    engine_module.solve_fisher_kpp_torch = patched


def _scale_prior_weights(weights_inv: str, scale: float) -> str:
    """Multiply the four weights of the prior in the reference weight string.

    The prior is a penalty that pulls the four parameters toward their start
    values; ``--prior-scale 0`` switches it off.

    ``train_opts['weights_inv']`` is a flat "name,value,name,value" string, and
    the four prior terms are the ones whose name ends in ``_reg``: rD_reg,
    rRHO_reg, th1_reg and th2_reg. Rewriting it here lets the PINN arm run with
    the prior off, as CMA-ES can; nothing in the reference is edited, only the
    option it already reads.
    """
    parts = weights_inv.split(",")
    for index in range(0, len(parts) - 1, 2):
        if parts[index].endswith("_reg"):
            parts[index + 1] = repr(float(parts[index + 1]) * scale)
    return ",".join(parts)


def _scale_seg_weights(weights_inv: str, scale: float) -> str:
    """Multiply the two mask-loss weights (seg1, seg2) in the reference weight string."""
    parts = weights_inv.split(",")
    for index in range(0, len(parts) - 1, 2):
        if parts[index] in ("seg1", "seg2"):
            parts[index + 1] = repr(float(parts[index + 1]) * scale)
    return ",".join(parts)


def _build_options(config: dict, paths, device: str | None, prior_scale: float = 1.0):
    """Create the reference ``Options`` object with our settings pinned."""
    sys.path.insert(0, str(REFERENCE_DIR))
    from Options import Options  # noqa: E402  (import needs the path above)

    options = Options()
    opts = options.opts
    params = config["parameters"]
    pinn = config["pinn"]

    opts["device"] = device or config["runtime"]["device"]
    opts["seed"] = config["runtime"]["seed"]
    opts["skip_fig"] = True
    opts["logger_opts"]["use_mlflow"] = False
    opts["logger_opts"]["use_stdout"] = True
    opts["logger_opts"]["use_tmp"] = True

    opts["dataset_opts"]["segpre_file"] = str(paths.tumor_seg)
    opts["dataset_opts"]["wm_file"] = str(paths.wm)
    opts["dataset_opts"]["gm_file"] = str(paths.gm)
    opts["dataset_opts"]["csf_file"] = str(paths.csf)
    if paths.brain_mask is not None:
        opts["dataset_opts"]["brain_mask_file"] = str(paths.brain_mask)
    if paths.recurrence is not None:
        opts["dataset_opts"]["segrec_file"] = str(paths.recurrence)

    opts["pde_opts"]["factor"] = config["model"]["factor"]
    opts["pde_opts"]["h_init"] = config["initial_condition"]["h_init"]
    opts["pde_opts"]["r_init"] = config["initial_condition"]["r_init"]
    opts["pde_opts"]["prior_th1"] = params["th_wt"]["init"]
    opts["pde_opts"]["prior_th2"] = params["th_tc"]["init"]
    opts["pde_opts"]["radius_method"] = config["characteristic"]["radius_method"]
    opts["pde_opts"]["dat_batch_size"] = pinn["dat_batch_size"]
    opts["pde_opts"]["res_batch_size"] = pinn["res_batch_size"]
    # No effect: the reference hard-codes steepness 20 and never reads ksigmoid.
    opts["pde_opts"]["ksigmoid"] = config["objective"]["steepness"]

    # Every option that decides the network's shape is pinned here, not left to
    # the reference code's defaults. In configs/base.yaml they equal those
    # defaults except separable (false, a dense network); pinning them makes the
    # choice explicit and checkable.
    # Zhang et al. App. E (p. 14) specify a plain 4x128 tanh MLP; `arch: mmlp`
    # with `fourier: true` is a modified MLP with a random Fourier input layer,
    # which is what the released code builds and therefore what ran.
    opts["nn_opts"]["separable"] = pinn["separable"]
    opts["nn_opts"]["depth"] = pinn["depth"]
    opts["nn_opts"]["width"] = pinn["width"]
    opts["nn_opts"]["arch"] = pinn["arch"]
    opts["nn_opts"]["fourier"] = bool(pinn["fourier"])
    opts["nn_opts"]["sigma"] = float(pinn["sigma"])
    opts["nn_opts"]["output_rank"] = int(pinn["output_rank"])
    opts["nn_opts"]["act"] = pinn["act"]

    opts["train_opts"]["iter_init"] = pinn["iter_init"]
    opts["train_opts"]["iter_inv"] = pinn["iter_inv"]
    opts["train_opts"]["lr_init"] = float(pinn["lr_init"])
    opts["train_opts"]["lr_inv"] = float(pinn["lr_inv"])
    # The paper-settings run: the training settings printed by Zhang et al., where
    # the release differs. docs/PINN_SETTINGS.md lists each one with its locator.
    paper = pinn.get("paper_settings")
    if paper:
        w_seg = repr(float(paper["w_seg"]))
        # Zhang et al. Eq. (14): L_PDE + L_char, both weight 1 (p. 6: "all the other losses are weighted by 1").
        opts["train_opts"]["weights_init"] = "res,1.0,uchar_res,1.0"
        # Zhang et al. Eq. (19): L_PDE + w_SEG L_SEG + box penalties (Eq. 18, weight 1); no prior.
        opts["train_opts"]["loss_inv"] = "res,seg1,seg2,rD_reg,rRHO_reg,th1_reg,th2_reg,thrange,murange"
        opts["train_opts"]["weights_inv"] = (f"res,1.0,seg1,{w_seg},seg2,{w_seg},rD_reg,0.0,rRHO_reg,0.0,"
                                             "th1_reg,0.0,th2_reg,0.0,thrange,1.0,murange,1.0")
        # Zhang et al. App. E: "ADAM" (the release uses AMSGrad).
        opts["train_opts"]["optim_opts"] = f"amsgrad,{bool(paper['amsgrad'])}"
        # Zhang et al. App. F sample early times densely; they do not weight the residual in time.
        opts["pde_opts"]["texp_weight"] = float(paper["texp_weight"])
        # Zhang et al. Sec. 2.3: u = t u_NN + u_0, without the release's extra factor that forces u to
        # the initial value on the ball's surface.
        opts["pde_opts"]["force_bc"] = bool(paper["force_bc"])

    if prior_scale != 1.0:
        opts["train_opts"]["weights_inv"] = _scale_prior_weights(
            opts["train_opts"]["weights_inv"], prior_scale
        )
    # No-data test: 0 removes both mask terms from fine-tuning, so mu_D moves only
    # under the residual. The losses are still computed and logged.
    seg_scale = pinn.get("seg_weight_scale")
    if seg_scale is not None:
        opts["train_opts"]["weights_inv"] = _scale_seg_weights(
            opts["train_opts"]["weights_inv"], float(seg_scale)
        )

    options.process_options()
    return options


def _patch_ball_threshold(engine_module, threshold: float) -> None:
    """Collocation ball where the characteristic solution exceeds ``threshold``.

    Zhang et al. App. F (p. 15) enclose "the region where u_FDM > 0.001"; the release hard-codes
    0.01 (``Engine.py:491``) and passes it to ``gen_samples`` as its ninth argument.
    """
    original = engine_module.gen_samples

    def patched(*args):
        args = list(args)
        args[8] = threshold
        return original(*args)

    engine_module.gen_samples = patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-density", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "random seed of the PINN; overrides runtime.seed from the config. The reference "
            "implementation seeds its own network initialization from this value, and Lu et "
            "al. (2021, DeepXDE), section 2.3, recommend repeating a PINN from several "
            "initializations because it converges to different solutions. Omit to use the "
            "config value. A run with an overridden random seed writes to "
            "<case>__pinn_seed<N>.json so nothing is overwritten; put repeats in their own "
            "--out directory, so that runs with different random seeds are never averaged "
            "by accident"
        ),
    )
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=1.0,
        help=(
            "multiply the weights of the prior (a penalty that pulls the four parameters "
            "toward their start values) in weights_inv. 1.0 reproduces the score of the "
            "reference code; 0.0 switches the prior off"
        ),
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    # Written back into the config before _build_options reads it, so the record's
    # embedded config block states the random seed the run used. The reference
    # Options object takes it from runtime.seed.
    config_seed = config["runtime"]["seed"]
    seed = config_seed if args.seed is None else args.seed
    config["runtime"]["seed"] = seed
    seed_tag = "" if seed == config_seed else f"_seed{seed}"

    # The clock starts here, before any case file is touched, and it runs in the
    # CMA-ES script from the same point. Everything each method needs in order to
    # answer -- reading the case, the phase field, the characteristic grid search,
    # then the method itself -- is inside the reported run time.
    started = time.perf_counter()

    paths = (
        synthetic_paths(args.data_root, args.case)
        if args.synthetic
        else find_case(args.data_root, args.case)
    )
    case = load_case(
        paths,
        cache_dir=Path(config["data"]["cache_dir"]),
        phase_field_iters=config["model"]["phase_field"]["n_iter"],
        phase_field_epsilon=config["model"]["phase_field"]["epsilon_mm"],
        device=args.device,
    )

    options = _build_options(config, paths, args.device, args.prior_scale)
    sys.path.insert(0, str(REFERENCE_DIR))
    import Engine as engine_module  # noqa: E402
    from Engine import Engine  # noqa: E402

    _patch_scalar_conversion(engine_module)
    paper_block = config["pinn"].get("paper_settings")
    if paper_block:
        _patch_ball_threshold(engine_module, float(paper_block["ball_threshold"]))

    # The original TensorFlow code's early stopping, only when the config asks for
    # it (configs/paper_schedule.yaml). base.yaml has no such block, so a run
    # with base.yaml keeps the release's fixed step count.
    stop_log: dict = {}
    stop_block = config["pinn"].get("early_stopping")
    param_lr_block = config["pinn"].get("param_lr")
    if param_lr_block and not stop_block:
        raise ValueError("pinn.param_lr needs an early_stopping block (as in configs/paper_schedule.yaml): "
                         "the separate rate is applied inside its training loop")
    if stop_block:
        import Trainer as trainer_module  # noqa: E402

        from experiments.pinn_early_stopping import EarlyStopRule, install
        from experiments.pinn_lbfgs import LbfgsSettings
        from experiments.pinn_paper_settings import PaperSettings
        from experiments.pinn_param_lr import ParamLrSettings

        # The original code's L-BFGS stage, only with an lbfgs block (configs/paper_schedule_lbfgs_full.yaml).
        lbfgs_block = config["pinn"].get("lbfgs")
        install(trainer_module.Trainer, EarlyStopRule.from_config(stop_block), stop_log,
                lbfgs=LbfgsSettings.from_config(lbfgs_block) if lbfgs_block else None,
                param_lr=ParamLrSettings.from_config(param_lr_block) if param_lr_block else None,
                paper=(PaperSettings.from_config(paper_block, seed=seed)
                       if paper_block else None))
    elif paper_block:
        raise ValueError("pinn.paper_settings needs an early_stopping block (as in configs/paper_schedule.yaml)")

    original_stdout = sys.stdout
    recorder = StepRecorder(original_stdout, started)
    sys.stdout = recorder
    try:
        engine = Engine(options.opts)
        engine.run()
    finally:
        sys.stdout = original_stdout
    wall_clock = time.perf_counter() - started
    setup_s = recorder.history[0]["elapsed_s"] if recorder.history else None

    density = np.asarray(engine.vdict["upred_fdm"][..., -1], dtype=np.float32)
    estimate = {
        "mu_d": float(np.squeeze(engine.pde.dataset["rD_pred"])),
        "mu_rho": float(np.squeeze(engine.pde.dataset["rRHO_pred"])),
        "th_wt": float(np.squeeze(engine.pde.dataset["th1_pred"])),
        "th_tc": float(np.squeeze(engine.pde.dataset["th2_pred"])),
    }
    char = {k: float(v) for k, v in engine.char_params.items() if np.isscalar(v)}
    d_eff = estimate["mu_d"] * char.get("Dc", float("nan")) * char.get("T", float("nan"))
    rho_eff = estimate["mu_rho"] * char.get("rhoc", float("nan")) * char.get("T", float("nan"))

    metrics = segmentation_dice(
        density, case.segmentation, th_wt=estimate["th_wt"], th_tc=estimate["th_tc"]
    )
    if case.recurrence is not None:
        metrics.update(
            coverage_metrics(
                density,
                case.segmentation,
                case.recurrence,
                case.brain_mask,
                margin_mm=config["evaluation"]["ctv_margin_mm"],
                voxel_mm=config["model"]["voxel_mm"],
            )
        )

    truth_file = args.data_root / args.case / "truth.json"
    if args.synthetic and truth_file.is_file():
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        metrics.update(
            parameter_errors(
                ParameterTruth(d_eff, rho_eff, estimate["th_wt"], estimate["th_tc"]),
                ParameterTruth(truth["d_eff"], truth["rho_eff"], truth["th_wt"], truth["th_tc"]),
            )
        )
        metrics.update(
            mu_errors(
                estimate["mu_d"],
                estimate["mu_rho"],
                truth.get("mu_d_true", float("nan")),
                truth.get("mu_rho_true", float("nan")),
            )
        )

    lam = float(np.sqrt(d_eff / rho_eff)) if rho_eff > 0 else float("nan")
    record = {
        "case_id": args.case,
        "method": METHOD,
        "synthetic": bool(args.synthetic),
        "provenance": {
            **provenance(repo),
            "reference_commit": "99af20f",
            "prior_scale": args.prior_scale,
            "seed": seed,
            **device_facts(),
            # The resolved network options, copied from the object the reference
            # code actually built from, so a record states its own architecture
            # instead of leaving a reader to infer it from Options.py defaults.
            "nn_opts": dict(options.opts["nn_opts"]),
            "reference_patched": "solve_fisher_kpp_torch wrapped for NumPy 2 scalar conversion",
            # Reach test: None unless rD and rRHO had their own Adam rate, which
            # is our modification, not part of the published method.
            "param_lr": param_lr_block or None,
        },
        "config": config,
        "characteristic": char,
        "estimate": {**estimate, "d_eff": d_eff, "rho_eff": rho_eff, "lambda_mm": lam},
        "metrics": metrics,
        "cost": {
            "wall_clock_s": wall_clock,
            # The reference does its own setup inside run(): the characteristic
            # grid search, the phase field and the training-sample generation.
            # None of it is printed, but the first recorded step marks its end,
            # so the split below is measured, not estimated.
            "setup_s": setup_s,
            "solve_s": wall_clock - setup_s if setup_s is not None else None,
            "n_evaluations": None,
            "stop_reason": (
                "; ".join(f"{stage}: {entry['stop_reason']} after {entry['steps_run']} of "
                          f"{entry['max_steps']}" for stage, entry in stop_log.items())
                if stop_block else "fixed iteration budget"
            ),
        },
        # Per stage: step budget, steps run, the step and reason of the stop, and
        # every checked value of the watched loss. Empty without early stopping.
        "early_stopping": {"rule": stop_block, "stages": stop_log} if stop_block else None,
        "history": recorder.history,
    }

    out_file = args.out / f"{args.case}__{METHOD}{seed_tag}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if args.save_density:
        nib.save(
            nib.Nifti1Image(density, case.affine),
            args.out / f"{args.case}__{METHOD}{seed_tag}_density.nii.gz",
        )

    engine.logger.close()
    print(f"{args.case}: PINN finished in {wall_clock / 60:.1f} min -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
