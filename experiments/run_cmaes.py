#!/usr/bin/env python
"""Calibrate one case with CMA-ES and write one result record.

One invocation handles one case, so a cohort can run as a job array, and every
record carries its own provenance: git commit, GPU model, job id, random seed
and the full configuration. ``experiments/run_pinn.py`` writes the same schema, so
the table scripts can pool both arms without special cases.

Usage::

    python -m experiments.run_cmaes --case RHUH-0001 \
        --data-root src/data/predict_gbm/predict_gbm --out outputs/runs
    python -m experiments.run_cmaes --case SYN-RHUH-0002-d1.2-r0.6 \
        --data-root outputs/synthetic --synthetic --out outputs/runs
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from src.data.predict_gbm import CasePaths, find_case, load_case
from src.evaluation.metrics import (
    ParameterTruth,
    coverage_metrics,
    mu_errors,
    parameter_errors,
    segmentation_dice,
)
from src.models.characteristic import estimate_characteristic
from src.models.fisher_kpp import gaussian_seed, solve_fisher_kpp
from src.training.cmaes_calibration import CalibrationSettings, calibrate
from src.training.objective import ObjectiveWeights, SegmentationObjective

METHOD = "cmaes"


def git_commit(repo: Path) -> str:
    """Short commit hash of the repository, or ``unknown``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # Without a .git directory, read the commit from a VERSION file if present.
    stamp = Path(repo) / "VERSION"
    if stamp.is_file():
        return stamp.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def device_facts() -> dict:
    """Device memory, compute capability, ``CUDA_VISIBLE_DEVICES`` and MIG UUID.

    These add to the device name in ``provenance.gpu_model``. Every field is
    ``None`` off-GPU, so a CPU record stays readable by the table scripts.
    """
    facts: dict = {
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
        "gpu_capability": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mig_uuid": None,
    }
    try:
        if not torch.cuda.is_available():
            return facts
        facts["gpu_name"] = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        facts["gpu_total_memory_bytes"] = int(properties.total_memory)
        facts["gpu_capability"] = f"{properties.major}.{properties.minor}"
        # A MIG instance reports its own UUID, of the form MIG-<uuid>; the device
        # name alone cannot tell two instances apart.
        uuid = getattr(properties, "uuid", None)
        if uuid is not None:
            facts["mig_uuid"] = str(uuid)
    except Exception as error:  # a broken driver must not lose a finished run
        facts["gpu_probe_error"] = repr(error)
    return facts


def _cma_version() -> str:
    """The installed pycma version, or a marker. Never raises: a missing library
    must not lose a finished run, and the PINN arm does not import cma at all."""
    try:
        import cma

        return str(getattr(cma, "__version__", "unknown"))
    except Exception:
        return "absent"


def provenance(repo: Path) -> dict:
    """Everything needed to trace a number back to the run that made it."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "git_commit": git_commit(repo),
        "gpu_model": gpu,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        # The CMA-ES arm *is* the pycma library: cmaes_calibration.py overrides
        # only six of its options, so every strategy constant is a pycma default
        # and an upgrade would change the search silently.
        "cma_version": _cma_version(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def synthetic_paths(root: Path, case_id: str) -> CasePaths:
    """Paths of a synthetic case, which has no recurrence map."""
    case_dir = root / case_id
    paths = CasePaths(
        case_id=case_id,
        tumor_seg=case_dir / "tumor_seg.nii.gz",
        wm=case_dir / "wm_pbmap.nii.gz",
        gm=case_dir / "gm_pbmap.nii.gz",
        csf=case_dir / "csf_pbmap.nii.gz",
        brain_mask=case_dir / "t1c_bet_mask.nii.gz",
        recurrence=None,
    )
    paths.require()
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--synthetic", action="store_true", help="case built by make_synthetic")
    parser.add_argument("--device", default=None)
    parser.add_argument("--time-budget-s", type=float, default=None)
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=1.0,
        help=(
            "multiply the weights of the prior, a penalty that pulls the four parameters "
            "toward their start values. 1.0 reproduces the score of the reference code; "
            "0.0 switches the prior off"
        ),
    )
    parser.add_argument("--save-density", action="store_true", help="also write the predicted field")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "random seed of CMA-ES; overrides cmaes.seed from the config. CMA-ES samples "
            "its population from a normal distribution, so the random seed changes the "
            "search. pycma treats 0 as a clock seed, so a run with random seed 0 cannot "
            "be repeated exactly. Omit to use the config value, as in our paper's main "
            "runs. A run with an overridden random seed writes to "
            "<case>__cmaes_seed<N>.json so nothing is overwritten; put repeats in their "
            "own --out directory, so that runs with different random seeds are never "
            "averaged by accident"
        ),
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    # The random seed is written back into the config before anything reads it,
    # so the record's embedded config block states the random seed the run used.
    # The suffix keeps a repeat from overwriting the baseline record.
    config_seed = config["cmaes"]["seed"]
    seed = config_seed if args.seed is None else args.seed
    config["cmaes"]["seed"] = seed
    seed_tag = "" if seed == config_seed else f"_seed{seed}"

    # The clock starts here, before any case file is touched, and the PINN script
    # starts it at the same point.
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
        # Renormalization-off refit: false skips the second renormalization of the
        # tissue maps inside the tumor, so the fit sees the generator's own field.
        # Absent = true, as in the main runs.
        renormalise_tumour=bool(config["data"].get("renormalise_tumour", True)),
    )

    params = config["parameters"]
    scaling = estimate_characteristic(
        case.mask_wt,
        case.mask_tc,
        x0=case.x0,
        th_wt=params["th_wt"]["init"],
        th_tc=params["th_tc"]["init"],
        h_init=config["initial_condition"]["h_init"],
        r_init=config["initial_condition"]["r_init"],
        xdim=config["characteristic"]["xdim"],
        radius_method=config["characteristic"]["radius_method"],
        domain=config["characteristic"]["domain_mm"],
    )

    # Optional: the tumor seed position x0 of the generator instead of the
    # centroid of the case's own mask. The positions are precomputed from the
    # source patients' masks (scripts/x1_generator_seeds.py), so the fit needs no
    # access to the source patient's data. The characteristic scale above keeps
    # case.x0, so mu stays in the coordinates the truth is stored in.
    x0 = case.x0
    x0_file = config["initial_condition"].get("x0_file")
    if x0_file:
        seeds = json.loads((repo / x0_file).read_text(encoding="utf-8"))
        if args.case not in seeds["cases"]:
            raise KeyError(f"{args.case} has no tumor seed position in {x0_file}")
        x0 = np.asarray(seeds["cases"][args.case]["x0_vox"], dtype=float)

    u0 = gaussian_seed(
        case.shape,
        x0,
        h_init=config["initial_condition"]["h_init"],
        r_init=config["initial_condition"]["r_init"],
        h=config["model"]["voxel_mm"],
    )

    # The prior (a penalty that pulls the four parameters toward their start
    # values) charges (mu - 1)^2 at weight 1e-3 while the MRI terms carry 1e-5,
    # so a step of 0.15 in mu_D costs more in prior than the whole data term is
    # worth. --prior-scale 0 switches the prior off and measures what the MRI
    # alone supports.
    weights = ObjectiveWeights(**config["objective"]["weights"])
    if args.prior_scale != 1.0:
        weights = replace(
            weights,
            reg_mu_d=weights.reg_mu_d * args.prior_scale,
            reg_mu_rho=weights.reg_mu_rho * args.prior_scale,
            reg_th_wt=weights.reg_th_wt * args.prior_scale,
            reg_th_tc=weights.reg_th_tc * args.prior_scale,
        )

    objective = SegmentationObjective(
        case.mask_wt,
        case.mask_tc,
        case.phi,
        prior_th_wt=params["th_wt"]["init"],
        prior_th_tc=params["th_tc"]["init"],
        weights=weights,
        steepness=config["objective"]["steepness"],
        mu_range=(params["mu_d"]["min"], params["mu_d"]["max"]),
        th_range=(params["th_wt"]["min"], params["th_wt"]["max"]),
        # Each parameter's own box. In the configs of our paper's main runs mu_rho
        # shares mu_D's box and th_tc shares th_wt's.
        mu_rho_range=(params["mu_rho"]["min"], params["mu_rho"]["max"]),
        th_tc_range=(params["th_tc"]["min"], params["th_tc"]["max"]),
    )

    # Optional: both thresholds held at the truth and removed from the search.
    # Synthetic cases only; nothing else knows a true threshold.
    fixed: dict[str, float] = {}
    fix = config["cmaes"].get("fix_thresholds")
    if fix:
        truth_path = args.data_root / args.case / "truth.json"
        if fix != "truth" or not args.synthetic or not truth_path.is_file():
            raise ValueError("cmaes.fix_thresholds supports only 'truth', on a synthetic case "
                             "with a truth.json")
        known = json.loads(truth_path.read_text(encoding="utf-8"))
        fixed = {"th_wt": float(known["th_wt"]), "th_tc": float(known["th_tc"])}

    settings = CalibrationSettings(
        lower=(params["mu_d"]["min"], params["mu_rho"]["min"], params["th_wt"]["min"], params["th_tc"]["min"]),
        upper=(params["mu_d"]["max"], params["mu_rho"]["max"], params["th_wt"]["max"], params["th_tc"]["max"]),
        initial=(params["mu_d"]["init"], params["mu_rho"]["init"], params["th_wt"]["init"], params["th_tc"]["init"]),
        sigma0=config["cmaes"]["sigma0"],
        popsize=config["cmaes"]["popsize"],
        max_evaluations=config["cmaes"]["max_evaluations"],
        tol_fun=float(config["cmaes"]["tol_fun"]),
        seed=config["cmaes"]["seed"],
        time_budget_s=args.time_budget_s,
        fixed=fixed,
    )

    result = calibrate(
        p_wm=case.p_wm,
        p_gm=case.p_gm,
        phi=case.phi,
        u0=u0,
        objective=objective,
        scaling=scaling,
        settings=settings,
        factor=config["model"]["factor"],
        voxel_mm=config["model"]["voxel_mm"],
        cfl_safety=config["model"]["cfl_safety"],
        device=args.device,
    )

    wall_clock = time.perf_counter() - started
    setup_s = wall_clock - result.wall_clock_s

    # The optimizer measures its own progress from its own start, so every
    # history entry is shifted onto the shared origin above. Both arms then plot
    # against the same zero and the two cost curves can be read together.
    for entry in result.history:
        entry["elapsed_s"] += setup_s

    best = result.best_parameters
    d_eff, rho_eff = scaling.to_physical(best["mu_d"], best["mu_rho"])
    final = solve_fisher_kpp(
        case.p_wm,
        case.p_gm,
        case.phi,
        u0,
        d_w=d_eff,
        rho=rho_eff,
        t_final=1.0,
        factor=config["model"]["factor"],
        n_save=2,
        h=config["model"]["voxel_mm"],
        device=args.device,
    ).final
    density = final * case.phi

    metrics = segmentation_dice(
        density, case.segmentation, th_wt=best["th_wt"], th_tc=best["th_tc"]
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
                ParameterTruth(d_eff, rho_eff, best["th_wt"], best["th_tc"]),
                ParameterTruth(truth["d_eff"], truth["rho_eff"], truth["th_wt"], truth["th_tc"]),
            )
        )
        metrics.update(
            mu_errors(
                best["mu_d"],
                best["mu_rho"],
                truth.get("mu_d_true", float("nan")),
                truth.get("mu_rho_true", float("nan")),
            )
        )

    record = {
        "case_id": args.case,
        "method": METHOD,
        "synthetic": bool(args.synthetic),
        "provenance": {
            **provenance(repo),
            "prior_scale": args.prior_scale,
            "seed": seed,
            **device_facts(),
            # Empty / equal to the centroid except in the runs with the thresholds
            # or the tumor seed position fixed.
            "fixed_parameters": fixed,
            "x0_vox": [float(v) for v in x0],
            "x0_mask_centroid_vox": [float(v) for v in case.x0],
            "renormalise_tumour": bool(config["data"].get("renormalise_tumour", True)),
        },
        "config": config,
        "characteristic": scaling.as_dict(),
        "estimate": {**best, "d_eff": d_eff, "rho_eff": rho_eff,
                     "lambda_mm": scaling.infiltration_length(best["mu_d"], best["mu_rho"])},
        "metrics": metrics,
        "cost": {
            "wall_clock_s": wall_clock,
            "setup_s": setup_s,
            "solve_s": result.wall_clock_s,
            "n_evaluations": result.n_evaluations,
            "stop_reason": result.stop_reason,
        },
        "history": result.history,
    }

    out_file = args.out / f"{args.case}__{METHOD}{seed_tag}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if args.save_density:
        nib.save(
            nib.Nifti1Image(density.astype(np.float32), case.affine),
            args.out / f"{args.case}__{METHOD}{seed_tag}_density.nii.gz",
        )

    print(
        f"{args.case}: score {result.best_score:.6e}, "
        f"{result.n_evaluations} solves, {result.wall_clock_s / 60:.1f} min -> {out_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
