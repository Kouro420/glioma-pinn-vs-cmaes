#!/usr/bin/env python
"""Generate the synthetic test cases, the primary outcome of our paper.

Patient data carries no ground truth: nobody knows the true cell density or the
true parameters. Only synthetic tumors can answer "which method recovers the
truth", so these cases carry the main claim.

The danger is the inverse crime. If the truth were produced by the same
discretization the baseline later inverts, CMA-ES could recover it exactly while
the PINN could not, and the comparison would measure the setup rather than the
methods. Three measures reduce that:

1. the truth is solved on a grid with half the voxel size (0.5 mm), with the
   correspondingly smaller time step,
2. the result is block-averaged back to the 1 mm grid both methods use,
3. spatially correlated noise is added before thresholding, as in
   Zhang et al. (2025), Sec. 3.1.

What remains: both methods still assume the true model
*form*, so this is a parameter-recovery test, not a model-adequacy test.

Usage::

    python -m src.data.make_synthetic --case RHUH-0001 --data-root src/data/predict_gbm/predict_gbm \
        --out outputs/synthetic --n-cases 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter

from src.data.predict_gbm import find_case, load_case
from src.models.characteristic import CharacteristicScaling, estimate_characteristic
from src.models.fisher_kpp import gaussian_seed, phase_field, solve_fisher_kpp

__all__ = ["SyntheticTruth", "default_truths", "make_case", "main"]


@dataclass
class SyntheticTruth:
    """Ground truth of one synthetic case.

    ``d_eff`` and ``rho_eff`` are the identifiable products D*t_end and
    rho*t_end at ``t_final = 1``; ``lam`` is the infiltration length in mm.
    """

    case_id: str
    source_case: str
    d_eff: float
    rho_eff: float
    th_wt: float
    th_tc: float
    lam: float
    wave_travel_mm: float
    noise_sigma: float
    noise_smoothing_mm: float
    fine_voxel_mm: float
    # What the two methods actually search for. Both start from mu = 1, so an
    # error in mu is the number that says whether a method moved toward the
    # truth or stayed at the grid-search point. It cannot be set here: the
    # characteristic scale comes from the masks, which do not exist until the
    # case is simulated, so make_case fills these in.
    mu_d_true: float = float("nan")
    mu_rho_true: float = float("nan")
    # What was asked for, on the source patient's characteristic scale. The pair
    # above is recomputed from the synthetic case's own masks, which is the scale
    # the two methods will actually see, so the two do not match exactly. Both
    # are kept: the target says what was simulated, the truth says what a method
    # searching this case has to find.
    mu_d_target: float = float("nan")
    mu_rho_target: float = float("nan")


def default_truths(
    scalings: Mapping[str, "CharacteristicScaling"],
    *,
    mu_ds: Sequence[float] = (0.2, 0.4, 0.8, 1.2),
    mu_rhos: Sequence[float] = (0.6, 1.3),
    threshold_pairs: Sequence[tuple[float, float]] = ((0.25, 0.60), (0.45, 0.80)),
    noise_sigma: float = 0.05,
    noise_smoothing_mm: float = 3.0,
    fine_voxel_mm: float = 0.5,
    case_prefix: str = "SYN",
) -> list[SyntheticTruth]:
    """The truth set: every (mu_D, mu_rho) pair on every source brain.

    The parameters are chosen as multiples of each brain's own characteristic
    scale, not as absolute values, and this is the point of the design.

    An absolute (lambda, rho) grid can produce cases with no core mask: a large
    lambda spreads the cell density out while a small rho never saturates it, so
    the peak stays below the core threshold. Multiples of the brain's own scale
    stay plausible: mu = 1 is the tumor the shared grid search finds for this
    brain, so a modest multiple of it is still a plausible tumor.

    It also makes the primary outcome direct. Both methods start at
    mu_D = mu_rho = 1 and search in exactly these coordinates, so "did the search
    move toward the truth, or only away from the start?" is read off the grid
    without any conversion.

    The truth thresholds are 0.25/0.60 and 0.45/0.80, never the start values 0.35/0.70:
    a method that refuses to move would otherwise score a perfect threshold
    error. The pair is tied to mu_rho, because that is what sets how far the
    density saturates and therefore which thresholds a case can support. With
    eight cells per brain there is no room to cross it against everything, so
    threshold recovery is reported per pair and never pooled.

    Args:
        scalings: the characteristic scaling of each source brain, estimated from
            that patient's own masks. This fixes what mu = 1 means for the brain.
        mu_ds, mu_rhos: multiples of the characteristic scale to simulate.
        threshold_pairs: (whole-tumor, core) truth thresholds, one per mu_rho.
        noise_sigma, noise_smoothing_mm, fine_voxel_mm: imaging-noise and
            refinement settings, recorded in every truth file.
        case_prefix: first part of every case id. "SYN" for the base set;
            the design-range set uses "ENV" so its ids can never collide with it.

    Returns:
        ``len(scalings) * len(mu_ds) * len(mu_rhos)`` truths, in a fixed order.
    """
    truths: list[SyntheticTruth] = []
    for source_case, scaling in scalings.items():
        d_at_one = scaling.d_char * scaling.time
        rho_at_one = scaling.rho_char * scaling.time
        for mu_d in mu_ds:
            for index, mu_rho in enumerate(mu_rhos):
                th_wt, th_tc = threshold_pairs[index % len(threshold_pairs)]
                d_eff = mu_d * d_at_one
                rho_eff = mu_rho * rho_at_one
                truths.append(
                    SyntheticTruth(
                        case_id=f"{case_prefix}-{source_case}-d{mu_d:g}-r{mu_rho:g}",
                        source_case=source_case,
                        d_eff=float(d_eff),
                        rho_eff=float(rho_eff),
                        th_wt=float(th_wt),
                        th_tc=float(th_tc),
                        lam=float(np.sqrt(d_eff / rho_eff)),
                        wave_travel_mm=float(2.0 * np.sqrt(d_eff * rho_eff)),
                        noise_sigma=float(noise_sigma),
                        noise_smoothing_mm=float(noise_smoothing_mm),
                        fine_voxel_mm=float(fine_voxel_mm),
                        mu_d_target=float(mu_d),
                        mu_rho_target=float(mu_rho),
                    )
                )
    return truths


def _upsample(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """Trilinear refinement of a 3D map by an integer factor."""
    tensor = torch.as_tensor(volume, dtype=torch.float32)[None, None]
    fine = F.interpolate(tensor, scale_factor=factor, mode="trilinear", align_corners=False)
    return fine[0, 0].numpy()


def _block_average(volume: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average neighboring voxels back to the coarse grid."""
    shape = []
    for n in volume.shape:
        shape.extend([n // factor, factor])
    trimmed = volume[tuple(slice(0, (n // factor) * factor) for n in volume.shape)]
    return trimmed.reshape(shape).mean(axis=(1, 3, 5))


def _correlated_noise(
    shape: tuple[int, ...],
    sigma: float,
    smoothing_mm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smooth Gaussian noise, normalized to standard deviation ``sigma``."""
    white = rng.standard_normal(shape).astype(np.float32)
    smooth = gaussian_filter(white, sigma=smoothing_mm)
    std = float(smooth.std())
    if std == 0.0:
        return np.zeros(shape, dtype=np.float32)
    return (sigma * smooth / std).astype(np.float32)


def make_case(
    case,
    truth: SyntheticTruth,
    *,
    out_dir: Path,
    factor: int = 2,
    tissue_factor: float = 10.0,
    phase_field_iters: int = 100,
    phase_field_epsilon: float = 3.0,
    device: str | None = None,
    seed: int = 0,
    prior_th_wt: float = 0.35,
    prior_th_tc: float = 0.70,
    h_init: float = 0.5,
    r_init: float = 0.01,
    xdim: int = 3,
    radius_method: str = "rvol",
    domain_mm: float = 180.0,
    fine_phi: np.ndarray | None = None,
) -> dict:
    """Build one synthetic case on the geometry of a real patient.

    Args:
        case: a loaded :class:`src.data.predict_gbm.Case` supplying tissue maps.
        truth: the parameters to simulate.
        out_dir: directory receiving the NIfTI files and the truth record.
        factor: refinement factor of the truth solve (2 gives 0.5 mm).
        tissue_factor: white-to-gray diffusivity ratio.

    Returns:
        A record with the truth, the mask sizes and a boundary-contact warning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fine_wm = _upsample(case.p_wm, factor)
    fine_gm = _upsample(case.p_gm, factor)
    fine_csf = _upsample(case.p_csf, factor)
    # The fine phase field depends only on the brain, not on the parameters being
    # simulated, so every case on one brain gets the same field. Relaxing it is
    # the most expensive step in this function at 0.5 mm, which is why the caller
    # may compute it once per brain and pass it in.
    if fine_phi is None:
        fine_phi = phase_field(
            fine_wm,
            fine_gm,
            fine_csf,
            n_iter=phase_field_iters,
            epsilon=phase_field_epsilon,
            h=1.0 / factor,
            device=device,
        )

    h_fine = 1.0 / factor
    x0_mm = case.x0 * 1.0  # the coarse grid is 1 mm, so indices are millimeters
    u0 = gaussian_seed(fine_wm.shape, x0_mm, h_init=0.5, r_init=0.01, h=h_fine)

    solution = solve_fisher_kpp(
        fine_wm,
        fine_gm,
        fine_phi,
        u0,
        d_w=truth.d_eff,
        rho=truth.rho_eff,
        t_final=1.0,
        factor=tissue_factor,
        n_save=2,
        h=h_fine,
        device=device,
    )

    density = _block_average(solution.final, factor)
    density = np.clip(density, 0.0, 1.0)

    noisy = density + _correlated_noise(
        density.shape, truth.noise_sigma, truth.noise_smoothing_mm, rng
    )
    noisy = np.clip(noisy, 0.0, 1.0)
    # The noise is added everywhere, so a few of its tail voxels cross the
    # threshold outside the head: at sigma = 0.05 a 0.25 threshold is five
    # standard deviations, which over a nine-million-voxel volume is a handful of
    # stray voxels. They are not tumor, and no segmentation tool would mark them.
    # The density is zero outside the brain by construction, so the masks are
    # confined there too.
    noisy = np.where(case.brain_mask, noisy, 0.0)

    mask_wt = noisy > truth.th_wt
    mask_tc = noisy > truth.th_tc
    segmentation = np.zeros(density.shape, dtype=np.int16)
    segmentation[mask_wt] = 2  # edema
    segmentation[mask_tc] = 3  # enhancing

    # A truth whose peak density never reaches its own core threshold has no core
    # mask, and the grid search downstream then dies with a generic "masks must be
    # non-empty". Fail here instead, naming the case and the numbers, because the
    # fault is in the chosen (lambda, rho, threshold) combination and not in the
    # solver: strong diffusion flattens the profile, so a large lambda at a small
    # rho spreads the tumor without ever saturating it.
    for name, mask, threshold in (("whole-tumor", mask_wt, truth.th_wt),
                                  ("core", mask_tc, truth.th_tc)):
        if not mask.any():
            raise ValueError(
                f"{truth.case_id}: the {name} mask is empty. Its threshold is "
                f"{threshold:.2f} but the density peaks at {noisy.max():.3f} "
                f"(lambda={truth.lam:g} mm, rho_eff={truth.rho_eff:g}, "
                f"D_eff={truth.d_eff:g}). Lower the threshold or change the truth."
            )

    touches_edge = bool(
        mask_wt[0].any()
        or mask_wt[-1].any()
        or mask_wt[:, 0].any()
        or mask_wt[:, -1].any()
        or mask_wt[..., 0].any()
        or mask_wt[..., -1].any()
    )

    affine = case.affine
    nib.save(nib.Nifti1Image(segmentation, affine), out_dir / "tumor_seg.nii.gz")
    nib.save(nib.Nifti1Image(density.astype(np.float32), affine), out_dir / "truth_density.nii.gz")
    nib.save(nib.Nifti1Image(case.p_wm, affine), out_dir / "wm_pbmap.nii.gz")
    nib.save(nib.Nifti1Image(case.p_gm, affine), out_dir / "gm_pbmap.nii.gz")
    nib.save(nib.Nifti1Image(case.p_csf, affine), out_dir / "csf_pbmap.nii.gz")
    nib.save(
        nib.Nifti1Image(case.brain_mask.astype(np.uint8), affine),
        out_dir / "t1c_bet_mask.nii.gz",
    )

    # Both methods begin at mu_D = mu_rho = 1 on the characteristic scale that
    # this grid search sets, and the grid search runs on the generated masks with
    # the start values of the thresholds, because no method knows the truth
    # thresholds. Storing
    # the truth in the same coordinates is what makes "did the search move toward
    # the answer, or only away from the start?" a question the results can answer.
    scaling = estimate_characteristic(
        mask_wt,
        mask_tc,
        x0=case.x0,
        th_wt=prior_th_wt,
        th_tc=prior_th_tc,
        h_init=h_init,
        r_init=r_init,
        xdim=xdim,
        radius_method=radius_method,
        domain=domain_mm,
    )
    truth = replace(
        truth,
        mu_d_true=float(truth.d_eff / (scaling.d_char * scaling.time)),
        mu_rho_true=float(truth.rho_eff / (scaling.rho_char * scaling.time)),
    )

    record = {
        **asdict(truth),
        "characteristic": scaling.as_dict(),
        "voxels_wt": int(mask_wt.sum()),
        "voxels_tc": int(mask_tc.sum()),
        "touches_domain_edge": touches_edge,
        "fine_steps": solution.n_steps,
        "fine_dt": solution.dt,
    }
    (out_dir / "truth.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--case",
        required=True,
        action="append",
        help="source patient supplying the geometry; repeat for more than one brain",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="the start values, the initial condition and the grid-search settings are read from here, "
             "so that the truth in mu-coordinates is expressed on the same scale the runs use",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help="truncate the list; the default builds every case on every brain",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed of the noise of case 0 (case i uses seed + i); "
                             "default: synthetic.seed of the config")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    params = config["parameters"]
    ic = config["initial_condition"]
    char = config["characteristic"]
    syn = config["synthetic"]
    seed = args.seed if args.seed is not None else int(syn["seed"])

    # One load per brain, not one per case: load_case relaxes the phase field,
    # which is the expensive part and is identical for every case on that brain.
    sources = {
        case_id: load_case(
            find_case(args.data_root, case_id),
            cache_dir=args.out / "_cache",
            phase_field_iters=config["model"]["phase_field"]["n_iter"],
            phase_field_epsilon=config["model"]["phase_field"]["epsilon_mm"],
            device=args.device,
        )
        for case_id in args.case
    }
    refine = int(round(config["model"]["voxel_mm"] / syn["fine_voxel_mm"]))
    # What mu = 1 means is a property of the brain, so each source patient's own
    # masks fix its scale before any truth is chosen.
    scalings = {}
    for case_id, source in sources.items():
        scalings[case_id] = estimate_characteristic(
            source.mask_wt,
            source.mask_tc,
            x0=source.x0,
            th_wt=params["th_wt"]["init"],
            th_tc=params["th_tc"]["init"],
            h_init=ic["h_init"],
            r_init=ic["r_init"],
            xdim=char["xdim"],
            radius_method=char["radius_method"],
            domain=char["domain_mm"],
        )
        sc = scalings[case_id]
        print("%s: mu=1 means D*t = %.1f mm^2 and rho*t = %.2f" %
              (case_id, sc.d_char * sc.time, sc.rho_char * sc.time))

    # The design-range set (configs/env_cases.yaml) sets its own grid, prefix and
    # kept box in the config; without those keys this builds the base set.
    grid = {key: tuple(syn[key]) for key in ("mu_ds", "mu_rhos") if key in syn}
    truths = default_truths(
        scalings,
        noise_sigma=syn["noise_sigma"],
        noise_smoothing_mm=syn["noise_smoothing_mm"],
        fine_voxel_mm=syn["fine_voxel_mm"],
        case_prefix=syn.get("case_prefix", "SYN"),
        **grid,
    )
    if args.n_cases is not None:
        truths = truths[: args.n_cases]

    fine_fields: dict[str, np.ndarray] = {}
    records = []
    for index, truth in enumerate(truths):
        if truth.source_case not in fine_fields:
            source = sources[truth.source_case]
            fine_fields[truth.source_case] = phase_field(
                _upsample(source.p_wm, refine),
                _upsample(source.p_gm, refine),
                _upsample(source.p_csf, refine),
                n_iter=config["model"]["phase_field"]["n_iter"],
                epsilon=config["model"]["phase_field"]["epsilon_mm"],
                h=1.0 / refine,
                device=args.device,
            )
            print(f"fine phase field ready for {truth.source_case}")
        record = make_case(
            sources[truth.source_case],
            truth,
            out_dir=args.out / truth.case_id,
            factor=refine,
            fine_phi=fine_fields[truth.source_case],
            tissue_factor=config["model"]["factor"],
            phase_field_iters=config["model"]["phase_field"]["n_iter"],
            phase_field_epsilon=config["model"]["phase_field"]["epsilon_mm"],
            device=args.device,
            seed=seed + index,
            prior_th_wt=params["th_wt"]["init"],
            prior_th_tc=params["th_tc"]["init"],
            h_init=ic["h_init"],
            r_init=ic["r_init"],
            xdim=char["xdim"],
            radius_method=char["radius_method"],
            domain_mm=char["domain_mm"],
        )
        records.append(record)
        flag = "  WARNING: tumor reaches the domain edge" if record["touches_domain_edge"] else ""
        print(f"{truth.case_id}: WT {record['voxels_wt']:>8d} voxels, TC {record['voxels_tc']:>8d}{flag}")

    # Design-range set: a case is used only if its truth, re-estimated on the case's
    # own masks (the scale the fit sees), lies inside the box for both mu. Every
    # case stays on disk and in index.json with its "kept" flag.
    box = syn.get("keep_truth_box")
    for record in records:
        record["kept"] = box is None or all(
            box[0] <= record[key] <= box[1] for key in ("mu_d_true", "mu_rho_true"))
    kept = [r["case_id"] for r in records if r["kept"]]
    if box is not None:
        print(f"kept {len(kept)} of {len(records)} cases with both true mu in "
              f"[{box[0]}, {box[1]}]; excluded: "
              f"{', '.join(r['case_id'] for r in records if not r['kept']) or 'none'}")

    (args.out / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    # The case list a job array indexes, in the order the cases were built, so
    # that array task i always refers to the same case.
    case_list = args.out / "cases_synthetic.txt"
    case_list.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"case list: {case_list}  (array indices 0-{len(kept) - 1})")

    edge = [r["case_id"] for r in records if r["touches_domain_edge"]]
    if edge:
        print(f"WARNING: {len(edge)} case(s) reach the domain edge: {', '.join(edge)}")
    print(f"wrote {len(records)} synthetic cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
