#!/usr/bin/env python
"""Map the calibration objective over the whole :math:`(\mu_D, \mu_\rho)` box.

The output is the score as a surface over the search box, the objective profiles of our paper.

## A profile, not a slice

At each grid point the PDE is solved **once**. The forward solution depends on
:math:`(\mu_D, \mu_\rho)` alone, never on the two density thresholds, so the
thresholds are then minimized *on the stored field* without solving again. A
slice at fixed thresholds would show a valley partly shaped by the wrong
thresholds, which is a different object.

The two thresholds decouple **exactly**. ``seg_wt`` is a function of ``th_wt``
alone and ``seg_tc`` of ``th_tc`` alone (``SegmentationObjective.__call__`` in
``src/training/objective.py``), and their penalties are separate, so minimizing
them jointly is minimizing them one at a time. The search runs on the GPU, and
the reported numbers then come from the numpy objective called at the chosen
thresholds. On the first point computed, the script checks that the two agree
and records the check in the metadata.

## The prior is off by default

The prior is a penalty pulling the four parameters toward their start values.
``--prior-scale 0.0``, the default, switches it off. With the prior on, the
surface shows the prior's bowl at :math:`(1,1)` as well as the valley that the
data impose. A run with the prior on is allowed; the script then prints a warning.

## Resumable

One row is appended and flushed per grid point, and a new call skips rows that
are already present. An interrupted grid does not have to start again.

Usage::

    # time four points and write nothing
    python scripts/profile_objective.py --case SYN-RHUH-0001-d0.2-r0.6 --probe

    # one case
    python scripts/profile_objective.py --case SYN-RHUH-0001-d0.2-r0.6

    # all four profiled cases
    python scripts/profile_objective.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.run_cmaes import synthetic_paths  # noqa: E402
from src.data.predict_gbm import load_case  # noqa: E402
from src.models.characteristic import estimate_characteristic  # noqa: E402
from src.models.fisher_kpp import gaussian_seed, solve_fisher_kpp  # noqa: E402
from src.training.objective import ObjectiveWeights, SegmentationObjective  # noqa: E402

# The four profiled cases: both source brains, both extremes of the truth design.
D24_CASES = (
    "SYN-RHUH-0001-d0.2-r0.6",
    "SYN-RHUH-0001-d1.2-r1.3",
    "SYN-RHUH-0002-d0.2-r1.3",
    "SYN-RHUH-0002-d1.2-r0.6",
)

# Columns of the raw grid file. Fixed here so a resumed run cannot write a
# different schema than the rows already on disk.
COLUMNS = (
    "mu_d",
    "mu_rho",
    "total",
    "data_weighted",
    "seg_wt",
    "seg_tc",
    "th_wt",
    "th_tc",
    "prior_weighted",
    "solve_s",
)

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def git_commit() -> str:
    """Short commit of this repository; without git, the VERSION stamp; else ``unknown``.

    Without a .git directory, the commit is read from a VERSION file if present.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        stamp = REPO / "VERSION"
        return stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else "unknown"


class ThresholdSearch:
    """Minimize one threshold on a stored field, on the GPU.

    The objective's mask term is
    ``mean(((sigmoid(a*(u - th)) - mask) * phi)**2) / norm`` and this class
    reproduces it in torch so a 60-point search costs a fraction of one PDE solve.
    It only finds the threshold: the value written to the output file is always
    recomputed by the numpy objective afterwards.
    """

    def __init__(self, objective: SegmentationObjective, device: torch.device) -> None:
        self.device = device
        self.steepness = float(objective.steepness)
        self.phi = torch.as_tensor(objective.phi, device=device, dtype=torch.float32)
        self.mask = {
            "wt": torch.as_tensor(objective.mask_wt, device=device, dtype=torch.float32),
            "tc": torch.as_tensor(objective.mask_tc, device=device, dtype=torch.float32),
        }
        self.norm = {"wt": objective._norm_wt, "tc": objective._norm_tc}
        w = objective.weights
        # The weighted contribution of one threshold to the total: its data term,
        # its own prior, and nothing else. The range penalty is zero because the
        # search never leaves th_range.
        self.seg_weight = {"wt": float(w.seg_wt), "tc": float(w.seg_tc)}
        self.reg_weight = {"wt": float(w.reg_th_wt), "tc": float(w.reg_th_tc)}
        self.prior = {"wt": objective.prior_th_wt, "tc": objective.prior_th_tc}
        # Each threshold searches its own range.
        self.range = {"wt": (float(objective.th_range[0]), float(objective.th_range[1])),
                      "tc": (float(objective.th_tc_range[0]), float(objective.th_tc_range[1]))}

    def load(self, u_final: np.ndarray) -> None:
        self.u = torch.as_tensor(u_final, device=self.device, dtype=torch.float32)

    def cost(self, which: str, th: float) -> float:
        pred = torch.sigmoid(self.steepness * (self.u - th))
        seg = torch.mean(((pred - self.mask[which]) * self.phi) ** 2).item() / self.norm[which]
        reg = (th - self.prior[which]) ** 2
        return self.seg_weight[which] * seg + self.reg_weight[which] * reg

    def minimise(self, which: str, n_coarse: int = 41, n_refine: int = 40) -> float:
        """Coarse scan, then golden-section inside the winning bracket.

        The coarse scan is what makes this safe: the mask term need not be unimodal
        over the whole interval, so a bracket is located globally first and the
        refinement is only asked to be local.
        """
        grid = np.linspace(*self.range[which], n_coarse)
        costs = [self.cost(which, float(th)) for th in grid]
        best = int(np.argmin(costs))
        left = float(grid[max(best - 1, 0)])
        right = float(grid[min(best + 1, n_coarse - 1)])

        a, b = left, right
        c = b - GOLDEN * (b - a)
        d = a + GOLDEN * (b - a)
        fc, fd = self.cost(which, c), self.cost(which, d)
        for _ in range(n_refine):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - GOLDEN * (b - a)
                fc = self.cost(which, c)
            else:
                a, c, fc = c, d, fd
                d = a + GOLDEN * (b - a)
                fd = self.cost(which, d)
            if b - a < 1e-9:
                break
        return float(0.5 * (a + b))


def build(case_id: str, data_root: Path, config: dict, prior_scale: float, device: str | None,
          paths=None, renormalise_tumour: bool = True):
    """Load a case and wire the objective exactly as ``experiments/run_cmaes.py`` does.

    ``paths`` defaults to the synthetic layout; a patient passes ``find_case(...)``.
    ``renormalise_tumour=False`` is only for the checks that switch off the second tissue
    renormalization (see ``load_case``).
    """
    paths = paths if paths is not None else synthetic_paths(data_root, case_id)
    case = load_case(
        paths,
        cache_dir=Path(config["data"]["cache_dir"]),
        phase_field_iters=config["model"]["phase_field"]["n_iter"],
        phase_field_epsilon=config["model"]["phase_field"]["epsilon_mm"],
        device=device,
        renormalise_tumour=renormalise_tumour,
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
    u0 = gaussian_seed(
        case.shape,
        case.x0,
        h_init=config["initial_condition"]["h_init"],
        r_init=config["initial_condition"]["r_init"],
        h=config["model"]["voxel_mm"],
    )
    weights = ObjectiveWeights(**config["objective"]["weights"])
    if prior_scale != 1.0:
        weights = replace(
            weights,
            reg_mu_d=weights.reg_mu_d * prior_scale,
            reg_mu_rho=weights.reg_mu_rho * prior_scale,
            reg_th_wt=weights.reg_th_wt * prior_scale,
            reg_th_tc=weights.reg_th_tc * prior_scale,
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
        mu_rho_range=(params["mu_rho"]["min"], params["mu_rho"]["max"]),
        th_tc_range=(params["th_tc"]["min"], params["th_tc"]["max"]),
    )
    return case, scaling, u0, objective


def grid_axis(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, n)


def mu_d_axis(n: int, floor: float) -> np.ndarray:
    """The diffusion axis, with the degenerate left endpoint lifted off zero.

    ``mu_D = 0`` makes the whole diffusion field zero and ``solve_fisher_kpp``
    refuses it, because that is no longer a diffusion problem. The column is the
    low-diffusion end of the direction this figure is about, so the endpoint is
    moved to ``floor``, two orders of magnitude below the grid step, and the
    metadata records that it was moved.
    """
    axis = np.linspace(0.0, 2.0, n)
    axis[0] = floor
    return axis


def zoom_axis(lo: float, hi: float, step: float, floor: float | None = None) -> np.ndarray:
    """A fine axis from ``lo`` to ``hi`` inclusive at ``step``.

    Built by counting steps, not by ``arange``, so float drift cannot drop or add
    the end point. A zero diffusion end is lifted to ``floor`` exactly as
    :func:`mu_d_axis` does, and for the same reason.
    """
    n = int(round((hi - lo) / step)) + 1
    axis = lo + step * np.arange(n)
    if floor is not None and axis[0] <= 0.0:
        axis[0] = floor
    return axis


def load_zoom(path: Path, case_id: str) -> dict:
    """The fine window of one case from ``configs/profile_zoom.yaml``."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if case_id not in spec["cases"]:
        raise SystemExit(f"{case_id} has no window in {path}")
    window = spec["cases"][case_id]
    return {"step": float(spec["step"]), "mu_d": [float(v) for v in window["mu_d"]],
            "mu_rho": [float(v) for v in window["mu_rho"]], "source": str(path)}


def done_points(path: Path) -> set[tuple[str, str]]:
    """Grid points already in the file, keyed by their printed form.

    Keyed on the *text* rather than the float so a resumed run compares what was
    written, not what a re-parsed float happens to be.
    """
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["mu_d"], row["mu_rho"]) for row in csv.DictReader(handle)}


def run_case(case_id: str, arguments: argparse.Namespace) -> dict:
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    arguments.out.mkdir(parents=True, exist_ok=True)
    tag = "priorON" if arguments.prior_scale == 1.0 else f"prior{arguments.prior_scale:g}"
    zoom = load_zoom(arguments.zoom, case_id) if arguments.zoom else None
    grid_tag = f"zoom{zoom['step']:g}" if zoom else f"grid{arguments.grid}"
    csv_path = arguments.out / f"{case_id}__{tag}__{grid_tag}.csv"
    meta_path = csv_path.with_suffix(".json")

    print(f"== {case_id} ==")
    if arguments.prior_scale != 0.0:
        print(f"!! prior_scale = {arguments.prior_scale}. This surface shows the prior's bowl as")
        print("!! well as the data's valley. Our paper's profiles use prior_scale 0.0.")

    started_all = time.perf_counter()
    case, scaling, u0, objective = build(
        case_id, arguments.data_root, config, arguments.prior_scale, arguments.device
    )
    device = torch.device(
        arguments.device
        if arguments.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    search = ThresholdSearch(objective, device)
    setup_s = time.perf_counter() - started_all
    print(f"   setup {setup_s:.1f} s | shape {tuple(case.shape)} | device {device}")

    if zoom:
        axis_d = zoom_axis(*zoom["mu_d"], zoom["step"], floor=arguments.mu_d_floor)
        axis_rho = zoom_axis(*zoom["mu_rho"], zoom["step"])
        print(f"   ZOOM {zoom['source']}: mu_d {zoom['mu_d']} x mu_rho {zoom['mu_rho']} "
              f"step {zoom['step']} -> {len(axis_d)} x {len(axis_rho)} points")
    else:
        axis_d = mu_d_axis(arguments.grid, arguments.mu_d_floor)
        axis_rho = grid_axis(0.0, 2.0, arguments.grid)
    points = [(float(a), float(b)) for a in axis_d for b in axis_rho]
    if arguments.probe:
        # Four points: both degenerate corners first, then the middle and the
        # far corner, which is the most expensive to solve.
        points = [points[0], points[len(points) - 1],
                  points[len(points) // 2], points[len(points) // 4]]
        print(f"   PROBE: {len(points)} points, nothing is written")

    already = set() if arguments.probe else done_points(csv_path)
    if already:
        print(f"   resuming: {len(already)} of {len(points)} rows already present")

    handle = None
    writer = None
    if not arguments.probe:
        fresh = not csv_path.is_file()
        handle = csv_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if fresh:
            writer.writeheader()
            handle.flush()

    verified = None
    timings: list[float] = []
    for index, (mu_d, mu_rho) in enumerate(points):
        key = (f"{mu_d:.6f}", f"{mu_rho:.6f}")
        if key in already:
            continue
        point_started = time.perf_counter()
        d_eff, rho_eff = scaling.to_physical(mu_d, mu_rho)
        solution = solve_fisher_kpp(
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
            cfl_safety=config["model"]["cfl_safety"],
            device=arguments.device,
        )
        search.load(solution.final)
        th_wt = search.minimise("wt")
        th_tc = search.minimise("tc")
        terms = objective(
            solution.final, mu_d=mu_d, mu_rho=mu_rho, th_wt=th_wt, th_tc=th_tc
        )
        solve_s = time.perf_counter() - point_started
        timings.append(solve_s)

        if verified is None:
            # The fast path and the canonical objective must agree, or the search
            # was minimizing something else. Checked once per case, on the first
            # point computed, and recorded in the metadata.
            fast = (search.cost("wt", th_wt) + search.cost("tc", th_tc))
            exact = (
                objective.weights.seg_wt * terms.seg_wt
                + objective.weights.seg_tc * terms.seg_tc
                + objective.weights.reg_th_wt * terms.reg_th_wt
                + objective.weights.reg_th_tc * terms.reg_th_tc
            )
            scale = max(abs(exact), 1e-30)
            verified = {"fast": fast, "exact": exact, "rel_error": abs(fast - exact) / scale}
            print(f"   fast-path check: rel_error {verified['rel_error']:.2e}")
            if verified["rel_error"] > 1e-5:
                raise SystemExit(
                    "the GPU threshold search disagrees with the canonical objective by "
                    f"{verified['rel_error']:.2e}; nothing further is trustworthy"
                )

        if writer is not None:
            writer.writerow({
                "mu_d": f"{mu_d:.6f}",
                "mu_rho": f"{mu_rho:.6f}",
                "total": repr(terms.total),
                "data_weighted": repr(terms.data_weighted),
                "seg_wt": repr(terms.seg_wt),
                "seg_tc": repr(terms.seg_tc),
                "th_wt": repr(th_wt),
                "th_tc": repr(th_tc),
                "prior_weighted": repr(terms.prior_weighted),
                "solve_s": f"{solve_s:.3f}",
            })
            handle.flush()

        remaining = len(points) - len(already) - len(timings)
        if index % 20 == 0 or arguments.probe:
            mean_s = sum(timings) / len(timings)
            print(f"   [{len(timings)}] mu=({mu_d:.2f},{mu_rho:.2f}) total={terms.total:.6e} "
                  f"th=({th_wt:.3f},{th_tc:.3f}) {solve_s:.1f}s "
                  f"| eta {remaining * mean_s / 60:.0f} min")

    if handle is not None:
        handle.close()

    # Merge with the stored metadata, never overwrite it: a resumed run computes
    # few points or none, and its own numbers would replace those of the
    # finished case.
    previous = {}
    if meta_path.is_file():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    previous_cost = previous.get("cost", {})
    previous_search = previous.get("threshold_search", {})

    # The fast-path check only runs when a point is computed; otherwise keep
    # the stored one.
    if verified is None:
        verified = previous_search.get("verification")

    sessions = list(previous_cost.get("sessions", []))
    if timings:
        sessions.append({
            "points": len(timings),
            "mean_s_per_point": sum(timings) / len(timings),
            "wall_clock_s": time.perf_counter() - started_all,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    elif not sessions and previous_cost.get("points_computed"):
        # Older metadata without "sessions": keep its numbers as one session.
        sessions.append({
            "points": previous_cost["points_computed"],
            "mean_s_per_point": previous_cost.get("mean_s_per_point"),
            "wall_clock_s": previous_cost.get("wall_clock_s"),
            "written_utc": previous.get("provenance", {}).get("written_utc"),
        })
    # The number of points is read from the CSV, not from the session tally:
    # a run stopped in the middle of a case writes no metadata.
    rows_in_csv = len(done_points(csv_path)) if (not arguments.probe and csv_path.is_file()) else None
    total_points = rows_in_csv if rows_in_csv is not None else sum(s["points"] for s in sessions)
    weighted = [s for s in sessions if s.get("mean_s_per_point") is not None]
    # None, not NaN, so that the file stays valid strict JSON.
    mean_s = (sum(s["points"] * s["mean_s_per_point"] for s in weighted)
              / sum(s["points"] for s in weighted)) if weighted else None

    truth_file = arguments.data_root / case_id / "truth.json"
    truth = json.loads(truth_file.read_text(encoding="utf-8")) if truth_file.is_file() else {}
    metadata = {
        "case_id": case_id,
        "grid": ({"zoom": zoom, "n_mu_d": len(axis_d), "n_mu_rho": len(axis_rho),
                  "points": len(axis_d) * len(axis_rho),
                  "mu_d_floor": arguments.mu_d_floor}
                 if zoom else
                 {"n": arguments.grid, "lo": 0.0, "hi": 2.0,
                  "points": arguments.grid * arguments.grid,
                  "mu_d_floor": arguments.mu_d_floor,
                  "mu_d_floor_note": "mu_d = 0 gives a zero diffusion field, which "
                                     "solve_fisher_kpp refuses; the first column sits "
                                     "at mu_d_floor instead"}),
        "prior_scale": arguments.prior_scale,
        "config": str(arguments.config),
        "weights": objective.weights.as_dict() if hasattr(objective.weights, "as_dict") else None,
        "truth": {"mu_d_true": truth.get("mu_d_true"), "mu_rho_true": truth.get("mu_rho_true"),
                  "mu_d_target": truth.get("mu_d_target"),
                  "mu_rho_target": truth.get("mu_rho_target")},
        "characteristic": scaling.as_dict(),
        "threshold_search": {"n_coarse": 41, "n_refine": 40, "method": "coarse scan + golden section",
                             "verification": verified},
        "cost": {"setup_s": setup_s,
                 "points_computed": total_points,
                 "points_this_session": len(timings),
                 "points_timed": sum(s["points"] for s in sessions),
                 "mean_s_per_point": mean_s,
                 "mean_covers_all_points": (rows_in_csv is not None
                                            and rows_in_csv == sum(s["points"] for s in sessions)),
                 "wall_clock_s": time.perf_counter() - started_all,
                 "sessions": sessions},
        "provenance": {
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    if arguments.probe:
        if mean_s is None:
            print("   PROBE computed no points, so there is nothing to estimate from")
            return metadata
        n_points = len(axis_d) * len(axis_rho)
        total_min = n_points * mean_s / 60.0
        print(f"   probe: {mean_s:.2f} s per point "
              f"-> {len(axis_d)}x{len(axis_rho)} = {n_points} points is about "
              f"{total_min:.0f} min for this case")
        return metadata
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    print(f"   wrote {csv_path}")
    print(f"   wrote {meta_path}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", action="append", default=[],
                        help="case id; repeatable")
    parser.add_argument("--all", action="store_true",
                        help="the four profiled cases")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/synthetic_local"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/profiles"))
    parser.add_argument("--grid", type=int, default=21,
                        help="points per axis; 21 gives the 441-point grid of our paper")
    parser.add_argument("--mu-d-floor", type=float, default=1e-3,
                        help="mu_d = 0 is a zero diffusion field and the solver refuses it, "
                             "so the first column sits here instead")
    parser.add_argument("--prior-scale", type=float, default=0.0,
                        help="scale of the prior, the penalty pulling the four parameters toward "
                             "their start values; 0.0 switches it off, as in our paper's profiles")
    parser.add_argument("--device", default=None)
    parser.add_argument("--zoom", type=Path, default=None,
                        help="configs/profile_zoom.yaml: a fine window per case instead of "
                             "the 21x21 grid over [0, 2]^2. Writes "
                             "<case>__prior0__zoom0.02.csv beside the coarse file")
    parser.add_argument("--probe", action="store_true",
                        help="time four points, write nothing, print the estimate")
    arguments = parser.parse_args(argv)

    cases = list(D24_CASES) if arguments.all else list(arguments.case)
    if not cases:
        raise SystemExit("give --case <id> at least once, or --all")
    for case_id in cases:
        run_case(case_id, arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
