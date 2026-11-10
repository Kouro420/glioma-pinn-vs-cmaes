#!/usr/bin/env python
"""Draw the figures of our paper from the run records.

| label | paper figure | what it shows |
|---|---|---|
| F11 | Figure 1 | the tumors each method predicts, against the observed masks |
| F7 | Figure 2 | estimated against true mu_D and mu_rho, both methods |
| F8 | Figure 3 | the PINN's mu_D during fine-tuning |
| F2 | Figure S1 | the calibration objective over (mu_D, mu_rho), four synthetic cases |
| F1 | Figure S2 | the prior on against the prior off, both methods |
| F4 | Figure S3 | the matched comparison on the 16 synthetic cases |
| F6 | Figure S4 | the matched comparison on the 20 RHUH patients |

Every number is read from a file in outputs/. Each figure is written to
outputs/figures/ as PDF and PNG, with a .caption.txt beside it that states the
counts and p-values the figure shows.

The prior (a penalty pulling the four parameters toward their start values;
--prior-scale 0 switches it off) is off in every figure except F1, which
compares the two settings.

F11 reads outputs/tables/f11_slices.npz and f11_predictions.json, which
scripts/f11_predictions.py writes; that step needs the patient scans and a GPU.

Usage:
    python -m scripts.make_figures                 # every figure
    python -m scripts.make_figures --only F2 F4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "figures"

# One color per method, the same in every figure.
CMAES, PINN = "#1b5e9c", "#c1452b"
GREY = "#6b6b6b"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------- loading ----

def load_pairs(*folders: str) -> dict[str, dict[str, dict]]:
    """``{case_id: {arm: record}}`` for one condition, read from one or more folders."""
    pairs: dict[str, dict[str, dict]] = {}
    for path in sorted(p for folder in folders for p in (REPO / folder).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        pairs.setdefault(record["case_id"], {})[record["method"]] = record
    return {case: arms for case, arms in pairs.items()
            if "cmaes" in arms and "pinn" in arms}


def load_arms(cmaes_folder: str, pinn_folder: str) -> dict[str, dict[str, dict]]:
    """``{case_id: {arm: record}}`` with each arm read from its own folder, cases in both only."""
    cmaes = {r["case_id"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
             for p in sorted((REPO / cmaes_folder).glob("*__cmaes.json")))}
    pinn = {r["case_id"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((REPO / pinn_folder).glob("*__pinn.json")))}
    return {c: {"cmaes": cmaes[c], "pinn": pinn[c]} for c in sorted(set(cmaes) & set(pinn))}


def records_of(folder: str, method: str) -> dict[str, dict]:
    """``{case_id: record}`` of one arm in one folder (any random-seed tag in the file name)."""
    return {r["case_id"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((REPO / folder).glob(f"*__{method}*.json")))}


def metric(record: dict, key: str) -> float | None:
    value = (record.get("metrics") or {}).get(key)
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def sign_test(margins: list[float]) -> tuple[int, int, float]:
    """Wins, losses and the two-sided exact sign-test p. Ties are dropped."""
    wins = sum(1 for m in margins if m > 0)
    losses = sum(1 for m in margins if m < 0)
    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return wins, losses, min(1.0, 2 * tail)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def p_text(p: float, tex: bool = True) -> str:
    """A p-value at two significant digits, as outputs/tables/paired_tests.json reports it
    (the figure prints the table's value, not a rounding of it); powers of ten below 0.001."""
    if p >= 0.001:
        return f"{p:.2g}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$" if tex else f"{mantissa}e{int(exponent)}"


def paired_test(comparison: str, key: str) -> dict:
    """The sign and Wilcoxon test of one metric, from outputs/tables/paired_tests.json."""
    table = json.loads((REPO / "outputs" / "tables" / "paired_tests.json").read_text(encoding="utf-8"))
    return table["comparisons"][comparison]["metrics"][key]


def both_p(t: dict, tex: bool = True) -> str:
    """'12–4; sign p = 0.077, Wilcoxon p = 0.018': both tests named, as our paper prints them."""
    p = "$p$" if tex else "p"
    return (f"{t['cmaes_better']}–{t['pinn_better']}; sign {p} = {p_text(t['p_sign'], tex)}, "
            f"Wilcoxon {p} = {p_text(t['p_wilcoxon'], tex)}")


def save(figure, name: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(OUT / f"{name}.{suffix}")
    (OUT / f"{name}.caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(figure)
    log(f"  wrote {name}.pdf / .png / .caption.txt")


# ------------------------------------------------------------------ F1 -------

def figure_f1() -> None:
    """The prior on against the prior off, paired per case, both methods.

    A slope plot rather than two means, so that a case moving the other way stays visible.
    """
    # PINN: the reach test (the training schedule of Zhang et al. with its own learning rate for
    # rD and rRHO), prior on (runs_reach_prior) against prior off (runs_reach).
    # CMA-ES: runs_syn against runs_syn_np.
    def joined(cmaes_folder: str, pinn_folder: str) -> dict[str, dict[str, dict]]:
        c, p = records_of(cmaes_folder, "cmaes"), records_of(pinn_folder, "pinn")
        return {k: {"cmaes": c[k], "pinn": p[k]} for k in sorted(set(c) & set(p))}

    with_prior = joined("outputs/runs_syn", "outputs/runs_reach_prior")
    without = joined("outputs/runs_syn_np", "outputs/runs_reach")
    cases = sorted(set(with_prior) & set(without))

    keys = [("err_mu_d", r"$\mu_D$ error"), ("err_mu_rho", r"$\mu_\rho$ error")]
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))

    summary = []
    for axis, (key, label) in zip(axes, keys):
        for arm, colour in (("cmaes", CMAES), ("pinn", PINN)):
            before = [metric(with_prior[c][arm], key) for c in cases]
            after = [metric(without[c][arm], key) for c in cases]
            usable = [(b, a) for b, a in zip(before, after) if b is not None and a is not None]
            for b, a in usable:
                axis.plot([0, 1], [b, a], color=colour, alpha=0.35, linewidth=0.8)
            axis.plot([0, 1], [np.median([b for b, _ in usable]),
                               np.median([a for _, a in usable])],
                      color=colour, linewidth=2.6, marker="o", markersize=4,
                      label="CMA-ES (median)" if arm == "cmaes" else "PINN, reach test (median)")
            if key == "err_mu_d":
                summary.append((arm, np.median([b for b, _ in usable]),
                                np.median([a for _, a in usable])))
        axis.set_xticks([0, 1])
        # The tick labels name the setting. The term switched here is the prior of the PyTorch
        # release of Zhang et al., not the box of their paper.
        axis.set_xticklabels(["regulariser on\n(A)", "regulariser off\n(B)"])
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylabel(f"absolute {label}")
        axis.set_title(label)
    axes[0].legend(frameon=False, loc="upper right")

    parts = ", ".join(f"{'CMA-ES' if a == 'cmaes' else 'PINN'} "
                      f"{b:.3f} to {c:.3f}" for a, b, c in summary)

    # How far each method moves from its start mu_D = 1, with the prior on and off.
    moved = {}
    for source, tag in ((with_prior, "on"), (without, "off")):
        for arm in ("cmaes", "pinn"):
            deltas = [abs(source[c][arm]["estimate"]["mu_d"] - 1.0)
                      for c in cases
                      if isinstance((source[c][arm].get("estimate") or {}).get("mu_d"), float)]
            moved[(arm, tag)] = float(np.median(deltas)) if deltas else float("nan")

    caption = f"""
F1. The prior switched on and off, on the same {len(cases)} synthetic cases and for both methods.
The prior is the term (mu - 1)^2 in the objective of the PyTorch release of Zhang et al., which
pulls the parameters toward their start values; it is not the box [0.75, 1.25] of their paper.
CMA-ES, and the PINN on the training schedule of Zhang et al. with its own learning rate 1e-3 for
D and rho (the reach test, a change of ours). One faint line per case; the thick line is the median.
Median mu_D error: {parts}. With the prior on, neither method leaves its start. Both start at
mu_D = 1, and the median distance traveled is {moved[('cmaes', 'on')]:.4f} for CMA-ES and {moved[('pinn', 'on')]:.4f} for the PINN,
against {moved[('cmaes', 'off')]:.3f} and {moved[('pinn', 'off')]:.3f} with the prior off.
The distance stays within the bound given in our paper's section on the prior.
"""
    save(figure, "F1_prior_ablation", caption)


# ------------------------------------------------------------------ F2 -------

def figure_f2() -> None:
    """The objective over (mu_D, mu_rho), one panel per profiled case.

    The color scale is logarithmic and the contour lines are drawn at multiples of the
    minimum: the objective spans four decades, and a linear scale would show one dark
    corner only.
    """
    files = sorted((REPO / "outputs" / "profiles").glob("*__prior0__grid21.csv"))
    if not files:
        log("  F2 skipped: no profile CSV in outputs/profiles")
        return

    figure, axes = plt.subplots(1, len(files), figsize=(2.15 * len(files), 2.6),
                                constrained_layout=True)
    axes = np.atleast_1d(axes)
    # One color scale for all panels, so that the single color bar holds for each of them.
    top = max(float(np.nanmax(t) / np.nanmin(t)) for t in
              (np.genfromtxt(p, delimiter=",", names=True)["total"] for p in files))
    shared = matplotlib.colors.LogNorm(vmin=1.0, vmax=top)
    widths = []
    points = []   # (case, objective minimum, CMA-ES estimate, truth)
    edges = {}    # case -> whether the 2.0 basin reaches the edge of the search box
    for axis, path in zip(axes, files):
        facts = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        raw = np.genfromtxt(path, delimiter=",", names=True)
        mu_d = np.unique(raw["mu_d"])
        mu_rho = np.unique(raw["mu_rho"])
        total = raw["total"].reshape(len(mu_d), len(mu_rho)).T

        floor = float(np.nanmin(total))
        relative = total / floor
        mesh = axis.pcolormesh(mu_d, mu_rho, relative, shading="nearest", norm=shared, cmap="viridis_r")
        axis.contour(mu_d, mu_rho, relative, levels=[1.05, 1.25, 2.0],
                     colors="white", linewidths=0.6, alpha=0.8)

        truth = facts["truth"]
        axis.plot(truth["mu_d_true"], truth["mu_rho_true"], marker="*",
                  markersize=11, color="#ffd400", markeredgecolor="black",
                  markeredgewidth=0.4, linestyle="none", label="truth")

        # The objective's own minimum, read from the fine grid (step 0.02) where one exists,
        # and the converged CMA-ES estimate.
        case_id = facts["case_id"]
        zoom = path.with_name(path.name.replace("__grid21", "__zoom0.02"))
        source = np.genfromtxt(zoom, delimiter=",", names=True) if zoom.is_file() else raw
        best = int(np.argmin(source["total"]))
        minimum = (float(source["mu_d"][best]), float(source["mu_rho"][best]))
        axis.plot(*minimum, marker="X", markersize=7, color="white",
                  markeredgecolor="black", markeredgewidth=0.5, linestyle="none",
                  label="objective minimum")
        cmaes_file = REPO / "outputs" / "runs_syn_np" / f"{case_id}__cmaes.json"
        estimate = None
        if cmaes_file.is_file():
            est = json.loads(cmaes_file.read_text(encoding="utf-8"))["estimate"]
            estimate = (float(est["mu_d"]), float(est["mu_rho"]))
            axis.plot(*estimate, marker="o", markersize=5, color=CMAES,
                      markeredgecolor="white", markeredgewidth=0.6, linestyle="none",
                      label="CMA-ES estimate")
        points.append((case_id.replace("SYN-", ""), minimum, estimate,
                       (float(truth["mu_d_true"]), float(truth["mu_rho_true"]))))

        # Width of the basin along each axis at three levels, since the answer depends on the
        # level. The coarse grid step is 0.1, so at the tightest level a width is about one cell.
        case_widths = {}
        for level in (1.05, 1.25, 2.0):
            basin = relative <= level
            d_span = mu_d[basin.any(axis=0)]
            r_span = mu_rho[basin.any(axis=1)]
            case_widths[level] = (
                float(d_span.max() - d_span.min()) if d_span.size else 0.0,
                float(r_span.max() - r_span.min()) if r_span.size else 0.0)
        widths.append((facts["case_id"].replace("SYN-", ""), case_widths))
        wide = relative <= 2.0
        d_in, r_in = mu_d[wide.any(axis=0)], mu_rho[wide.any(axis=1)]
        edges[facts["case_id"].replace("SYN-", "")] = bool(
            d_in.min() <= mu_d.min() or d_in.max() >= mu_d.max()
            or r_in.min() <= mu_rho.min() or r_in.max() >= mu_rho.max())

        axis.set_xlabel(r"$\mu_D$")
        axis.set_title(facts["case_id"].replace("SYN-", ""), fontsize=7)
        axis.set_xlim(mu_d.min(), mu_d.max())
        axis.set_ylim(mu_rho.min(), mu_rho.max())
    axes[0].set_ylabel(r"$\mu_\rho$")
    axes[0].legend(frameon=False, loc="upper right", labelcolor="white", fontsize=6)
    bar = figure.colorbar(mesh, ax=axes.tolist(), label="objective / its minimum",
                          fraction=0.03, pad=0.01)
    # Plain tick labels instead of the log scale's default "2 x 10^0".
    ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50, 100) if t <= top]
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{t:g}" for t in ticks])
    bar.minorticks_off()

    def pct(found: float, true: float) -> str:
        return f"{100.0 * (found - true) / true:+.0f} %"

    marks = "; ".join(
        f"{name}: minimum ({m[0]:.2f}, {m[1]:.2f}), truth ({t[0]:.2f}, {t[1]:.2f}), "
        f"minimum misses mu_D by {pct(m[0], t[0])} and mu_rho by {pct(m[1], t[1])}"
        + (f"; CMA-ES ({e[0]:.2f}, {e[1]:.2f})" if e else "")
        for name, m, e, t in points)
    readable = [(name, w) for name, w in widths if not edges[name]]
    at_two = "; ".join(f"{name} {w[2.0][0]:.2f} against {w[2.0][1]:.2f}"
                       + (" (reaches the box edge: a lower bound)" if edges[name] else "")
                       for name, w in widths)
    caption = f"""
F2. The calibration objective on a 21 x 21 grid over (mu_D, mu_rho), prior off, thresholds
optimized on the stored field, one PDE solve per grid point. Color is the objective
divided by its own minimum, on a log scale; white contours mark 1.05, 1.25 and 2.0
times the minimum. Star: the true pair. White cross: the objective's own minimum (from the
fine grid). Blue dot: the converged CMA-ES estimate (prior off).

Position of the minimum. {marks}. In every case the minimum lies away from the truth, low in
mu_D, and misses mu_D by a larger fraction than mu_rho; CMA-ES stops close to the minimum. So in
these four cases the error in diffusion comes from the objective, not from the search.
Four cases with one noise realization each: an observation, not a test. Table:
outputs/tables/objective_minimum_vs_truth.json.

Basin width, mu_D against mu_rho. At twice the minimum (coarse grid): {at_two}.
Basins fully inside the search box at this level: {len(readable)} of {len(widths)}, so no count
of "wider" is given at this level.
"""
    # The fine grid behind the tight contour (step 0.02, windows in configs/profile_zoom.yaml).
    # Read from its own files; the figure itself draws the coarse grids.
    fine = []
    for path in sorted((REPO / "outputs" / "profiles").glob("*__prior0__zoom0.02.csv")):
        raw = np.genfromtxt(path, delimiter=",", names=True)
        basin = raw["total"] <= 1.05 * raw["total"].min()
        best = int(np.argmin(raw["total"]))
        dw, rw = float(np.ptp(raw["mu_d"][basin])), float(np.ptp(raw["mu_rho"][basin]))
        fine.append((path.name.split("__")[0].replace("SYN-", ""), dw, rw,
                     dw / float(raw["mu_d"][best]), rw / float(raw["mu_rho"][best])))
    if fine:
        narrower = sum(1 for f in fine if f[1] < f[2])
        wider_rel = sum(1 for f in fine if f[3] > f[4])
        rows = "; ".join(f"{name} {dw:.2f} against {rw:.2f} (relative {dr:.2f} against {rr:.2f})"
                         for name, dw, rw, dr, rr in fine)
        caption += f"""At 1.05 times the minimum (fine grid, step 0.02, windows fixed in advance in
configs/profile_zoom.yaml; the 1.05 contour lies wholly inside every window): {rows}. Two
measures: in absolute width, mu_D is narrower in {narrower} of {len(fine)}; relative to the
position of the minimum (width divided by the minimum's own coordinate), mu_D is wider in
{wider_rel} of {len(fine)}.
"""
    save(figure, "F2_objective_profile", caption)


# ------------------------------------------------------------------ F4 -------

def figure_f4() -> None:
    """Per-case paired margins, synthetic, prior off.

    Margins rather than two clouds of absolute values: the cases differ from each
    other far more than the methods differ within a case, so an unpaired plot would
    show the spread between cases and hide the comparison.
    """
    # CMA-ES against the PINN on the training schedule of Zhang et al., both prior off.
    pairs = load_arms("outputs/runs_syn_np", "outputs/runs_syn_np_paper")
    rows = [("dice_wt", "Dice WT", "higher better"),
            ("dice_tc", "Dice TC", "higher better"),
            ("err_mu_d", r"$\mu_D$ error", "lower better"),
            ("err_mu_rho", r"$\mu_\rho$ error", "lower better")]
    stats_key = "E2 synthetic, paper schedule, prior off"  # key in outputs/tables/paired_tests.json

    # One case order for every panel, so a row is the same case across panels: by source
    # brain, then true mu_D, then true mu_rho (top of each panel = first in this order).
    def order_key(case: str) -> tuple:
        m = pairs[case]["cmaes"]["metrics"]
        return (case.split("-d")[0], m["mu_d_true"], m["mu_rho_true"])

    cases = sorted(pairs, key=order_key)
    figure, axes = plt.subplots(1, len(rows), figsize=(7.2, 3.1), constrained_layout=True,
                                sharey=True)
    told = []
    for axis, (key, label, sense) in zip(axes, rows):
        margins = []
        for case in cases:
            arms = pairs[case]
            a, b = metric(arms["cmaes"], key), metric(arms["pinn"], key)
            # Always "how much better is CMA-ES", whatever the metric's direction.
            margins.append(np.nan if a is None or b is None
                           else (a - b) if sense == "higher better" else (b - a))
        valid = [m for m in margins if not np.isnan(m)]
        wins, losses, p = sign_test(valid)
        y = np.arange(len(cases))[::-1]
        colours = [CMAES if m > 0 else PINN for m in margins]
        axis.barh(y, margins, color=colours, height=0.7)
        axis.axvline(0, color="black", linewidth=0.8)
        if sense == "lower better" and min(valid) < 0:
            # Ticks on both sides of zero, few enough not to collide in the narrow panels,
            # with at least one negative tick, rounded toward zero so it lies inside the bars.
            step = 0.4 if max(valid) - min(valid) > 0.8 else 0.2
            ticks = [k * step for k in range(int(np.floor(min(valid) / step)), int(np.ceil(max(valid) / step)) + 1)
                     if min(valid) <= k * step <= max(valid) or k == 0]
            if not any(t < 0 for t in ticks):
                ticks.insert(0, -np.floor(-min(valid) * 10) / 10)
            axis.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda value, _: f"{value:g}".replace("-", "−")))
        split = sum(c.startswith(cases[0].split("-d")[0]) for c in cases)
        axis.axhline(len(cases) - split - 0.5, color=GREY, linewidth=0.6, linestyle=":")
        test = paired_test(stats_key, key)
        if (test["cmaes_better"], test["pinn_better"]) != (wins, losses):
            raise ValueError(f"F4 {key}: {wins}-{losses} here, {test['cmaes_better']}-{test['pinn_better']} in paired_tests.json")
        # Both tests in the title, one per line, because the panels are narrow.
        axis.set_title(f"{label}: {wins}–{losses}\nsign $p$ = {p_text(test['p_sign'])}\n"
                       f"Wilcoxon $p$ = {p_text(test['p_wilcoxon'])}", fontsize=7.5)
        axis.set_xlabel(f"{label}, CMA-ES − PINN" if sense == "higher better"
                        else f"|{label}|, PINN − CMA-ES")
        told.append((label, test, float(np.median(valid))))
    axes[0].set_yticks(np.arange(len(cases))[::-1])
    axes[0].set_yticklabels([f"{c.split('-d')[0].replace('SYN-', '')}  "
                             f"{pairs[c]['cmaes']['metrics']['mu_d_true']:.2f} / "
                             f"{pairs[c]['cmaes']['metrics']['mu_rho_true']:.2f}"
                             for c in cases], fontsize=5.5)
    axes[0].set_ylabel(r"case: source brain, true $\mu_D$ / $\mu_\rho$")

    rows_text = "; ".join(f"{label} {both_p(t, tex=False)}, median margin {m:+.4f}"
                          for label, t, m in told)
    caption = f"""
F4. The matched comparison on {len(pairs)} synthetic cases, prior off: one forward
model, one score, one starting point, structural MRI only. Each bar is one case; positive
means CMA-ES did better (Dice: CMA-ES minus PINN; absolute error: PINN minus CMA-ES). All panels
use one case order (source brain, then true mu_D, then true mu_rho; dotted line between the two
brains), so a row is the same case in every panel. Exact sign test and Wilcoxon signed-rank test,
two-sided, ties dropped, from outputs/tables/paired_tests.json: {rows_text}.
Run time is not compared.

CMA-ES: outputs/runs_syn_np. PINN on the training schedule of Zhang et al.:
outputs/runs_syn_np_paper. Parameter errors are absolute errors against each case's truth;
how each method's estimate follows the truth is in outputs/tables/e2_synthetic.json.
"""
    save(figure, "F4_matched_comparison", caption)


# ------------------------------------------------------------------ F6 -------

# Patients, prior off: CMA-ES against the PINN on the training schedule of Zhang et al.
PATIENT_B = ("outputs/runs_patient_np", "outputs/runs_patient_np_paper")


def figure_f6() -> None:
    """Per-patient paired values, both metrics: the distribution, not only the winner."""
    pairs = load_pairs(*PATIENT_B)
    cases = sorted(pairs)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)
    told = []
    for axis, key, label, short in ((axes[0], "dice_wt", "Dice, whole tumour", "Dice WT"),
                                    (axes[1], "dice_tc", "Dice, tumour core", "Dice TC")):
        a_values, b_values = [], []
        for case in cases:
            a, b = metric(pairs[case]["cmaes"], key), metric(pairs[case]["pinn"], key)
            if a is None or b is None:
                continue
            a_values.append(a)
            b_values.append(b)
        axis.plot(b_values, a_values, "o", color=GREY, markersize=5, markeredgewidth=0)
        lo = min(min(a_values), min(b_values)) - 0.02
        hi = max(max(a_values), max(b_values)) + 0.02
        axis.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect("equal")
        wins, losses, _ = sign_test([x - y for x, y in zip(a_values, b_values)])
        test = paired_test("E3 patients, paper schedule, prior off", key)
        if (test["cmaes_better"], test["pinn_better"]) != (wins, losses):
            raise ValueError(f"F6 {key}: {wins}-{losses} here, {test['cmaes_better']}-{test['pinn_better']} in paired_tests.json")
        told.append((label, test, float(np.median(a_values)), float(np.median(b_values))))
        axis.set_title(f"{label}: CMA-ES {wins}–{losses} PINN\nsign $p$ = {p_text(test['p_sign'])}, "
                       f"Wilcoxon $p$ = {p_text(test['p_wilcoxon'])}")
        axis.set_xlabel(f"{short}, PINN")
        axis.set_ylabel(f"{short}, CMA-ES")
        axis.annotate("above the line:\nCMA-ES better", xy=(0.04, 0.88),
                      xycoords="axes fraction", fontsize=6.5, color=GREY)

    rows_text = "; ".join(f"{label}: {both_p(t, tex=False)}, medians {ma:.4f} (CMA-ES) and {mb:.4f} (PINN)"
                          for label, t, ma, mb in told)
    caption = f"""
F6. The matched comparison on {len(cases)} RHUH patients, prior off. One point per patient; the
line is equality; above it CMA-ES fits the masks better. Exact sign test and Wilcoxon signed-rank
test, two-sided, from outputs/tables/paired_tests.json: {rows_text}. Patients have no known true
parameters, so this figure compares the fit to the masks only and says nothing about parameter
accuracy; for that, see the synthetic cases.

CMA-ES: outputs/runs_patient_np. PINN on the training schedule of Zhang et al.:
outputs/runs_patient_np_paper.
"""
    save(figure, "F6_patient_comparison", caption)


# ------------------------------------------------------------------ F7 -------

def figure_f7() -> None:
    """Estimate against truth, both methods, both mu.

    Row 1: the 16 base cases (CMA-ES, prior off; the PINN on the training schedule of
    Zhang et al.; the PINN with its own learning rate for rD and rRHO, the reach test).
    Row 2: the 12 design-range cases, whose truths lie inside the box [0.75, 1.25] of
    Zhang et al. Identity line solid; the start value 1 dotted.
    """
    # Legend labels use the words of our paper's text.
    base = {"CMA-ES": (records_of("outputs/runs_syn_np", "cmaes"), CMAES, "o", "full"),
            "PINN, paper schedule": (records_of("outputs/runs_syn_np_paper", "pinn"), PINN, "o", "none"),
            "PINN, reach test": (records_of("outputs/runs_reach", "pinn"), PINN, "^", "full")}
    env = {"CMA-ES": (records_of("outputs/runs_env", "cmaes"), CMAES, "s", "full"),
           "PINN, paper schedule": (records_of("outputs/runs_env", "pinn"), PINN, "s", "none")}
    # Design-range row: one marker per source brain, square RHUH-0001, diamond RHUH-0002.
    brain_marker = {"RHUH-0001": "s", "RHUH-0002": "D"}
    figure, axes = plt.subplots(2, 2, figsize=(6.6, 6.2))
    lines = []
    for row, (title, sets) in enumerate((("16 base cases", base), ("12 design-range cases", env))):
        for column, key in enumerate(("mu_d", "mu_rho")):
            axis = axes[row][column]
            allv = []
            for label, (recs, colour, marker, fill) in sets.items():
                truth = [r["metrics"][f"{key}_true"] for r in recs.values()]
                est = [r["estimate"][key] for r in recs.values()]
                allv += truth + est
                face = colour if fill == "full" else "none"
                if row == 0:
                    axis.scatter(truth, est, s=18, marker=marker, color=face,
                                 edgecolor=colour, linewidth=0.9, label=label, zorder=3)
                else:
                    for brain, shape in brain_marker.items():
                        pick = [i for i, c in enumerate(recs) if brain in c]
                        axis.scatter([truth[i] for i in pick], [est[i] for i in pick], s=18,
                                     marker=shape, color=face, edgecolor=colour, linewidth=0.9,
                                     label=f"{label}, {brain}", zorder=3)
                lines.append(f"{title}, {key}, {label}: n = {len(truth)}, r = {pearson(truth, est):.3f}")
            # The axes start at 0 and end with a margin, so no marker sits on a spine.
            lo, hi = 0.0, max(allv) * 1.08
            axis.plot([lo, hi], [lo, hi], color=GREY, linewidth=0.8)
            axis.axhline(1.0, color=GREY, linewidth=0.6, linestyle=":")
            axis.set_xlim(lo, hi)
            axis.set_ylim(lo, hi)
            sym = r"\mu_D" if key == "mu_d" else r"\mu_\rho"
            axis.set_xlabel(f"true ${sym}$")
            axis.set_ylabel(f"estimated ${sym}$")
            axis.set_title(f"{title}: ${sym}$")
            # No legend may cross the start line mu = 1 or a marker. Base row: top left of the
            # mu_D panel, above that line. Design-range row: the empty lower right of the mu_rho panel.
            if (row, column) == (0, 0):
                axis.legend(frameon=False, loc="upper left", fontsize=6.5)
            elif (row, column) == (1, 1):
                handles, labels = axes[1][0].get_legend_handles_labels()
                axis.legend(handles, labels, frameon=False, loc="lower right", fontsize=6.5)
    figure.tight_layout()
    caption = ("F7. Estimate against truth for both methods. Solid line: estimate = truth; dotted line: "
               "the start value 1 of both methods. Top: the 16 base cases (CMA-ES filled circles; the PINN "
               "on the training schedule of Zhang et al. open circles; the PINN in the reach test, own "
               "learning rate 1e-3 for mu_D and mu_rho, filled triangles). Bottom: the 12 design-range "
               "cases (truths inside [0.75, 1.25]), CMA-ES filled and the PINN open, squares for source "
               "brain RHUH-0001 and diamonds for RHUH-0002. Pearson r per set:\n" + "\n".join(lines))
    save(figure, "F7_estimate_vs_truth", caption)


# ------------------------------------------------------------------ F8 -------

def figure_f8() -> None:
    """The PINN's mu_D against fine-tuning step, every prior-off run.

    mu_D is held at 1 in pre-training and moves only in fine-tuning. With one Adam group
    (the training schedule of Zhang et al.) it falls at about the rate bound in every case;
    with its own learning rate (the reach test) it falls further, still in every case,
    whatever the truth.
    """
    # Panel titles use the words of our paper's text.
    sets = [("paper schedule,\none Adam group (lr 1e-4)", "outputs/runs_syn_np_paper"),
            ("reach test: own rate\nfor $\\mu_D$, $\\mu_\\rho$ (lr 1e-3)", "outputs/runs_reach"),
            ("design-range test: paper\nschedule, truths in [0.75, 1.25]", "outputs/runs_env")]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=0.2, vmax=1.6)
    lines = []
    for axis, (title, folder) in zip(axes, sets):
        recs = records_of(folder, "pinn")
        down = 0
        for r in recs.values():
            h = r["history"]
            start = next((e["step"] for e in h if e["rD"] < 0.9999), None)
            if start is None:
                continue
            steps = [e["step"] - start for e in h if e["step"] >= start - 10]
            values = [e["rD"] for e in h if e["step"] >= start - 10]
            truth = r["metrics"]["mu_d_true"]
            axis.plot(steps, values, color=cmap(norm(truth)), linewidth=0.8)
            down += values[-1] < 1.0
        axis.axhline(1.0, color=GREY, linewidth=0.6, linestyle=":")
        axis.set_title(title, fontsize=7)
        axis.set_xlabel("fine-tuning step")
        lines.append(f"{title.replace(chr(10), ' ')}: {len(recs)} runs; final mu_D below the start in {down}")
    axes[0].set_ylabel(r"PINN $\mu_D$")
    bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.85)
    bar.set_label(r"true $\mu_D$")
    caption = ("F8. The PINN's mu_D during fine-tuning (pre-training holds it at the start value 1), "
               "every prior-off run; color = the case's true mu_D. "
               + "; ".join(lines) + ".")
    save(figure, "F8_pinn_mu_d_training", caption)


# ------------------------------------------------------------------ F11 ------

WT_COLOUR, TC_COLOUR = "#00b7eb", "#ff3b30"       # whole tumor (FLAIR mask), core (T1c mask)


def mark_sides(axis) -> None:
    """R and L at the image edges (radiological convention)."""
    for x, letter, align in ((0.02, "R", "left"), (0.98, "L", "right")):
        axis.text(x, 0.5, letter, transform=axis.transAxes, color="#ffd400", fontsize=6,
                  ha=align, va="center")


def outline(axis, mask2d: np.ndarray, colour: str, width: float = 0.9) -> None:
    if mask2d.any():
        axis.contour(mask2d.astype(float), levels=[0.5], colors=[colour], linewidths=width)


def bare(axis, title: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title)
    mark_sides(axis)


def figure_f11() -> None:
    """What the fitted models predict, on three cases chosen by a rule fixed in advance.

    Reads outputs/tables/f11_slices.npz and f11_predictions.json, written by
    scripts/f11_predictions.py (the forward solves and the Dice check live there).
    """
    data = np.load(REPO / "outputs" / "tables" / "f11_slices.npz")
    info = json.loads((REPO / "outputs" / "tables" / "f11_predictions.json").read_text(encoding="utf-8"))
    rows = info["rows"]
    k = info["evkb_pilot_index"]
    # The clinical pilot row (labeled EvKB, Evangelisches Klinikum Bethel) is drawn only where its images
    # are at hand; they are not public.
    tags = [t for t in ("synthetic", "rhuh", "evkb") if t in rows]
    labels = {"synthetic": f"synthetic\n{rows['synthetic']['case']}",
              "rhuh": f"RHUH patient\n{rows['rhuh']['case']}",
              "evkb": f"EvKB\npilot case {k}"}
    columns = [("obs", "observed masks"), ("cmaes", "CMA-ES prediction"),
               ("pinn", "PINN prediction (paper schedule)")]
    figure, axes = plt.subplots(len(tags), 3, figsize=(6.6, 2.3 * len(tags)), constrained_layout=True)
    for r, tag in enumerate(tags):
        inplane = data[f"{tag}_brain3d_inplane"]
        xs, ys = np.where(inplane)
        box = (slice(max(xs.min() - 4, 0), xs.max() + 5), slice(max(ys.min() - 4, 0), ys.max() + 5))

        def show(a):
            return np.fliplr(np.rot90(a[box[0], box[1]]))

        back = show(data[f"{tag}_back"])
        obs_wt, obs_tc = show(data[f"{tag}_obs_wt"]), show(data[f"{tag}_obs_tc"])
        for c, (arm, title) in enumerate(columns):
            axis = axes[r][c]
            if arm != "obs" and arm not in rows[tag]["arms"]:
                axis.axis("off")
                axis.text(0.5, 0.5, "no PINN run:\nthe pilot ran\nCMA-ES only", ha="center",
                          va="center", fontsize=7, color=GREY, transform=axis.transAxes)
                continue
            axis.imshow(back, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            outline(axis, obs_wt, WT_COLOUR)
            outline(axis, obs_tc, TC_COLOUR)
            if arm != "obs":
                for part, colour in (("wt", WT_COLOUR), ("tc", TC_COLOUR)):
                    pred = show(data[f"{tag}_{arm}_{part}"])
                    if pred.any():
                        axis.contour(pred.astype(float), levels=[0.5], colors=[colour],
                                     linewidths=0.9, linestyles="--")
                check = rows[tag]["arms"][arm]
                axis.text(0.97, 0.97, f"Dice WT {check['dice_wt']:.2f}\nDice TC {check['dice_tc']:.2f}",
                          transform=axis.transAxes, color="white", fontsize=6.5, va="top",
                          ha="right", bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0))
            bare(axis, title if r == 0 else "")
            if c == 0:
                axis.set_ylabel(labels[tag], fontsize=7)
    handles = [plt.Line2D([], [], color=WT_COLOUR, label="whole tumour"),
               plt.Line2D([], [], color=TC_COLOUR, label="tumour core"),
               plt.Line2D([], [], color=GREY, label="observed (solid)"),
               plt.Line2D([], [], color=GREY, linestyle="--", label="predicted (dashed)")]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)

    lines = []
    for tag in tags:
        name = labels[tag].replace("\n", " ")
        for arm, check in rows[tag]["arms"].items():
            lines.append(f"{name}, {arm}: mu_D {check['mu_d']:.3f}, mu_rho {check['mu_rho']:.3f}, "
                         f"thresholds {check['th_wt']:.3f}/{check['th_tc']:.3f}; Dice WT "
                         f"{check['dice_wt']:.4f} (record {check['record_dice_wt']:.4f}), Dice TC "
                         f"{check['dice_tc']:.4f} (record {check['record_dice_tc']:.4f}); axial z = "
                         f"{rows[tag]['z_ras']}")
    n_ev = len(info["evkb_dice_wt_sorted"])
    caption = f"""
F11. What the fitted models predict. One axial slice per case through the centroid of the tumor
core, cropped to the brain, radiological convention (anterior at the top, the patient's right on the
image left, R). Blue: whole tumor; red: tumor core. Solid: the observed masks the methods receive
(for the synthetic case, the noisy masks); dashed: each method's prediction, the finite-difference
solution at the method's estimate times the phase field, thresholded at the method's own fitted
thresholds, which is the computation behind every printed Dice (experiments/run_cmaes.py,
experiments/run_pinn.py). Background: the white-matter map (synthetic) and the skull-stripped FLAIR
(patients). Prior off; the PINN on the training schedule of Zhang et al.
The cases follow a rule fixed before any prediction was drawn, not the best case: row 1,
{rows['synthetic']['case']}, one of the four profiled synthetic cases; row 2, of the 20 RHUH
patients, the one whose Dice gap (CMA-ES minus PINN, mean of the whole-tumor and core gaps) lies
nearest the median gap ({info['rhuh_gap_median']:.4f}); with 20 patients the median lies halfway
between the two middle patients, so both are equally near and the rule's tie-break (the lower ID)
chose {rows['rhuh']['case']}""" + (
        f"""; row 3, the clinical pilot case (EvKB, Evangelisches Klinikum Bethel) with
the median CMA-ES Dice WT of the {n_ev} (the 9th), labeled by its position in the pilot's ID order
only. """ + ("The PINN ran on this one pilot case only, as in the main comparison; its parameters are "
             "not interpreted." if "pinn" in rows["evkb"]["arms"]
             else "The pilot ran CMA-ES alone, so row 3 has no PINN panel.")
        if "evkb" in rows else ". The clinical pilot row of Figure 1 in our paper needs patient "
                               "images that are not public.") + f"""
Every Dice was recomputed on the whole volume and equals the run record (largest difference
{max(max(abs(ch['dice_wt'] - ch['record_dice_wt']), abs(ch['dice_tc'] - ch['record_dice_tc'])) for t in rows.values() for ch in t['arms'].values()):.4f}):
""" + "\n".join(lines) + """
Sources: outputs/tables/f11_predictions.json and f11_slices.npz (scripts/f11_predictions.py);
records outputs/runs_syn_np, runs_syn_np_paper, runs_patient_np, runs_patient_np_paper, and the pilot
records outside the repository.
"""
    save(figure, "F11_predictions", caption)


FIGURES = {"F1": figure_f1, "F2": figure_f2, "F4": figure_f4, "F6": figure_f6,
           "F7": figure_f7, "F8": figure_f8, "F11": figure_f11}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None)
    arguments = parser.parse_args()

    for name in (arguments.only or sorted(FIGURES)):
        log(f"== {name} ==")
        FIGURES[name]()
    log("")
    log(f"figures in {OUT}")
    log("Each figure has a .caption.txt beside it, with the counts and p-values it shows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
