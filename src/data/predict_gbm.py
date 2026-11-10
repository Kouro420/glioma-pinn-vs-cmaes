#!/usr/bin/env python
"""Loader for one PREDICT-GBM case, shared by both calibration methods.

PREDICT-GBM (Zimmer et al. 2026, Hugging Face ``LZimmer/PREDICT-GBM``, MIT)
ships every input this repository needs, already skull-stripped, registered to the
SRI24 atlas and resampled to 1 mm isotropic:

    tumor_seg.nii.gz         pre-operative tumor, labels 1/2/3
    recurrence_preop.nii.gz  follow-up recurrence in pre-operative space, 1/2/3/4
    wm_pbmap.nii.gz          white-matter probability
    gm_pbmap.nii.gz          gray-matter probability
    csf_pbmap.nii.gz         cerebrospinal-fluid probability
    t1c_bet_mask.nii.gz      brain mask

The cohort used here is the RHUH subset, processed in fixed patient-ID order.
Nothing in this module derives masks or tissue maps from images; segmentation
is taken as given, exactly as the benchmark and the reference PINN receive it.

The phase field is not shipped, so it is computed once per case with the same
Cahn-Hilliard relaxation the reference uses (100 iterations, epsilon = 3 mm)
and cached next to the outputs, because it is identical for every candidate
parameter set and would otherwise be recomputed thousands of times.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from src.models.fisher_kpp import phase_field

__all__ = ["CasePaths", "Case", "find_case", "load_case"]

DEFAULT_FILENAMES = {
    "tumor_seg": "tumor_seg.nii.gz",
    "recurrence": "recurrence_preop.nii.gz",
    "wm": "wm_pbmap.nii.gz",
    "gm": "gm_pbmap.nii.gz",
    "csf": "csf_pbmap.nii.gz",
    "brain_mask": "t1c_bet_mask.nii.gz",
}

LABEL_WT = (1, 2, 3)
LABEL_TC = (1, 3)


@dataclass
class CasePaths:
    """Where the six input files of one case live."""

    case_id: str
    tumor_seg: Path
    wm: Path
    gm: Path
    csf: Path
    brain_mask: Path | None = None
    recurrence: Path | None = None

    def require(self) -> None:
        """Raise if a mandatory file is missing."""
        for name in ("tumor_seg", "wm", "gm", "csf"):
            path = getattr(self, name)
            if path is None or not Path(path).is_file():
                raise FileNotFoundError(f"case {self.case_id}: missing {name} at {path}")


@dataclass
class Case:
    """One case, ready for calibration."""

    case_id: str
    p_wm: np.ndarray
    p_gm: np.ndarray
    p_csf: np.ndarray
    segmentation: np.ndarray
    brain_mask: np.ndarray
    phi: np.ndarray
    x0: np.ndarray
    affine: np.ndarray
    recurrence: np.ndarray | None = None

    @property
    def mask_wt(self) -> np.ndarray:
        """Whole tumor: necrotic, edema and enhancing."""
        return np.isin(self.segmentation, LABEL_WT)

    @property
    def mask_tc(self) -> np.ndarray:
        """Tumor core: necrotic and enhancing."""
        return np.isin(self.segmentation, LABEL_TC)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.p_wm.shape)


def find_case(root: Path, case_id: str, filenames: dict[str, str] | None = None) -> CasePaths:
    """Locate the files of ``case_id`` below ``root``.

    The archive nests each case in its own directory, and the search is by
    filename rather than by a hard-coded layout, so the depth does not matter.
    The first match in sorted order wins, which keeps the
    result reproducible.
    """
    names = {**DEFAULT_FILENAMES, **(filenames or {})}
    case_dirs = sorted(p for p in root.rglob(case_id) if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"no directory named {case_id!r} below {root}")
    case_dir = case_dirs[0]

    def locate(key: str) -> Path | None:
        hits = sorted(case_dir.rglob(names[key]))
        return hits[0] if hits else None

    paths = CasePaths(
        case_id=case_id,
        tumor_seg=locate("tumor_seg"),
        wm=locate("wm"),
        gm=locate("gm"),
        csf=locate("csf"),
        brain_mask=locate("brain_mask"),
        recurrence=locate("recurrence"),
    )
    paths.require()
    return paths


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def tumour_centroid(segmentation: np.ndarray) -> np.ndarray:
    """Centroid of the tumor core, in zero-based voxel coordinates.

    The reference stores this center one-based and subtracts one before use
    (``Engine.py``, lines 197 and 464); the value here is the same point.
    """
    core = np.isin(segmentation, LABEL_TC)
    region = core if core.any() else np.isin(segmentation, LABEL_WT)
    coords = np.argwhere(region)
    if coords.size == 0:
        raise ValueError("the segmentation contains no tumor voxels")
    return coords.mean(axis=0)


def rescale_tissue_in_tumour(
    p_wm: np.ndarray, p_gm: np.ndarray, mask_wt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Renormalize the white/gray split inside the tumor, as the reference does.

    Inside the whole-tumor mask the SRI24 tissue maps no longer sum to one:
    the segmentation software assigns part of the probability mass to the lesion
    itself, so ``p_wm + p_gm`` falls well below one and the diffusion field
    ``D = d_w (p_wm + p_gm / factor)`` is depressed exactly where the tumor is.
    ``Engine._setup_data`` (lines 176-185 of the reference) therefore divides both
    maps by their sum inside the tumor, and falls back to an even 0.5/0.5 split
    where the sum vanishes.

    The CMA-ES arm must see the same diffusion field as the PINN arm, otherwise
    the two methods calibrate different models and the matched comparison does
    not hold. This function is the CMA-ES side of that agreement.

    Args:
        p_wm, p_gm: white- and gray-matter probability maps.
        mask_wt: whole-tumor mask (segmentation labels 1, 2, 3).

    Returns:
        New white- and gray-matter maps; the inputs are left unchanged.
    """
    p_wm = np.array(p_wm, dtype=np.float64, copy=True)
    p_gm = np.array(p_gm, dtype=np.float64, copy=True)

    tumour = mask_wt > 0
    total = p_wm + p_gm
    positive = tumour & (total > 1e-6)
    p_wm[positive] /= total[positive]
    p_gm[positive] /= total[positive]

    empty = tumour & (total <= 1e-6)
    p_wm[empty] = 0.5
    p_gm[empty] = 0.5
    return p_wm, p_gm


def _phase_field_key(p_wm: np.ndarray, p_gm: np.ndarray, p_csf: np.ndarray,
                     n_iter: int, epsilon: float) -> str:
    """Short hash of everything the phase field is computed from."""
    digest = hashlib.blake2b(digest_size=8)
    for array in (p_wm, p_gm, p_csf):
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    digest.update(f"{int(n_iter)}|{float(epsilon)!r}".encode())
    return digest.hexdigest()


def load_case(
    paths: CasePaths,
    *,
    cache_dir: Path | None = None,
    phase_field_iters: int = 100,
    phase_field_epsilon: float = 3.0,
    device: str | None = None,
    renormalise_tumour: bool = True,
) -> Case:
    """Read one case and attach its phase field and tumor seed position x0.

    Args:
        paths: located input files.
        cache_dir: directory for the cached phase field; skipped when ``None``.
        phase_field_iters, phase_field_epsilon: relaxation settings, matching
            the reference implementation.
        device: torch device for the phase-field relaxation.
        renormalise_tumour: apply :func:`rescale_tissue_in_tumour` (the default).
            ``False`` skips it, for the runs that test the renormalization itself:
            a synthetic case's maps were already renormalized once by the generator.

    Returns:
        The assembled :class:`Case`.
    """
    paths.require()
    p_wm, affine = _load(paths.wm)
    p_gm, _ = _load(paths.gm)
    p_csf, _ = _load(paths.csf)
    segmentation, _ = _load(paths.tumor_seg)
    segmentation = np.rint(segmentation).astype(np.int16)

    # Before anything is derived from them: the reference renormalizes the
    # tissue split inside the tumor, and computes the phase field from the
    # renormalized maps (Engine.py lines 176-185, then line 243). Both must
    # happen here too, or the two arms solve different equations.
    if renormalise_tumour:
        p_wm, p_gm = rescale_tissue_in_tumour(p_wm, p_gm, np.isin(segmentation, LABEL_WT))

    if paths.brain_mask is not None and Path(paths.brain_mask).is_file():
        brain_mask = _load(paths.brain_mask)[0] > 0
    else:
        brain_mask = p_wm > 0

    recurrence = None
    if paths.recurrence is not None and Path(paths.recurrence).is_file():
        recurrence = np.rint(_load(paths.recurrence)[0]).astype(np.int16)

    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # The file name carries a hash of the tissue maps and the relaxation
        # settings, because different synthetic variants share a case id.
        cache_file = cache_dir / (
            f"{paths.case_id}_phi_v3_"
            f"{_phase_field_key(p_wm, p_gm, p_csf, phase_field_iters, phase_field_epsilon)}.npy"
        )

    if cache_file is not None and cache_file.is_file():
        phi = np.load(cache_file)
    else:
        phi = phase_field(
            p_wm,
            p_gm,
            p_csf,
            n_iter=phase_field_iters,
            epsilon=phase_field_epsilon,
            device=device,
        )
        if cache_file is not None:
            np.save(cache_file, phi)

    return Case(
        case_id=paths.case_id,
        p_wm=p_wm,
        p_gm=p_gm,
        p_csf=p_csf,
        segmentation=segmentation,
        brain_mask=brain_mask.astype(bool),
        phi=phi,
        x0=tumour_centroid(segmentation),
        affine=affine,
        recurrence=recurrence,
    )
