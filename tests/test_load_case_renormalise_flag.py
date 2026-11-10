"""Renormalization switch: load_case(renormalise_tumour=False) leaves the tissue maps as stored."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.predict_gbm import CasePaths, load_case  # noqa: E402


def write(path: Path, array: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), str(path))
    return path


def test_flag_skips_only_the_tumour_rescaling(tmp_path):
    shape = (12, 12, 12)
    wm = np.full(shape, 0.3, np.float32)
    gm = np.full(shape, 0.3, np.float32)
    csf = np.zeros(shape, np.float32)
    seg = np.zeros(shape, np.float32)
    seg[5:8, 5:8, 5:8] = 2
    paths = CasePaths(case_id="T", tumor_seg=write(tmp_path / "seg.nii.gz", seg),
                      wm=write(tmp_path / "wm.nii.gz", wm), gm=write(tmp_path / "gm.nii.gz", gm),
                      csf=write(tmp_path / "csf.nii.gz", csf), brain_mask=None, recurrence=None)
    twice = load_case(paths, cache_dir=None, phase_field_iters=2, device="cpu")
    once = load_case(paths, cache_dir=None, phase_field_iters=2, device="cpu",
                     renormalise_tumour=False)
    assert np.isclose(float(twice.p_wm[6, 6, 6]), 0.5)   # 0.3 / (0.3 + 0.3)
    assert np.isclose(float(once.p_wm[6, 6, 6]), 0.3)    # as stored
    assert np.isclose(float(twice.p_wm[0, 0, 0]), float(once.p_wm[0, 0, 0]))  # outside: equal
