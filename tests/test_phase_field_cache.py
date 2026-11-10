"""The phase-field cache key separates cases that share an id.

The synthetic variants without noise or on the fit grid reuse the 16 synthetic case
ids with different tissue maps. A key made of the case id alone would let a second
variant read the first one's phase field.
"""

import nibabel as nib
import numpy as np

import src.data.predict_gbm as pg


def write_case(root, case_id, wm_value):
    case = root / case_id
    case.mkdir(parents=True)
    shape = (12, 12, 12)
    wm = np.zeros(shape, np.float32)
    wm[2:10, 2:10, 2:10] = wm_value
    gm = np.zeros(shape, np.float32)
    gm[2:10, 2:10, 2:10] = 1.0 - wm_value
    csf = np.zeros(shape, np.float32)
    seg = np.zeros(shape, np.int16)
    seg[5:7, 5:7, 5:7] = 1
    for name, data in [("wm_pbmap", wm), ("gm_pbmap", gm), ("csf_pbmap", csf), ("tumor_seg", seg)]:
        nib.save(nib.Nifti1Image(data, np.eye(4)), case / f"{name}.nii.gz")
    return pg.CasePaths(case_id=case_id, tumor_seg=case / "tumor_seg.nii.gz",
                        wm=case / "wm_pbmap.nii.gz", gm=case / "gm_pbmap.nii.gz",
                        csf=case / "csf_pbmap.nii.gz")


def test_same_id_different_maps_do_not_share_a_cache(tmp_path, monkeypatch):
    calls = []
    real = pg.phase_field
    monkeypatch.setattr(pg, "phase_field", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = tmp_path / "cache"
    a = write_case(tmp_path / "variant_a", "SYN-X", 0.9)
    b = write_case(tmp_path / "variant_b", "SYN-X", 0.4)

    first = pg.load_case(a, cache_dir=cache, phase_field_iters=5, device="cpu")
    second = pg.load_case(b, cache_dir=cache, phase_field_iters=5, device="cpu")
    assert len(calls) == 2  # the second variant computed its own field
    assert len(list(cache.glob("SYN-X_phi_v3_*.npy"))) == 2

    again = pg.load_case(a, cache_dir=cache, phase_field_iters=5, device="cpu")
    assert len(calls) == 2  # same inputs: read from the cache
    assert np.array_equal(again.phi, first.phi)

    other_iters = pg.load_case(a, cache_dir=cache, phase_field_iters=7, device="cpu")
    assert len(calls) == 3  # a different setting is a different field
    assert second.case_id == other_iters.case_id == "SYN-X"
