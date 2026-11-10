#!/usr/bin/env python
"""Download the PREDICT-GBM cases this repository needs, without keeping the whole archive.

PREDICT-GBM (Zimmer et al. 2026) is distributed as one archive on Hugging Face,
``LZimmer/PREDICT-GBM/predict_gbm.tar.gz`` (19.5 GB, all cohorts). The cases sit in it in no particular
order, so a single case cannot be fetched on its own. This script reads the archive while it downloads,
unpacks only the files of the requested cases, and stops as soon as every requested case is complete.
Nothing but those files is written to disk.

    python scripts/fetch_predict_gbm.py                 # RHUH-0001 and RHUH-0002
    python scripts/fetch_predict_gbm.py --patients      # RHUH-0001 to RHUH-0020 (reads nearly the whole archive)
    python scripts/fetch_predict_gbm.py --archive predict_gbm.tar.gz   # use an archive you already downloaded

RHUH-0001 and RHUH-0002 are the two source brains of every synthetic case, so they are enough for the
one-case reproduction (``repro/one_case/``) and for all synthetic experiments. The 20 patients are needed
only to rerun the patient comparison.

The cases land in ``src/data/predict_gbm/predict_gbm/<case>/``, where the configs expect them; pass
``src/data/predict_gbm/predict_gbm`` as ``--data-root``. Only the eight files the code reads are kept
per case. The data are the benchmark's: read its terms at https://huggingface.co/datasets/LZimmer/PREDICT-GBM.
Needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/LZimmer/PREDICT-GBM/resolve/main/predict_gbm.tar.gz"
ARCHIVE_BYTES = 19_548_290_251  # size of predict_gbm.tar.gz on Hugging Face, used for the progress display
NEEDED = (
    "tumor_seg.nii.gz",           # pre-operative tumor labels (both masks come from it)
    "recurrence_preop.nii.gz",    # follow-up recurrence, used by the patient evaluation
    "wm_pbmap.nii.gz", "gm_pbmap.nii.gz", "csf_pbmap.nii.gz",   # tissue maps
    "t1c_bet_mask.nii.gz",        # brain mask
    "t1c_bet_normalized.nii.gz", "flair_bet_normalized.nii.gz",  # images, only for figures
)
SOURCE_BRAINS = ["RHUH-0001", "RHUH-0002"]
PATIENTS = [f"RHUH-{i:04d}" for i in range(1, 21)]


class Progress:
    """File-like wrapper that counts the compressed bytes read and prints the progress now and then."""

    def __init__(self, raw, total: int) -> None:
        self.raw, self.total, self.read_bytes = raw, total, 0
        self.started = self.last = time.monotonic()

    def read(self, n: int = -1) -> bytes:
        chunk = self.raw.read(n)
        self.read_bytes += len(chunk)
        now = time.monotonic()
        if now - self.last > 10:
            self.last = now
            rate = self.read_bytes / max(now - self.started, 1e-9) / 1e6
            print(f"  read {self.read_bytes / 1e9:5.2f} of {self.total / 1e9:.1f} GB "
                  f"({100 * self.read_bytes / self.total:4.1f} %), {rate:.0f} MB/s", flush=True)
        return chunk


def open_source(args) -> tuple[object, int]:
    if args.archive:
        path = Path(args.archive)
        return path.open("rb"), path.stat().st_size
    request = urllib.request.Request(args.url, headers={"User-Agent": "glioma-pinn-vs-cmaes/fetch_predict_gbm"})
    response = urllib.request.urlopen(request, timeout=60)
    size = int(response.headers.get("Content-Length") or ARCHIVE_BYTES)
    return response, size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--patients", action="store_true", help="the 20 RHUH patients (RHUH-0001 to RHUH-0020)")
    group.add_argument("--cases", nargs="+", metavar="ID", help="any case IDs of the archive")
    ap.add_argument("--out", type=Path, default=REPO / "src" / "data" / "predict_gbm",
                    help="target folder; the cases land in <out>/predict_gbm/<case>/")
    ap.add_argument("--archive", help="read a local copy of predict_gbm.tar.gz instead of downloading")
    ap.add_argument("--url", default=URL, help="archive URL (default: the Hugging Face file)")
    args = ap.parse_args()

    wanted = set(PATIENTS if args.patients else (args.cases or SOURCE_BRAINS))
    root = args.out / "predict_gbm"
    todo = {c for c in wanted if not all((root / c / f).is_file() for f in NEEDED)}
    if not todo:
        print(f"All {len(wanted)} cases are already in {root}.")
        return 0
    print(f"Fetching {len(todo)} case(s) into {root}: {', '.join(sorted(todo))}")
    print("The archive is read as a stream and the download stops when every case is complete.", flush=True)

    raw, size = open_source(args)
    stream = Progress(raw, size)
    done: set[str] = set()
    current = None
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            case = parts[1] if len(parts) > 2 else None
            # The files of one case are stored together, so a case is complete once the archive moves on.
            if current in todo and case != current:
                done.add(current)
                print(f"  {current} complete", flush=True)
                if done == todo:
                    break
            current = case
            if case in todo and member.isfile() and parts[-1] in NEEDED:
                target = root / case / parts[-1]
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                with target.open("wb") as handle:
                    while block := source.read(1 << 20):
                        handle.write(block)
        else:
            if current in todo:
                done.add(current)
    raw.close()

    missing = {c: [f for f in NEEDED if not (root / c / f).is_file()] for c in wanted}
    missing = {c: m for c, m in missing.items() if m}
    print(f"Read {stream.read_bytes / 1e9:.2f} GB of the archive.")
    if missing:
        for case, files in sorted(missing.items()):
            print(f"  MISSING {case}: {', '.join(files)}")
        return 1
    print(f"Done: {len(wanted)} case(s) with all {len(NEEDED)} files in {root}")
    print(f"Use --data-root {root.relative_to(REPO) if root.is_relative_to(REPO) else root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
