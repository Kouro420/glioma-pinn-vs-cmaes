#!/usr/bin/env python
"""Download the run records and objective profiles of our paper (one archive on Zenodo) into outputs/.

With them, every figure and table of our paper can be rebuilt without a new run
(``python -m scripts.make_figures`` and the table scripts in docs/REPRODUCE.md).

The archive: https://doi.org/10.5281/zenodo.24107357 (version 1.0, CC BY 4.0).

    python scripts/fetch_records.py
    python scripts/fetch_records.py --archive records.zip   # an archive already downloaded
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Version 1.0 of the archive, https://doi.org/10.5281/zenodo.24107357
ZENODO_URL = "https://zenodo.org/records/24107357/files/glioma-pinn-vs-cmaes-records.zip?download=1"
MD5 = "65cba3c9ec4cb04e7f4cbeb00069f3b0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="use this local archive instead of downloading")
    args = ap.parse_args()
    archive = args.archive
    if archive is None:
        if not ZENODO_URL:
            print("The Zenodo link is added at release. Until then, download the archive by hand "
                  "and pass --archive.")
            return 1
        archive = REPO / "outputs" / "records.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ZENODO_URL}")
        urllib.request.urlretrieve(ZENODO_URL, archive)
    if MD5:
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != MD5:
            print(f"checksum mismatch: {digest}, expected {MD5}")
            return 1
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.startswith(("outputs/runs_", "outputs/profiles/"))]
        for name in names:
            z.extract(name, REPO)
    print(f"extracted {len(names)} run records and objective profiles into {REPO / 'outputs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
