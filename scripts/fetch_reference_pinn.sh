#!/usr/bin/env bash
# Clone the reference PINN (Zhang et al. 2025, PyTorch release) at the commit used for every run of our paper.
# The release carries no license, so it is cloned, not copied into this repository. Nothing in it is
# edited; experiments/run_pinn.py applies two runtime patches (docs/REFERENCE_PINN.md).
set -euo pipefail
COMMIT=99af20f
TARGET="$(dirname "$0")/../external/pinngbm-torch"
if [ ! -d "$TARGET/.git" ]; then
  git clone https://github.com/Rayzhangzirui/pinngbm-torch.git "$TARGET"
fi
git -C "$TARGET" checkout --quiet "$COMMIT"
echo "reference PINN at $(git -C "$TARGET" rev-parse --short HEAD) in $TARGET"
