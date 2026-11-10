# Windows PowerShell version of fetch_reference_pinn.sh: clone the reference PINN at commit 99af20f.
$Commit = "99af20f"
$Target = Join-Path $PSScriptRoot "..\external\pinngbm-torch"
if (-not (Test-Path (Join-Path $Target ".git"))) {
    git clone https://github.com/Rayzhangzirui/pinngbm-torch.git $Target
}
git -C $Target checkout --quiet $Commit
Write-Output "reference PINN at $(git -C $Target rev-parse --short HEAD) in $Target"
