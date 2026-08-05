# Headless smoke run of splat_segmentation.ipynb + verification.
# Usage: .\run_seg_smoke.ps1 [-SplatPly <path>] [-ClickFrame 0] [-ClickU 252] [-ClickV 200]
# Needs gsplat at runtime -> on Windows launch via run_seg_smoke.cmd (sets MSVC/CUDA env).
param(
    [string]$SplatPly = "_work/splats_voxel.ply",
    [int]$ClickFrame = 0,
    [int]$ClickU = 252,
    [int]$ClickV = 200
)
$ErrorActionPreference = "Stop"
$PY = "$env:LOCALAPPDATA\miniconda3\envs\splat\python.exe"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force _work | Out-Null

& $PY -m papermill splat_segmentation.ipynb _work\seg_out.ipynb `
    -p SPLAT_PLY $SplatPly -p CLICK_FRAME $ClickFrame -p CLICK_U $ClickU -p CLICK_V $ClickV `
    --cwd $PSScriptRoot
if ($LASTEXITCODE -ne 0) {
    Write-Host "papermill FAILED - inspect _work\seg_out.ipynb for the failing cell"
    exit 1
}
& $PY "$PSScriptRoot\verify_seg.py" "$PSScriptRoot\_work\segment" "$PSScriptRoot\_work\cameras.npz" "$PSScriptRoot\$SplatPly"
exit $LASTEXITCODE
