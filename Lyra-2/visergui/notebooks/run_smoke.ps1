# Headless smoke run of splat_init_and_train.ipynb + verification.
# Usage: .\run_smoke.ps1 [-Video <path>] [-MaxFrames 8] [-TrainSteps 0]
#                        [-Mode 3dgs|2dgs] [-ShDeg 2] [-ShRamp 1000]
#                        [-LpipsWeight 0.05] [-Densify 1]
# Training runs on Windows need the MSVC/CUDA env — use run_train_smoke.cmd for those.
param(
    [string]$Video = "../arch_orbit1.mp4",
    [int]$MaxFrames = 8,
    [int]$TrainSteps = 0,
    [string]$Mode = "3dgs",
    [int]$ShDeg = 2,
    [int]$ShRamp = 1000,
    [double]$LpipsWeight = 0.05,
    [int]$Densify = 1
)
$ErrorActionPreference = "Stop"
$PY = "$env:LOCALAPPDATA\miniconda3\envs\splat\python.exe"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force _work | Out-Null

$dens = if ($Densify -ne 0) { "True" } else { "False" }
& $PY -m papermill splat_init_and_train.ipynb _work\smoke_out.ipynb `
    -p VIDEO_OR_DIR $Video -p MAX_FRAMES $MaxFrames -p TRAIN_STEPS $TrainSteps `
    -p MODE $Mode -p SH_MAX_DEG $ShDeg -p SH_RAMP $ShRamp `
    -p LPIPS_WEIGHT $LpipsWeight -y "USE_DENSIFY: $dens" `
    --cwd $PSScriptRoot
if ($LASTEXITCODE -ne 0) {
    Write-Host "papermill FAILED - inspect _work\smoke_out.ipynb for the failing cell"
    exit 1
}
& $PY "$PSScriptRoot\verify_init.py" "$PSScriptRoot\_work"
if ($LASTEXITCODE -ne 0) { exit 1 }
if ($TrainSteps -gt 0) {
    & $PY "$PSScriptRoot\verify_train.py" "$PSScriptRoot\_work" $ShDeg
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
exit 0
