# Headless smoke run of splat_init_and_train.ipynb + init verification.
# Usage: .\run_smoke.ps1 [-Video <path>] [-MaxFrames 8] [-TrainSteps 0]
param(
    [string]$Video = "../arch_orbit1.mp4",
    [int]$MaxFrames = 8,
    [int]$TrainSteps = 0
)
$ErrorActionPreference = "Stop"
$PY = "$env:LOCALAPPDATA\miniconda3\envs\splat\python.exe"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force _work | Out-Null

& $PY -m papermill splat_init_and_train.ipynb _work\smoke_out.ipynb `
    -p VIDEO_OR_DIR $Video -p MAX_FRAMES $MaxFrames -p TRAIN_STEPS $TrainSteps `
    --cwd $PSScriptRoot
if ($LASTEXITCODE -ne 0) {
    Write-Host "papermill FAILED - inspect _work\smoke_out.ipynb for the failing cell"
    exit 1
}
& $PY "$PSScriptRoot\verify_init.py" "$PSScriptRoot\_work"
exit $LASTEXITCODE
