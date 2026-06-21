param(
  [int]$Port = 7867
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot "local-inference\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  throw "Python virtualenv not found. Run .\scripts\setup_windows.ps1 first."
}

Set-Location (Join-Path $repoRoot "local-inference")
$env:GRIPSENSE_RFDETR_DEVICE = "cpu"

Write-Host "Starting GripSense local inference server on http://127.0.0.1:$Port" -ForegroundColor Cyan
& $venvPython -m uvicorn server:app --host 127.0.0.1 --port $Port
