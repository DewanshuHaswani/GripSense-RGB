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
$env:GRIPSENSE_YOLO_DEVICE = "cpu"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "4" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "4" }

Write-Host "Starting GripSense local inference server on http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "Health check: http://127.0.0.1:$Port/health" -ForegroundColor DarkCyan
& $venvPython -m uvicorn server:app --host 127.0.0.1 --port $Port --workers 1 --timeout-keep-alive 30 --limit-concurrency 16
