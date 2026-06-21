param(
  [switch]$WithRealSense,
  [switch]$SkipWarmup
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "GripSense RGB Windows setup" -ForegroundColor Cyan
Write-Host "Repo: $repoRoot"

function Assert-Command {
  param([string]$Name, [string]$InstallHint)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "$Name was not found. $InstallHint"
  }
}

Assert-Command "node" "Install Node.js 20.19 or newer from https://nodejs.org/"
Assert-Command "npm" "Install Node.js 20.19 or newer from https://nodejs.org/"

$nodeVersionText = (& node -v).TrimStart("v")
$nodeParts = $nodeVersionText.Split(".")
$nodeMajor = [int]$nodeParts[0]
$nodeMinor = [int]$nodeParts[1]
if ($nodeMajor -lt 20 -or ($nodeMajor -eq 20 -and $nodeMinor -lt 19)) {
  throw "Node.js $nodeVersionText is too old. Install Node.js 20.19 or newer for Vite 8."
}

$pythonLauncher = Get-Command "py" -ErrorAction SilentlyContinue
if ($pythonLauncher) {
  $pythonCmd = "py"
  $pythonArgs = @("-3")
} elseif (Get-Command "python" -ErrorAction SilentlyContinue) {
  $pythonCmd = "python"
  $pythonArgs = @()
} else {
  throw "Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/windows/"
}

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
& npm install

$venvDir = Join-Path $repoRoot "local-inference\.venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
  & $pythonCmd @pythonArgs -m venv $venvDir
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing local inference dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $repoRoot "local-inference\requirements.txt")

if ($WithRealSense) {
  Write-Host "Installing pyrealsense2 for Intel RealSense D445 depth..." -ForegroundColor Cyan
  & $venvPython -m pip install pyrealsense2
}

if (-not $SkipWarmup) {
  Write-Host "Warming up RF-DETR-Seg Nano. This may download model weights and can take several minutes on first run." -ForegroundColor Cyan
  & $venvPython (Join-Path $repoRoot "local-inference\warmup_models.py") --model rfdetr
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Terminal 1: npm run dev"
Write-Host "Terminal 2: .\scripts\start_windows_inference.ps1"
Write-Host "Open: http://127.0.0.1:5173/?version=v8"
Write-Host "RealSense mode: http://127.0.0.1:5173/?version=v9"
