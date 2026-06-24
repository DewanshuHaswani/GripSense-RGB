param(
  [int]$Port = 7676,
  [int]$InferencePort = 7867
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:VITE_GRIPSENSE_INFERENCE_TARGET = "http://127.0.0.1:$InferencePort"
$env:VITE_GRIPSENSE_RFDETR_ENDPOINT = "http://127.0.0.1:$InferencePort/api/rfdetr/analyze"
$env:VITE_GRIPSENSE_YOLO_ENDPOINT = "/api/gripsense/yolo/analyze"
$env:VITE_GRIPSENSE_REALSENSE_ENDPOINT = "http://127.0.0.1:$InferencePort/api/realsense/depth-signal"
$env:VITE_GRIPSENSE_V3_ENDPOINT = "http://127.0.0.1:$InferencePort/v3/analyze-frame"

Write-Host "Starting GripSense frontend on http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "Local inference proxy target: $env:VITE_GRIPSENSE_INFERENCE_TARGET" -ForegroundColor Cyan
& npm run dev -- --host 127.0.0.1 --port $Port
