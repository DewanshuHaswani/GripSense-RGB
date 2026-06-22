#!/usr/bin/env bash
set -euo pipefail

WITH_REALSENSE=0
SKIP_WARMUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-realsense)
      WITH_REALSENSE=1
      shift
      ;;
    --skip-warmup)
      SKIP_WARMUP=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--with-realsense] [--skip-warmup]" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "GripSense RGB macOS setup"
echo "Repo: $REPO_ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js was not found. Install Node.js 20.19 or newer first." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found. Install Node.js 20.19 or newer first." >&2
  exit 1
fi

NODE_VERSION="$(node -p "process.versions.node")"
NODE_MAJOR="${NODE_VERSION%%.*}"
NODE_MINOR="$(node -p "process.versions.node.split('.')[1]")"
if (( NODE_MAJOR < 20 || (NODE_MAJOR == 20 && NODE_MINOR < 19) )); then
  echo "Node.js $NODE_VERSION is too old. Install Node.js 20.19 or newer for Vite 8." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3.10 or newer first." >&2
  exit 1
fi

echo "Installing frontend dependencies..."
npm install

VENV_DIR="$REPO_ROOT/local-inference/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "Installing local inference dependencies..."
"$VENV_PYTHON" -m pip install -r "$REPO_ROOT/local-inference/requirements.txt"

if [[ "$WITH_REALSENSE" -eq 1 ]]; then
  echo "Installing pyrealsense2 for Intel RealSense depth..."
  "$VENV_PYTHON" -m pip install pyrealsense2
fi

if [[ "$SKIP_WARMUP" -eq 0 ]]; then
  echo "Warming up RF-DETR-Seg Nano. This may download model weights and can take several minutes on first run."
  "$VENV_PYTHON" "$REPO_ROOT/local-inference/warmup_models.py" --model rfdetr
fi

echo
echo "Setup complete."
echo "Terminal 1: ./scripts/start_mac_inference.sh"
echo "Terminal 2: ./scripts/start_mac_frontend.sh"
echo "Open V10 proxy mode: http://127.0.0.1:7676/?version=v10"
echo "Open V8 direct mode: http://127.0.0.1:7676/?version=v8"
