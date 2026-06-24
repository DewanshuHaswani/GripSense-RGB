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

prepend_homebrew_path() {
  if command -v brew >/dev/null 2>&1; then
    eval "$(brew shellenv)"
    return
  fi
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    return
  fi
  if [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

ensure_homebrew() {
  prepend_homebrew_path
  if command -v brew >/dev/null 2>&1; then
    return
  fi
  echo "Homebrew was not found. Install Homebrew first, then rerun this setup:" >&2
  echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"' >&2
  exit 1
}

node_is_new_enough() {
  command -v node >/dev/null 2>&1 || return 1
  node -e "const [major, minor] = process.versions.node.split('.').map(Number); process.exit(major > 20 || (major === 20 && minor >= 19) ? 0 : 1)"
}

python_is_new_enough() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

ensure_homebrew

if ! node_is_new_enough; then
  echo "Installing or upgrading Node.js with Homebrew..."
  brew install node
  brew upgrade node || true
  prepend_homebrew_path
fi

if ! command -v npm >/dev/null 2>&1 || ! node_is_new_enough; then
  echo "Node.js 20.19 or newer is still not available on PATH after Homebrew install." >&2
  echo "Close and reopen Terminal, then rerun ./scripts/setup_mac.sh." >&2
  exit 1
fi

echo "Using Node.js $(node -v) and npm $(npm -v)"

if ! python_is_new_enough; then
  echo "Installing Python 3 with Homebrew..."
  brew install python
  brew upgrade python || true
  prepend_homebrew_path
fi

if ! python_is_new_enough; then
  echo "Python 3.10 or newer is still not available on PATH after Homebrew install." >&2
  echo "Close and reopen Terminal, then rerun ./scripts/setup_mac.sh." >&2
  exit 1
fi

echo "Using $(python3 --version)"

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
  echo "Warming up RF-DETR-Seg Nano and YOLO. This may download model weights and can take several minutes on first run."
  "$VENV_PYTHON" "$REPO_ROOT/local-inference/warmup_models.py" --model all
fi

echo
echo "Setup complete."
echo "Terminal 1: ./scripts/start_mac_inference.sh"
echo "Terminal 2: ./scripts/start_mac_frontend.sh"
echo "Open V10 proxy mode: http://127.0.0.1:7676/?version=v10"
echo "Open V8 direct mode: http://127.0.0.1:7676/?version=v8"
echo "Open V11 YOLO mode: http://127.0.0.1:7676/?version=v11"
