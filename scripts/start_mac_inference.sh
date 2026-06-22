#!/usr/bin/env bash
set -euo pipefail

PORT=7867

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      if [[ -z "$PORT" ]]; then
        echo "--port requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--port 7867]" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/local-inference/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python virtualenv not found. Run ./scripts/setup_mac.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT/local-inference"
export GRIPSENSE_RFDETR_DEVICE="${GRIPSENSE_RFDETR_DEVICE:-cpu}"

echo "Starting GripSense local inference server on http://127.0.0.1:$PORT"
"$VENV_PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "$PORT"
