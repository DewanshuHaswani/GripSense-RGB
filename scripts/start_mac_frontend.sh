#!/usr/bin/env bash
set -euo pipefail

PORT=7676
INFERENCE_PORT=7867

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
    --inference-port)
      INFERENCE_PORT="${2:-}"
      if [[ -z "$INFERENCE_PORT" ]]; then
        echo "--inference-port requires a value" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--port 7676] [--inference-port 7867]" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export VITE_GRIPSENSE_INFERENCE_TARGET="http://127.0.0.1:$INFERENCE_PORT"
export VITE_GRIPSENSE_RFDETR_ENDPOINT="http://127.0.0.1:$INFERENCE_PORT/api/rfdetr/analyze"
export VITE_GRIPSENSE_REALSENSE_ENDPOINT="http://127.0.0.1:$INFERENCE_PORT/api/realsense/depth-signal"
export VITE_GRIPSENSE_V3_ENDPOINT="http://127.0.0.1:$INFERENCE_PORT/v3/analyze-frame"

echo "Starting GripSense frontend on http://127.0.0.1:$PORT"
echo "V10 proxy target: $VITE_GRIPSENSE_INFERENCE_TARGET"
npm run dev -- --host 127.0.0.1 --port "$PORT"
