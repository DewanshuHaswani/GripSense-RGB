"""Download/cache local inference models before running GripSense.

This script intentionally does not store model weights in the Git repository.
RF-DETR downloads or resolves its model assets through the installed Python
package/runtime cache. Running this once on a fresh machine avoids the first
browser request paying the full model-load cost.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from PIL import Image


def warmup_rfdetr() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    started = time.perf_counter()
    print("Loading RF-DETR-Seg Nano on CPU...")
    try:
        from rfdetr import RFDETRSegNano
    except Exception as exc:
        raise RuntimeError("Could not import rfdetr. Run `pip install -r local-inference/requirements.txt` first.") from exc

    model = RFDETRSegNano()
    image = Image.new("RGB", (320, 320), (245, 245, 245))
    print("Running one tiny warmup prediction...")
    model.predict(image, threshold=0.6)
    elapsed = time.perf_counter() - started
    print(f"RF-DETR warmup complete in {elapsed:.1f}s.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm up GripSense local inference models.")
    parser.add_argument("--model", choices=["rfdetr"], default="rfdetr")
    args = parser.parse_args()

    if args.model == "rfdetr":
        warmup_rfdetr()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Warmup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
