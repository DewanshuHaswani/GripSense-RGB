# GripSense V3 Local Inference Service

This folder contains the local server contract for V3 grip detection.

The first implementation is an adapter scaffold: it returns a conservative V3-shaped response from the browser's V2 state so the app can exercise timeout, stale-result, malformed-result, and fallback behavior. The intended production path is to replace the placeholder scoring in `server.py` with model adapters for:

- SAM2/EfficientTAM-style video object masks.
- HaMeR/Hamba-style 3D hand mesh.
- FoundationPose/reference-image object pose when object profiles exist.
- Contact probability aggregation per finger and palm region.

Run locally:

```bash
cd local-inference
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 7867
```

The browser calls `POST http://127.0.0.1:7867/v3/analyze-frame` by default. Override with `VITE_GRIPSENSE_V3_ENDPOINT` when starting Vite.
