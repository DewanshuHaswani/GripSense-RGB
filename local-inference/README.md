# GripSense Local Inference Service

This folder contains local CPU inference endpoints for GripSense.

Endpoints:

- `POST /v3/analyze-frame`: V3 adapter scaffold. It returns a conservative V3-shaped response from the browser's V2 state so the app can exercise timeout, stale-result, malformed-result, and fallback behavior.
- `POST /api/rfdetr/analyze`: V8 live and Offline V2 RF-DETR-Seg Nano object evidence. Input is a multipart JPEG frame plus optional `handLandmarks`; output includes `id`, `label`, `score`, `bbox`, `maskPolygon`, `maskArea`, `center`, and `latencyMs` for each detection.

The intended V3 production path is to replace the placeholder scoring in `server.py` with model adapters for:

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

The browser calls:

- `POST http://127.0.0.1:7867/v3/analyze-frame` for V3. Override with `VITE_GRIPSENSE_V3_ENDPOINT`.
- `POST http://127.0.0.1:7867/api/rfdetr/analyze` for V8 and Offline V2. Override with `VITE_GRIPSENSE_RFDETR_ENDPOINT`.

RF-DETR runs on CPU by default. You can set `GRIPSENSE_RFDETR_DEVICE=cpu` explicitly before launching the server. V8 reports `RF-DETR unavailable` rather than inventing confidence if this server is not reachable.
