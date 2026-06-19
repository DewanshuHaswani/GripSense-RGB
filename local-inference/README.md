# GripSense Local Inference Service

This folder contains local CPU inference endpoints for GripSense.

Endpoints:

- `POST /v3/analyze-frame`: V3 adapter scaffold. It returns a conservative V3-shaped response from the browser's V2 state so the app can exercise timeout, stale-result, malformed-result, and fallback behavior.
- `POST /api/rfdetr/analyze`: V8/V9 live and Offline V2/V3 RF-DETR-Seg Nano object evidence. Input is a multipart JPEG frame plus optional `handLandmarks`; output includes `id`, `label`, `score`, `bbox`, `maskPolygon`, `maskArea`, `center`, and `latencyMs` for each detection.
- `POST /api/realsense/depth-signal`: V9 live and Offline V3 RealSense depth evidence. Input is current hand landmarks plus the selected object geometry; output includes depth contact, hand/object separation, stereo confidence, occlusion, surface continuity, and latency. If `pyrealsense2` or the camera is unavailable, the response reports `available: false` and the frontend does not fake depth confidence.

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
# Optional, only for V9 / Offline V3 RealSense depth:
pip install pyrealsense2
uvicorn server:app --host 127.0.0.1 --port 7867
```

On a fresh machine, keep this server running in one terminal and run the frontend from the repo root in another terminal:

```bash
npm install
npm run dev
```

Health check:

```bash
curl -I http://127.0.0.1:7867/docs
```

The browser calls:

- `POST http://127.0.0.1:7867/v3/analyze-frame` for V3. Override with `VITE_GRIPSENSE_V3_ENDPOINT`.
- `POST http://127.0.0.1:7867/api/rfdetr/analyze` for V8/V9 and Offline V2/V3. Override with `VITE_GRIPSENSE_RFDETR_ENDPOINT`.
- `POST http://127.0.0.1:7867/api/realsense/depth-signal` for V9 and Offline V3. Override with `VITE_GRIPSENSE_REALSENSE_ENDPOINT`.

RF-DETR runs on CPU by default. You can set `GRIPSENSE_RFDETR_DEVICE=cpu` explicitly before launching the server. V8 reports `RF-DETR unavailable` rather than inventing confidence if this server is not reachable.

RealSense depth is optional and hardware-dependent. Use a RealSense RGB-D camera as the browser camera source, keep it connected to the same machine running this server, and install `pyrealsense2` separately. V9/Offline V3 use depth only as contact evidence; RF-DETR masks still provide the object evidence.

Troubleshooting:

- `server unavailable` in V8 means the frontend could not get a valid RF-DETR response from this server.
- If `curl -I http://127.0.0.1:7867/docs` fails, the server is not reachable.
- If `/docs` works but V8 is unavailable, inspect the server terminal for missing `rfdetr`, model load/download failure, or CPU inference errors.
- If V9/Offline V3 show RealSense unavailable, install `pyrealsense2`, connect the RealSense camera, and restart this server.
