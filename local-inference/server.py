import json
import os
import time
from io import BytesIO
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image


DiagnosticCode = Literal[
    "object_uncertain",
    "hand_occluded",
    "contact_uncertain",
    "slip_risk",
    "server_unavailable",
    "strong_hold",
]


class Landmark(BaseModel):
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None


class FramePayload(BaseModel):
    dataUrl: str
    width: int
    height: int
    mirrored: bool
    coordinateSpace: Literal["video"] = "video"


class ObjectRegion(BaseModel):
    center: dict[str, float]
    radiusX: float
    radiusY: float
    angle: float
    confidence: float
    locked: bool
    source: str
    velocity: dict[str, float]
    contour: list[dict[str, float]] = Field(default_factory=list)
    tightness: float | None = None
    independentEvidenceScore: float | None = None
    relativeDriftScore: float | None = None


class V2Diagnostics(BaseModel):
    state: str
    mode: str
    issueCategory: str


class V2Evidence(BaseModel):
    fingerCurlScore: float = 0
    fingerSegmentContactScore: float = 0
    contactRoles: dict[str, float] = Field(default_factory=dict)
    palmObjectContainmentScore: float = 0
    thumbSupportScore: float = 0
    motionStabilityScore: float = 1
    objectLockQuality: float = 0
    independentObjectScore: float = 0
    temporalLockScore: float = 0


class V2Analysis(BaseModel):
    gripPercentage: float
    confidence: float
    diagnostics: V2Diagnostics
    evidence: V2Evidence
    objectLockQuality: float
    slipRisk: float


class ObjectIdentity(BaseModel):
    hasProfiles: bool
    score: float
    matched: bool
    name: str | None = None


class AnalyzeFrameRequest(BaseModel):
    version: Literal["v3"]
    frame: FramePayload
    timestamp: float
    hand: list[Landmark] | None = None
    object: ObjectRegion | None = None
    v2Analysis: V2Analysis
    objectIdentity: ObjectIdentity | None = None


class HandResponse(BaseModel):
    meshQuality: float
    occlusion: float
    handednessConfidence: float
    fingerArticulation: float
    joints: list[Landmark] | None = None


class ObjectResponse(BaseModel):
    present: bool
    maskConfidence: float
    maskStability: float
    identityConfidence: float
    poseConfidence: float
    lockConfidence: float


class ContactResponse(BaseModel):
    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float
    palm: float
    coverage: float
    opposingPairs: float


class TemporalResponse(BaseModel):
    continuity: float
    coupling: float
    slipRisk: float
    jitter: float


class AnalyzeFrameResponse(BaseModel):
    version: Literal["v3"]
    frameTimestamp: float
    latencyMs: float
    uncertainty: float
    hand: HandResponse
    object: ObjectResponse
    contact: ContactResponse
    temporal: TemporalResponse
    diagnostics: list[DiagnosticCode]


class RfdetrPoint(BaseModel):
    x: float
    y: float


class RfdetrBBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class RfdetrDetectionResponse(BaseModel):
    id: str
    label: str
    score: float
    bbox: RfdetrBBox
    maskPolygon: list[RfdetrPoint]
    maskArea: float
    center: RfdetrPoint
    latencyMs: float


class RfdetrAnalyzeResponse(BaseModel):
    detections: list[RfdetrDetectionResponse]
    latencyMs: float
    model: str
    device: str


_RFDETR_MODEL: Any | None = None
_RFDETR_CLASSES: dict[int, str] | list[str] | None = None
_RFDETR_MODEL_NAME = "RF-DETR-Seg Nano"
_RFDETR_DEVICE = os.environ.get("GRIPSENSE_RFDETR_DEVICE", "cpu").lower()

if _RFDETR_DEVICE == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


app = FastAPI(title="GripSense Local Inference", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/v3/analyze-frame", response_model=AnalyzeFrameResponse)
async def analyze_frame(request: AnalyzeFrameRequest) -> AnalyzeFrameResponse:
    """Adapter scaffold for the V3 perception contract.

    Replace this conservative placeholder with SAM2/EfficientTAM, HaMeR/Hamba,
    object pose, and contact-map model adapters. Until then, the response stays
    bounded by the browser's V2 evidence so the client can validate fusion and
    fallback behavior without shipping large model weights in the frontend repo.
    """

    v2 = request.v2Analysis
    has_hand = bool(request.hand)
    has_object = bool(request.object and request.object.locked)
    contact_roles = v2.evidence.contactRoles
    identity_score = request.objectIdentity.score if request.objectIdentity and request.objectIdentity.matched else 0.0

    object_quality = clamp(max(v2.objectLockQuality, v2.evidence.objectLockQuality))
    hand_quality = clamp(0.5 + v2.evidence.fingerCurlScore * 0.24 + v2.evidence.thumbSupportScore * 0.18) if has_hand else 0.0
    contact_quality = clamp(v2.evidence.fingerSegmentContactScore * 0.58 + v2.evidence.thumbSupportScore * 0.24)
    temporal_quality = clamp(v2.evidence.motionStabilityScore * 0.7 + v2.evidence.temporalLockScore * 0.3)

    diagnostics: list[DiagnosticCode] = []
    if not has_object or object_quality < 0.4:
        diagnostics.append("object_uncertain")
    elif not has_hand or hand_quality < 0.42:
        diagnostics.append("hand_occluded")
    elif v2.slipRisk > 0.56:
        diagnostics.append("slip_risk")
    elif contact_quality < 0.42:
        diagnostics.append("contact_uncertain")
    elif v2.gripPercentage >= 70:
        diagnostics.append("strong_hold")

    return AnalyzeFrameResponse(
        version="v3",
        frameTimestamp=request.timestamp,
        latencyMs=1.0,
        uncertainty=clamp(1 - (object_quality * 0.32 + hand_quality * 0.24 + contact_quality * 0.24 + temporal_quality * 0.2)),
        hand=HandResponse(
            meshQuality=hand_quality,
            occlusion=clamp(1 - hand_quality),
            handednessConfidence=0.86 if has_hand else 0.0,
            fingerArticulation=clamp(v2.evidence.fingerCurlScore),
            joints=request.hand,
        ),
        object=ObjectResponse(
            present=has_object,
            maskConfidence=object_quality if has_object else 0.0,
            maskStability=clamp(v2.evidence.temporalLockScore),
            identityConfidence=identity_score,
            poseConfidence=clamp(v2.evidence.independentObjectScore),
            lockConfidence=object_quality,
        ),
        contact=ContactResponse(
            thumb=clamp(contact_roles.get("thumb", v2.evidence.thumbSupportScore)),
            index=clamp(contact_roles.get("index", 0.0)),
            middle=clamp(contact_roles.get("middle", 0.0)),
            ring=clamp(contact_roles.get("ring", 0.0)),
            pinky=clamp(contact_roles.get("pinky", 0.0)),
            palm=clamp(contact_roles.get("palm", v2.evidence.palmObjectContainmentScore)),
            coverage=clamp(v2.evidence.fingerSegmentContactScore),
            opposingPairs=clamp((contact_roles.get("thumb", 0.0) + max(contact_roles.get("index", 0.0), contact_roles.get("middle", 0.0))) / 2),
        ),
        temporal=TemporalResponse(
            continuity=clamp(v2.evidence.temporalLockScore),
            coupling=temporal_quality,
            slipRisk=clamp(v2.slipRisk),
            jitter=clamp(1 - temporal_quality),
        ),
        diagnostics=diagnostics,
    )


@app.post("/api/rfdetr/analyze", response_model=RfdetrAnalyzeResponse)
async def analyze_rfdetr_frame(
    frame: UploadFile = File(...),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    mirrored: bool = Form(default=False),
    handLandmarks: str | None = Form(default=None),
    threshold: float = Form(default=0.35),
) -> RfdetrAnalyzeResponse:
    """Run RF-DETR-Seg Nano on a JPEG frame.

    The browser sends optional hand landmarks so future model adapters can use
    them server-side. V8 grip scoring still treats labels as diagnostics only.
    """

    del width, height, mirrored
    if handLandmarks:
        try:
            json.loads(handLandmarks)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="handLandmarks must be JSON when provided")

    model, classes = get_rfdetr_model()
    start = time.perf_counter()
    image_bytes = await frame.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    try:
        raw_detections = model.predict(image, threshold=float(threshold))
    except Exception as exc:  # pragma: no cover - depends on local model install/runtime.
        raise HTTPException(status_code=500, detail=f"RF-DETR inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    detections = normalize_rfdetr_detections(raw_detections, classes, latency_ms)
    return RfdetrAnalyzeResponse(
        detections=detections,
        latencyMs=latency_ms,
        model=_RFDETR_MODEL_NAME,
        device=_RFDETR_DEVICE,
    )


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def get_rfdetr_model() -> tuple[Any, dict[int, str] | list[str] | None]:
    global _RFDETR_MODEL, _RFDETR_CLASSES
    if _RFDETR_MODEL is not None:
        return _RFDETR_MODEL, _RFDETR_CLASSES
    try:
        from rfdetr import RFDETRSegNano
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except Exception as exc:  # pragma: no cover - depends on optional local dependency.
        raise HTTPException(
            status_code=503,
            detail="RF-DETR is not installed. Run `pip install -r local-inference/requirements.txt`.",
        ) from exc
    try:
        _RFDETR_MODEL = RFDETRSegNano()
    except Exception as exc:  # pragma: no cover - model weight downloads are environment-dependent.
        raise HTTPException(status_code=503, detail=f"RF-DETR model could not load: {exc}") from exc
    _RFDETR_CLASSES = COCO_CLASSES
    return _RFDETR_MODEL, _RFDETR_CLASSES


def normalize_rfdetr_detections(raw_detections: Any, classes: dict[int, str] | list[str] | None, latency_ms: float) -> list[RfdetrDetectionResponse]:
    xyxy = np.asarray(getattr(raw_detections, "xyxy", []), dtype=float)
    confidence = np.asarray(getattr(raw_detections, "confidence", []), dtype=float)
    class_ids = np.asarray(getattr(raw_detections, "class_id", []), dtype=int)
    masks = getattr(raw_detections, "mask", None)
    responses: list[RfdetrDetectionResponse] = []

    for index, box in enumerate(xyxy):
        if box.shape[0] < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        score = clamp(float(confidence[index])) if index < len(confidence) else 0.0
        class_id = int(class_ids[index]) if index < len(class_ids) else -1
        label = class_label(classes, class_id)
        mask = masks[index] if masks is not None and index < len(masks) else None
        polygon = mask_polygon(mask, (x1, y1, x2, y2))
        mask_area = float(np.asarray(mask, dtype=bool).sum()) if mask is not None else width * height
        responses.append(
            RfdetrDetectionResponse(
                id=f"rfdetr-{index}-{class_id}",
                label=label,
                score=score,
                bbox=RfdetrBBox(x=x1, y=y1, width=width, height=height),
                maskPolygon=[RfdetrPoint(x=point[0], y=point[1]) for point in polygon],
                maskArea=mask_area,
                center=RfdetrPoint(x=x1 + width / 2, y=y1 + height / 2),
                latencyMs=latency_ms,
            )
        )
    return responses


def class_label(classes: dict[int, str] | list[str] | None, class_id: int) -> str:
    if isinstance(classes, dict):
        return classes.get(class_id, "object")
    if isinstance(classes, list) and 0 <= class_id < len(classes):
        return classes[class_id]
    return "object"


def mask_polygon(mask: Any, box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    if mask is None:
        return bbox_polygon(box)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2 or not mask_array.any():
        return bbox_polygon(box)
    ys, xs = np.where(mask_array)
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    return [
        (mid_x, min_y),
        (max_x, mid_y),
        (mid_x, max_y),
        (min_x, mid_y),
    ]


def bbox_polygon(box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
