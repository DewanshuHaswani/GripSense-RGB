from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


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


app = FastAPI(title="GripSense V3 Local Inference", version="0.1.0")
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


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
