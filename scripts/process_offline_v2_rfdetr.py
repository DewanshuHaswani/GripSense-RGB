#!/usr/bin/env python3
"""Batch-process a video with the local RF-DETR Offline V2-style pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


DEFAULT_ENDPOINT = "http://127.0.0.1:7867/api/rfdetr/analyze"
DEFAULT_YOLO_ENDPOINT = "http://127.0.0.1:7867/api/yolo/analyze"
PROVIDER_LABELS = {
    "rfdetr": "Offline V2 RF-DETR",
    "yolo-max": "Offline YOLO Max",
}
PERSON_LABELS = {"person"}
BACKGROUND_LABELS = {
    "bed",
    "couch",
    "chair",
    "dining table",
    "tv",
    "laptop",
    "keyboard",
    "backpack",
    "suitcase",
}
OBJECT_PRIOR = {
    "bottle": 0.22,
    "cup": 0.22,
    "cell phone": 0.18,
    "remote": 0.16,
    "book": 0.12,
}
HANDHELD_AREA_MIN = 0.004
HANDHELD_AREA_MAX = 0.22
CONTACT_FLOOR = 0.32


@dataclass
class TrackPoint:
    time: float
    source: str
    label: str
    score: float
    object_score: float
    contact: float
    grip: float
    weak: bool
    slip: float
    x: float | None
    y: float | None
    width: float | None
    height: float | None
    latency_ms: float | None


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def copy_point(point: TrackPoint) -> TrackPoint:
    return TrackPoint(**asdict(point))


def has_geometry(point: TrackPoint) -> bool:
    return point.x is not None and point.y is not None and bool(point.width) and bool(point.height)


def point_center(point: TrackPoint) -> tuple[float, float]:
    return (float(point.x or 0.0) + float(point.width or 0.0) / 2.0, float(point.y or 0.0) + float(point.height or 0.0) / 2.0)


def detection_center(detection: dict[str, Any]) -> tuple[float, float]:
    center = detection.get("center") or {}
    return float(center.get("x", 0.0)), float(center.get("y", 0.0))


def detection_bbox(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = detection.get("bbox") or {}
    return (
        float(bbox.get("x", 0.0)),
        float(bbox.get("y", 0.0)),
        float(bbox.get("width", 0.0)),
        float(bbox.get("height", 0.0)),
    )


def spatial_continuity(detection: dict[str, Any], previous: dict[str, Any] | None, width: int, height: int) -> float:
    if not previous:
        return 0.0
    ax, ay = detection_center(detection)
    bx, by = detection_center(previous)
    distance = math.hypot(ax - bx, ay - by)
    return clamp(1.0 - distance / (0.24 * math.hypot(width, height)))


def detection_area_ratio(detection: dict[str, Any], width: int, height: int) -> float:
    _, _, w, h = detection_bbox(detection)
    return clamp((w * h) / max(1.0, width * height), 0.0, 4.0)


def hand_contact_evidence(frame: Any, detection: dict[str, Any]) -> float:
    """Estimate hand/object contact from skin-colored pixels touching the object boundary.

    The local YOLO model often keeps detecting a dropped can on the bed. That is useful
    object evidence, but it should not become grip evidence unless a hand-like skin region
    is actually adjacent to the object. This intentionally stays heuristic so the batch
    script works on office laptops without a MediaPipe Python install.
    """
    height, width = frame.shape[:2]
    x, y, w, h = detection_bbox(detection)
    if w <= 2 or h <= 2:
        return 0.0

    pad_x = max(14, int(w * 0.24))
    pad_y = max(14, int(h * 0.24))
    x0 = max(0, int(x) - pad_x)
    y0 = max(0, int(y) - pad_y)
    x1 = min(width, int(x + w) + pad_x)
    y1 = min(height, int(y + h) + pad_y)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    hsv_skin = cv2.inRange(hsv, (0, 22, 38), (28, 190, 255))
    ycrcb_skin = cv2.inRange(ycrcb, (35, 132, 78), (255, 182, 135))
    skin = cv2.bitwise_and(hsv_skin, ycrcb_skin)
    skin = cv2.medianBlur(skin, 5)

    crop_h, crop_w = skin.shape[:2]
    object_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    polygon = detection.get("maskPolygon") or []
    if len(polygon) >= 3:
        pts = np.array(
            [
                [
                    int(round(float(point.get("x", 0.0)) - x0)),
                    int(round(float(point.get("y", 0.0)) - y0)),
                ]
                for point in polygon
            ],
            dtype=np.int32,
        )
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, crop_w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, crop_h - 1))
        cv2.fillPoly(object_mask, [pts], 255)
    if int((object_mask > 0).sum()) < 24:
        local_x0 = max(0, int(x) - x0)
        local_y0 = max(0, int(y) - y0)
        local_x1 = min(crop_w, int(x + w) - x0)
        local_y1 = min(crop_h, int(y + h) - y0)
        object_mask[local_y0:local_y1, local_x0:local_x1] = 255

    kernel_size = max(7, int(min(w, h) * 0.1) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    near_kernel_size = max(kernel_size + 6, int(min(w, h) * 0.18) | 1)
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_kernel_size, near_kernel_size))
    boundary = cv2.subtract(cv2.dilate(object_mask, kernel), object_mask)
    ring = cv2.subtract(cv2.dilate(object_mask, near_kernel), object_mask)

    object_area = max(1, int((object_mask > 0).sum()))
    boundary_area = max(1, int((boundary > 0).sum()))
    ring_area = max(1, int((ring > 0).sum()))
    boundary_skin = float(((skin > 0) & (boundary > 0)).sum()) / boundary_area
    ring_skin = float(((skin > 0) & (ring > 0)).sum()) / ring_area
    object_skin = float(((skin > 0) & (object_mask > 0)).sum()) / object_area

    moments = cv2.moments(object_mask)
    if moments["m00"]:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = crop_w // 2
        cy = crop_h // 2

    def side_ratio(mask: Any) -> float:
        area = max(1, int((mask > 0).sum()))
        return float(((skin > 0) & (mask > 0)).sum()) / area

    left_mask = boundary.copy()
    left_mask[:, cx:] = 0
    right_mask = boundary.copy()
    right_mask[:, :cx] = 0
    top_mask = boundary.copy()
    top_mask[cy:, :] = 0
    bottom_mask = boundary.copy()
    bottom_mask[:cy, :] = 0
    side_values = [side_ratio(mask) for mask in (left_mask, right_mask, top_mask, bottom_mask)]
    side_hits = sum(1 for value in side_values if value > 0.04)

    adjacent = clamp((boundary_skin - 0.025) / 0.16)
    nearby = clamp((ring_skin - 0.015) / 0.12)
    occluding = clamp((object_skin - 0.045) / 0.16)
    contact = clamp(adjacent * 0.5 + nearby * 0.14 + occluding * 0.36)
    # Object labels, can graphics, and wood/skin-like surfaces can satisfy the
    # broad skin heuristic inside the object mask. Require real boundary skin
    # before allowing that interior occlusion signal to become grip evidence.
    if boundary_skin < 0.075:
        contact = min(contact, 0.06)
    elif boundary_skin < 0.11 and ring_skin < 0.13:
        contact = min(contact, 0.3)
    if side_hits <= 1:
        contact = min(contact, 0.08)
    elif side_hits == 2 and occluding < 0.2 and boundary_skin < 0.1:
        contact = min(contact, 0.2)
    elif side_hits == 2 and occluding < 0.32:
        contact = min(contact, 0.5)
    if boundary_skin < 0.018 and object_skin < 0.05:
        contact = min(contact, 0.06)
    return contact


def score_detection(detection: dict[str, Any], previous: dict[str, Any] | None, width: int, height: int) -> float:
    label = str(detection.get("label", "")).lower()
    if label in PERSON_LABELS:
        return -1.0
    x, y, w, h = detection_bbox(detection)
    if w <= 2 or h <= 2:
        return -1.0
    area_ratio = detection_area_ratio(detection, width, height)
    if area_ratio < HANDHELD_AREA_MIN or area_ratio > 0.55:
        return -0.25
    if label in BACKGROUND_LABELS and area_ratio > 0.28:
        return -0.5
    score = float(detection.get("score", 0.0))
    continuity = spatial_continuity(detection, previous, width, height)
    prior = OBJECT_PRIOR.get(label, 0.0)
    size_penalty = max(0.0, area_ratio - HANDHELD_AREA_MAX) * 0.75
    aspect = max(w, h) / max(1.0, min(w, h))
    shape_penalty = max(0.0, aspect - 4.5) * 0.05
    edge_margin = min(x, y, width - (x + w), height - (y + h))
    edge_penalty = 0.08 if edge_margin < -8 else 0.0
    return score * 0.58 + continuity * 0.32 + prior - size_penalty - shape_penalty - edge_penalty


def choose_object(detections: list[dict[str, Any]], previous: dict[str, Any] | None, width: int, height: int) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any]]] = []
    for detection in detections:
        if previous:
            continuity = spatial_continuity(detection, previous, width, height)
            label = str(detection.get("label", "")).lower()
            raw_score = float(detection.get("score", 0.0))
            if continuity < 0.08 and raw_score < 0.72 and label not in OBJECT_PRIOR:
                continue
        scored.append((score_detection(detection, previous, width, height), detection))
    scored = [item for item in scored if item[0] > 0.12]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def clamp_geometry(point: TrackPoint, width: int, height: int) -> TrackPoint:
    if not has_geometry(point):
        return point
    next_point = copy_point(point)
    next_point.width = clamp(float(next_point.width or 0.0), 10.0, width * 0.72)
    next_point.height = clamp(float(next_point.height or 0.0), 10.0, height * 0.72)
    next_point.x = clamp(float(next_point.x or 0.0), 0.0, max(0.0, width - float(next_point.width or 0.0)))
    next_point.y = clamp(float(next_point.y or 0.0), 0.0, max(0.0, height - float(next_point.height or 0.0)))
    return next_point


def hold_from_previous(previous: TrackPoint, current: TrackPoint, age: float) -> TrackPoint:
    held = copy_point(current)
    decay = clamp(1.0 - age / 1.8)
    held.source = "held"
    held.label = previous.label
    held.score = previous.score * 0.48 * decay
    held.object_score = previous.object_score * 0.7 * decay
    held.contact = previous.contact * 0.58 * decay
    held.grip = clamp(held.object_score * 0.58 + held.contact * 0.42)
    held.slip = clamp(max(0.0, 0.42 - held.contact) + 0.12)
    held.x, held.y, held.width, held.height = previous.x, previous.y, previous.width, previous.height
    held.weak = held.grip < 0.45 or held.object_score < 0.28
    return held


def smooth_geometry(points: list[TrackPoint], metadata: dict[str, Any]) -> list[TrackPoint]:
    if not points:
        return points
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    diag = max(1.0, math.hypot(width, height))
    forward: list[TrackPoint] = []
    last: TrackPoint | None = None
    last_velocity = (0.0, 0.0)
    for point in points:
        current = clamp_geometry(point, width, height)
        if not has_geometry(current):
            if last and current.time - last.time <= 1.8:
                current = hold_from_previous(last, current, current.time - last.time)
            forward.append(current)
            continue
        if last and has_geometry(last):
            dt = max(0.001, current.time - last.time)
            lx, ly = point_center(last)
            cx, cy = point_center(current)
            predicted = (lx + last_velocity[0] * dt, ly + last_velocity[1] * dt)
            raw_jump = math.hypot(cx - lx, cy - ly) / diag
            predicted_jump = math.hypot(cx - predicted[0], cy - predicted[1]) / diag
            weak_outlier = raw_jump > 0.2 and predicted_jump > 0.16 and current.object_score < max(0.55, last.object_score * 0.9)
            huge_outlier = raw_jump > 0.34 and predicted_jump > 0.24
            if weak_outlier or huge_outlier:
                current = hold_from_previous(last, current, dt)
                forward.append(current)
                last = current
                continue
            alpha = 0.34 if current.object_score >= 0.62 else 0.22
            if raw_jump > 0.15:
                alpha *= 0.58
            sx = lerp(lx, cx, alpha)
            sy = lerp(ly, cy, alpha)
            max_step = diag * (0.105 + alpha * 0.12)
            step = math.hypot(sx - lx, sy - ly)
            if step > max_step:
                ratio = max_step / step
                sx = lx + (sx - lx) * ratio
                sy = ly + (sy - ly) * ratio
            sw = lerp(float(last.width or current.width or 0.0), float(current.width or last.width or 0.0), 0.2)
            sh = lerp(float(last.height or current.height or 0.0), float(current.height or last.height or 0.0), 0.2)
            current.width = sw
            current.height = sh
            current.x = sx - sw / 2.0
            current.y = sy - sh / 2.0
            last_velocity = ((sx - lx) / dt, (sy - ly) / dt)
        if has_geometry(current):
            current = clamp_geometry(current, width, height)
            last = current
        forward.append(current)

    backward = [copy_point(point) for point in forward]
    next_point: TrackPoint | None = None
    for index in range(len(backward) - 1, -1, -1):
        current = backward[index]
        if next_point and has_geometry(current) and has_geometry(next_point):
            cx, cy = point_center(current)
            nx, ny = point_center(next_point)
            future_gap = max(0.001, next_point.time - current.time)
            future_jump = math.hypot(cx - nx, cy - ny) / diag
            blend = 0.24 if future_jump < 0.18 and future_gap <= 0.75 else 0.08
            sw = lerp(float(current.width or 0.0), float(next_point.width or current.width or 0.0), blend)
            sh = lerp(float(current.height or 0.0), float(next_point.height or current.height or 0.0), blend)
            sx = lerp(cx, nx, blend)
            sy = lerp(cy, ny, blend)
            current.width = sw
            current.height = sh
            current.x = sx - sw / 2.0
            current.y = sy - sh / 2.0
            current = clamp_geometry(current, width, height)
            backward[index] = current
        if has_geometry(current):
            next_point = current
    return backward


def analyze_frame(endpoint: str, frame: Any, threshold: float, max_width: int) -> tuple[list[dict[str, Any]], float | None]:
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / max(1, width))
    resized = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return [], None
    response = requests.post(
        endpoint,
        files={"frame": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
        data={"threshold": str(threshold)},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    detections = payload.get("detections") or []
    if scale != 1:
        factor = 1.0 / scale
        for detection in detections:
            bbox = detection.get("bbox") or {}
            for key in ("x", "y", "width", "height"):
                bbox[key] = float(bbox.get(key, 0.0)) * factor
            center = detection.get("center") or {}
            center["x"] = float(center.get("x", 0.0)) * factor
            center["y"] = float(center.get("y", 0.0)) * factor
            for point in detection.get("maskPolygon") or []:
                point["x"] = float(point.get("x", 0.0)) * factor
                point["y"] = float(point.get("y", 0.0)) * factor
            detection["maskArea"] = float(detection.get("maskArea", 0.0)) * factor * factor
    return detections, payload.get("latencyMs")


def smooth_points(points: list[TrackPoint]) -> list[TrackPoint]:
    if not points:
        return points
    smoothed: list[TrackPoint] = []
    for index, point in enumerate(points):
        neighbors = []
        point_has_contact = point.contact >= CONTACT_FLOOR
        for neighbor in points[max(0, index - 2) : min(len(points), index + 3)]:
            neighbor_has_contact = neighbor.contact >= CONTACT_FLOOR
            if neighbor_has_contact != point_has_contact:
                continue
            if has_geometry(point) and has_geometry(neighbor):
                px, py = point_center(point)
                nx, ny = point_center(neighbor)
                max_span = max(float(point.width or 0.0), float(point.height or 0.0), 1.0) * 1.7
                if math.hypot(px - nx, py - ny) > max_span:
                    continue
            neighbors.append(neighbor)
        if not neighbors:
            neighbors = [point]
        object_scores = [neighbor.object_score for neighbor in neighbors]
        contacts = [neighbor.contact for neighbor in neighbors]
        grip = [neighbor.grip for neighbor in neighbors]
        next_point = TrackPoint(**asdict(point))
        next_point.object_score = sum(object_scores) / len(object_scores)
        next_point.contact = sum(contacts) / len(contacts)
        next_point.grip = sum(grip) / len(grip)
        if next_point.contact < CONTACT_FLOOR:
            next_point.grip = min(next_point.grip, next_point.contact * 0.22)
        next_point.weak = next_point.grip < 0.45 or next_point.object_score < 0.28 or next_point.contact < CONTACT_FLOOR
        smoothed.append(next_point)
    return smoothed


def suppress_isolated_contact_spikes(points: list[TrackPoint]) -> list[TrackPoint]:
    if len(points) < 3:
        return points
    debounced = [copy_point(point) for point in points]
    for index in range(1, len(points) - 1):
        point = points[index]
        previous = points[index - 1]
        following = points[index + 1]
        if point.contact < 0.45 or point.grip < 0.45:
            continue
        previous_low = previous.contact < 0.25 or previous.grip < 0.28
        following_low = following.contact < 0.25 or following.grip < 0.28
        if not (previous_low and following_low):
            continue
        if has_geometry(previous) and has_geometry(point) and has_geometry(following):
            px, py = point_center(previous)
            cx, cy = point_center(point)
            fx, fy = point_center(following)
            span = max(float(point.width or 1.0), float(point.height or 1.0), 1.0)
            near_previous = math.hypot(cx - px, cy - py) < span * 0.55
            near_following = math.hypot(cx - fx, cy - fy) < span * 0.55
            if near_previous and near_following:
                continue
        next_point = debounced[index]
        next_point.source = "isolated"
        next_point.object_score = min(next_point.object_score, 0.2)
        next_point.contact = min(next_point.contact, 0.08)
        next_point.grip = min(next_point.grip, 0.08)
        next_point.slip = max(next_point.slip, 0.65)
        next_point.weak = True
    return debounced


def build_timeline(video_path: Path, endpoint: str, interval: float, threshold: float, max_width: int) -> tuple[list[TrackPoint], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps else 0.0

    points: list[TrackPoint] = []
    previous: dict[str, Any] | None = None
    last_seen: TrackPoint | None = None
    sample_time = 0.0
    while sample_time <= duration + 0.001:
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        detections, latency_ms = analyze_frame(endpoint, frame, threshold, max_width)
        selected = choose_object(detections, previous, width, height)
        source = "detector"
        label = "none"
        score = 0.0
        object_score = 0.0
        contact = 0.0
        x = y = w = h = None
        if selected:
            label = str(selected.get("label", "object"))
            score = float(selected.get("score", 0.0))
            continuity = spatial_continuity(selected, previous, width, height)
            object_score = clamp(score * 0.68 + continuity * 0.22 + OBJECT_PRIOR.get(label.lower(), 0.0))
            x, y, w, h = detection_bbox(selected)
            contact_evidence = hand_contact_evidence(frame, selected)
            contact = clamp(contact_evidence * 0.74 + continuity * 0.12 + object_score * 0.14)
            if contact_evidence < CONTACT_FLOOR:
                contact = min(contact, contact_evidence * 0.8)
                object_score = min(object_score, max(0.18, score * 0.42))
            elif contact_evidence < 0.34:
                contact = min(contact, contact_evidence * 0.95)
            elif contact_evidence < 0.55:
                contact = min(contact, contact_evidence * 1.05)
            previous = selected
        elif last_seen and sample_time - last_seen.time <= 1.25:
            source = "held"
            label = last_seen.label
            age = sample_time - last_seen.time
            decay = clamp(1.0 - age / 0.85)
            object_score = last_seen.object_score * 0.42 * decay
            contact = last_seen.contact * 0.34 * decay
            score = last_seen.score * 0.38 * decay
            x, y, w, h = last_seen.x, last_seen.y, last_seen.width, last_seen.height
        grip = clamp(object_score * 0.38 + contact * 0.62)
        if contact < CONTACT_FLOOR:
            grip = min(grip, contact * 0.22)
        slip = clamp(max(0.0, 0.42 - contact) + (0.16 if source == "held" else 0.0))
        point = TrackPoint(
            time=round(sample_time, 3),
            source=source,
            label=label,
            score=score,
            object_score=object_score,
            contact=contact,
            grip=grip,
            weak=grip < 0.45 or object_score < 0.28 or contact < CONTACT_FLOOR,
            slip=slip,
            x=x,
            y=y,
            width=w,
            height=h,
            latency_ms=latency_ms,
        )
        if selected:
            last_seen = point
        points.append(point)
        sample_time += interval
    cap.release()
    metadata = {"fps": fps, "frameCount": frame_count, "width": width, "height": height, "duration": duration}
    return smooth_geometry(smooth_points(suppress_isolated_contact_spikes(points)), metadata), metadata


def interpolate_point(points: list[TrackPoint], time_value: float) -> TrackPoint:
    if not points:
        raise ValueError("No timeline points")
    if time_value <= points[0].time:
        return points[0]
    for index in range(1, len(points)):
        right = points[index]
        left = points[index - 1]
        if time_value <= right.time:
            span = max(0.001, right.time - left.time)
            t = clamp((time_value - left.time) / span)
            left_contact = left.contact >= CONTACT_FLOOR and left.grip >= 0.28
            right_contact = right.contact >= CONTACT_FLOOR and right.grip >= 0.28
            if left_contact != right_contact:
                # Do not visually smear object evidence across a true contact boundary.
                # Offline samples are sparse by design, so a hard state boundary is less
                # misleading than interpolating a fake half-grip through empty frames.
                chosen = left if t < 0.5 else right
                return TrackPoint(**asdict(chosen))
            base = TrackPoint(**asdict(left))
            for key in ("score", "object_score", "contact", "grip", "slip"):
                setattr(base, key, lerp(float(getattr(left, key)), float(getattr(right, key)), t))
            if left.x is not None and right.x is not None:
                base.x = lerp(left.x, right.x, t)
                base.y = lerp(left.y or 0.0, right.y or 0.0, t)
                base.width = lerp(left.width or 0.0, right.width or 0.0, t)
                base.height = lerp(left.height or 0.0, right.height or 0.0, t)
            base.label = left.label if t < 0.5 else right.label
            base.source = left.source if t < 0.5 else right.source
            base.weak = base.grip < 0.45 or base.object_score < 0.28
            return base
    return points[-1]


def draw_overlay(frame: Any, point: TrackPoint, provider_label: str) -> Any:
    out = frame.copy()
    height, width = out.shape[:2]
    grip_pct = round(point.grip * 100)
    color = (92, 222, 147) if point.grip >= 0.68 else (70, 215, 238) if point.grip >= 0.38 else (88, 112, 242)
    if point.x is not None and point.width and point.height:
        center = (round(point.x + point.width / 2), round((point.y or 0) + point.height / 2))
        axes = (max(12, round(point.width / 2)), max(12, round(point.height / 2)))
        evidence = out.copy()
        cv2.ellipse(evidence, center, axes, 0, 0, 360, color, -1)
        cv2.addWeighted(evidence, 0.14, out, 0.86, 0, out)
        cv2.ellipse(out, center, axes, 0, 0, 360, color, 4)
        inner_axes = (max(8, round(axes[0] * 0.74)), max(8, round(axes[1] * 0.74)))
        cv2.ellipse(out, center, inner_axes, 0, 0, 360, color, 1)
        cv2.circle(out, center, 10, (93, 206, 255), -1)
    panel_w = 430
    overlay = out.copy()
    cv2.rectangle(overlay, (24, 24), (panel_w, 214), (14, 20, 31), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
    cv2.putText(out, provider_label, (46, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (147, 235, 255), 2)
    cv2.putText(out, f"{grip_pct}%", (46, 134), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (245, 248, 255), 5)
    state = "Strong grip" if point.grip >= 0.68 else "Improve grip" if point.grip >= 0.38 else "Object uncertain"
    cv2.putText(out, state, (46, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(out, f"object {round(point.object_score * 100)}%  contact {round(point.contact * 100)}%  {point.source}", (46, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 219, 235), 1)
    bar_x, bar_y, bar_w = 24, height - 42, width - 48
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (35, 43, 58), -1)
    progress = clamp(point.time / max(0.001, point.time + 0.001))
    del progress
    return out


def render_video(video_path: Path, points: list[TrackPoint], out_path: Path, output_fps: float, provider_label: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or output_fps
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = source_frames / fps if fps else points[-1].time
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    frame_index = 0
    while frame_index / output_fps <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, (frame_index / output_fps) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        point = interpolate_point(points, frame_index / output_fps)
        writer.write(draw_overlay(frame, point, provider_label))
        frame_index += 1
    writer.release()
    cap.release()


def write_outputs(
    video_path: Path,
    out_dir: Path,
    points: list[TrackPoint],
    metadata: dict[str, Any],
    output_fps: float,
    provider: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    suffix = "offline-yolo-max" if provider == "yolo-max" else "offline-v2-rfdetr"
    provider_label = PROVIDER_LABELS.get(provider, provider)
    json_path = out_dir / f"{stem}-{suffix}-timeline.json"
    csv_path = out_dir / f"{stem}-{suffix}-timeline.csv"
    report_path = out_dir / f"{stem}-{suffix}-report.md"
    mp4_path = out_dir / f"{stem}-{suffix}-annotated.mp4"
    payload = {"video": str(video_path), "provider": provider, "metadata": metadata, "timeline": [asdict(point) for point in points]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()) if points else ["time"])
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))
    avg_grip = sum(point.grip for point in points) / max(1, len(points))
    peak_grip = max((point.grip for point in points), default=0.0)
    weak_count = sum(1 for point in points if point.weak)
    slip_count = sum(1 for point in points if point.slip > 0.45)
    report_path.write_text(
        "\n".join(
            [
                f"# {provider_label} Batch Report",
                "",
                f"- Video: `{video_path.name}`",
                f"- Duration: {metadata['duration']:.2f}s",
                f"- Samples: {len(points)} at 0.5s interval",
                f"- Average grip: {round(avg_grip * 100)}%",
                f"- Peak grip: {round(peak_grip * 100)}%",
                f"- Weak samples: {weak_count}",
                f"- Slip-risk samples: {slip_count}",
                "",
                "Notes: person detections are rejected; large background furniture detections are penalized; brief detector misses are bridged only when contact is sustained.",
            ]
        ),
        encoding="utf-8",
    )
    render_video(video_path, points, mp4_path, output_fps, provider_label)
    return {"json": json_path, "csv": csv_path, "report": report_path, "mp4": mp4_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("process"))
    parser.add_argument("--provider", choices=["rfdetr", "yolo-max"], default="rfdetr")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--output-fps", type=float, default=30.0)
    args = parser.parse_args()

    start = time.perf_counter()
    endpoint = args.endpoint or (DEFAULT_YOLO_ENDPOINT if args.provider == "yolo-max" else DEFAULT_ENDPOINT)
    points, metadata = build_timeline(args.video, endpoint, args.interval, args.threshold, args.max_width)
    paths = write_outputs(args.video, args.out_dir, points, metadata, args.output_fps, args.provider)
    elapsed = time.perf_counter() - start
    print(json.dumps({"elapsedSeconds": round(elapsed, 2), "outputs": {key: str(value) for key, value in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
