# GripSense RGB AX Transformation Pipeline Deck

This folder contains the AX Transformation submission deck for **GripSense RGB**, invented by **Dewanshu Haswani** for Samsung Research Institute Bangalore (SRIB).

## Deliverable

- `GripSense_RGB_AX_Transformation_Pipeline.pptx`  
  A 22-slide professional PowerPoint deck covering the tool, pipeline, novelty, technical architecture, cross-team usefulness, adoption plan, limitations, roadmap, and live demo placeholder.

## Deck Story

GripSense RGB is positioned as an RGB-only object-interaction intelligence platform:

1. Train object profiles from cropped/masked multi-angle images.
2. Enable the target object from a trained object list.
3. Detect the target with temporal identity checks.
4. Estimate visual grip stability from hand pose, object evidence, and motion stability.
5. Use live webcam analysis or offline uploaded-video review.

The deck emphasizes that the grip percentage is **visual grip stability**, not true physical force. This makes the tool useful as an honest early-stage R&D platform before adding depth cameras, pressure sensors, smart objects, gloves, or other hardware.

## Teams Covered

- XR: markerless hand-object interaction probes and AR grip-state cues.
- Visual Intelligence: object-hand reasoning datasets, model evaluation, failure mining, and explainability.
- Robotics: human demonstration analysis, grasp affordance evidence, and manipulation test labels.
- On-device AI: privacy-first local inference, embedding readiness, latency/robustness benchmarking.

## App Screenshots And Assets

The deck includes project-local assets from `documentation/assets/`:

- `gripsense-v4-live-app.png`: current V4 desktop UI screenshot.
- `gripsense-v4-mobile.png`: current V4 mobile-width UI screenshot.
- `gripsense-field-hand-crop.png`: cropped field-observation screenshot focused on the hand overlay.
- `gripsense-ax-concept.png`: generated concept visual for the title slide.

The full face-visible field screenshot was intentionally not kept in this folder so the public repository remains safer.

## Live Demo Placeholder

Slide 12 and Slide 22 include clean placeholders for a live demo video. Recommended demo flow:

1. Open `http://127.0.0.1:5173/?version=v4`.
2. Train or select a target object.
3. Enable that object in the trained object list.
4. Show target detection turning stable/green after several frames.
5. Hold and move the object to show grip percentage and motion stability.
6. Upload a short video in offline mode and export CSV/JSON.

## Animation Note

The deck is structured as a reveal-friendly story with staged pipeline and adoption slides. The artifact export path keeps the file editable but does not add PowerPoint-native animations automatically. If animations are desired before submission, apply simple PowerPoint animations manually:

- Fade in each pipeline stage on Slide 5.
- Wipe or fade the state machine from left to right on Slide 9.
- Fade in team impact bullets on Slides 14-17.
- Replace the placeholders on Slides 12 and 22 with the final demo video.

## Generated Image Prompt

The title visual was generated with the built-in `imagegen` tool using this intent:

> A professional concept visual for an RGB-only grip estimation platform used in R&D, showing a human hand holding a generic object, webcam/computer vision overlays, hand landmarks, object contour, and subtle AI analysis lines. No text, no logo, no watermark.

