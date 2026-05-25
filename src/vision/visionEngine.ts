import {
  FilesetResolver,
  HandLandmarker,
  InteractiveSegmenter,
  ObjectDetector
} from '@mediapipe/tasks-vision';
import type { Landmark, Point } from './types';

const VISION_WASM_URL = '/mediapipe/wasm';
const HAND_MODEL_URL = '/mediapipe/models/hand_landmarker.task';
const OBJECT_MODEL_URL = '/mediapipe/models/efficientdet_lite0.tflite';
const SEGMENTER_MODEL_URL = '/mediapipe/models/magic_touch.tflite';

export type ModelState = 'idle' | 'loading' | 'ready' | 'failed';

export type VisionModelStatus = {
  hands: ModelState;
  detector: ModelState;
  segmenter: ModelState;
};

export type VisionEngine = {
  status: VisionModelStatus;
  detectHands(video: HTMLVideoElement, timestamp: number): Landmark[][];
  detectObjectBox(video: HTMLVideoElement, timestamp: number): DOMRectReadOnly | null;
  segmentAt(video: HTMLVideoElement, point: Point): Promise<boolean>;
  dispose(): void;
};

export async function createVisionEngine(onStatus: (status: VisionModelStatus) => void): Promise<VisionEngine> {
  const status: VisionModelStatus = { hands: 'loading', detector: 'loading', segmenter: 'loading' };
  const emit = () => onStatus({ ...status });
  emit();

  const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);
  let handLandmarker: HandLandmarker | null = null;
  let objectDetector: ObjectDetector | null = null;
  let segmenter: InteractiveSegmenter | null = null;

  try {
    handLandmarker = await createWithDelegateFallback((delegate) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.38,
        minHandPresenceConfidence: 0.38,
        minTrackingConfidence: 0.42
      })
    );
    status.hands = 'ready';
  } catch (error) {
    console.error('Hand model failed to load', describeModelError(error));
    status.hands = 'failed';
  }
  emit();

  try {
    objectDetector = await createWithDelegateFallback((delegate) =>
      ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: OBJECT_MODEL_URL,
          delegate
        },
        runningMode: 'VIDEO',
        scoreThreshold: 0.28,
        maxResults: 3
      })
    );
    status.detector = 'ready';
  } catch (error) {
    console.warn('Object detector is unavailable; continuing with heuristic tracking.', describeModelError(error));
    status.detector = 'failed';
  }
  emit();

  try {
    segmenter = await createWithDelegateFallback((delegate) =>
      InteractiveSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: SEGMENTER_MODEL_URL,
          delegate
        },
        runningMode: 'IMAGE',
        outputCategoryMask: true,
        outputConfidenceMasks: false
      })
    );
    status.segmenter = 'ready';
  } catch (error) {
    console.warn('Interactive segmenter is unavailable; click-to-lock still works without masks.', describeModelError(error));
    status.segmenter = 'failed';
  }
  emit();

  return {
    status,
    detectHands(video, timestamp) {
      if (!handLandmarker || status.hands !== 'ready') return [];
      const result = handLandmarker.detectForVideo(video, timestamp);
      return (result.landmarks ?? []) as Landmark[][];
    },
    detectObjectBox(video, timestamp) {
      if (!objectDetector || status.detector !== 'ready') return null;
      const detections = objectDetector.detectForVideo(video, timestamp).detections ?? [];
      const firstBox = detections[0]?.boundingBox;
      if (!firstBox) return null;
      return new DOMRectReadOnly(firstBox.originX, firstBox.originY, firstBox.width, firstBox.height);
    },
    async segmentAt(video, point) {
      if (!segmenter || status.segmenter !== 'ready') return false;
      try {
        await new Promise<void>((resolve) => {
          segmenter!.segment(video, { keypoint: point }, () => resolve());
        });
        return true;
      } catch (error) {
        console.warn('Interactive segment request failed', error);
        return false;
      }
    },
    dispose() {
      handLandmarker?.close();
      objectDetector?.close();
      segmenter?.close();
    }
  };
}

async function createWithDelegateFallback<T>(create: (delegate: 'GPU' | 'CPU') => Promise<T>) {
  try {
    return await create('GPU');
  } catch (gpuError) {
    console.warn('GPU delegate unavailable; retrying MediaPipe model on CPU.', describeModelError(gpuError));
    return create('CPU');
  }
}

function describeModelError(error: unknown) {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return String(error);
}
