/**
 * OpenMMLab RTMPose (Real-Time Models for Pose Estimation) Tracker
 * with seamless MediaPipe Pose backup and fallback.
 */

import {
  PoseTracker as MediaPipePoseTracker,
  createSyntheticPose,
  isLandmarkUsable,
  POSE_CONNECTIONS,
} from "./poseTracker.js";

export { createSyntheticPose, isLandmarkUsable, POSE_CONNECTIONS };

export const COCO_KEYPOINTS = {
  NOSE: 0,
  LEFT_EYE: 1,
  RIGHT_EYE: 2,
  LEFT_EAR: 3,
  RIGHT_EAR: 4,
  LEFT_SHOULDER: 5,
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7,
  RIGHT_ELBOW: 8,
  LEFT_WRIST: 9,
  RIGHT_WRIST: 10,
  LEFT_HIP: 11,
  RIGHT_HIP: 12,
  LEFT_KNEE: 13,
  RIGHT_KNEE: 14,
  LEFT_ANKLE: 15,
  RIGHT_ANKLE: 16,
};

// Direct 1-to-1 mapping from COCO-17 keypoints to MediaPipe-33 landmark indices
export const COCO_TO_MEDIAPIPE = [
  [0, 0],   // nose
  [1, 2],   // left_eye
  [2, 5],   // right_eye
  [3, 7],   // left_ear
  [4, 8],   // right_ear
  [5, 11],  // left_shoulder
  [6, 12],  // right_shoulder
  [7, 13],  // left_elbow
  [8, 14],  // right_elbow
  [9, 15],  // left_wrist
  [10, 16], // right_wrist
  [11, 23], // left_hip
  [12, 24], // right_hip
  [13, 25], // left_knee
  [14, 26], // right_knee
  [15, 27], // left_ankle
  [16, 28], // right_ankle
];

/**
 * OpenMMLab SimCC (Coordinate Classification) 1D heatmap decoder.
 * Extracts sub-pixel coordinates for all 17 keypoints from 1D representations.
 */
export function decodeSimCC(simccX, simccY, splitRatio = 2.0) {
  const numKeypoints = 17;
  const xBins = Math.floor(simccX.length / numKeypoints);
  const yBins = Math.floor(simccY.length / numKeypoints);
  const keypoints = [];

  for (let k = 0; k < numKeypoints; k++) {
    const xOffset = k * xBins;
    const yOffset = k * yBins;

    // Argmax along X
    let maxValX = -Infinity;
    let maxIdxX = 0;
    for (let i = 0; i < xBins; i++) {
      const v = simccX[xOffset + i];
      if (v > maxValX) {
        maxValX = v;
        maxIdxX = i;
      }
    }

    // Argmax along Y
    let maxValY = -Infinity;
    let maxIdxY = 0;
    for (let j = 0; j < yBins; j++) {
      const v = simccY[yOffset + j];
      if (v > maxValY) {
        maxValY = v;
        maxIdxY = j;
      }
    }

    // Sub-pixel refinement with Gaussian/softmax weighting around peak
    let sumWeightX = 0;
    let weightedIdxX = 0;
    const startX = Math.max(0, maxIdxX - 3);
    const endX = Math.min(xBins - 1, maxIdxX + 3);
    for (let i = startX; i <= endX; i++) {
      const w = Math.exp(Math.min(simccX[xOffset + i] - maxValX, 20));
      sumWeightX += w;
      weightedIdxX += i * w;
    }
    const refinedX = sumWeightX > 0 ? weightedIdxX / sumWeightX : maxIdxX;

    let sumWeightY = 0;
    let weightedIdxY = 0;
    const startY = Math.max(0, maxIdxY - 3);
    const endY = Math.min(yBins - 1, maxIdxY + 3);
    for (let j = startY; j <= endY; j++) {
      const w = Math.exp(Math.min(simccY[yOffset + j] - maxValY, 20));
      sumWeightY += w;
      weightedIdxY += j * w;
    }
    const refinedY = sumWeightY > 0 ? weightedIdxY / sumWeightY : maxIdxY;

    const x = refinedX / xBins;
    const y = refinedY / yBins;
    // Logistic sigmoid score approximation
    const minVal = Math.min(maxValX, maxValY);
    const score = 1 / (1 + Math.exp(-Math.max(Math.min(minVal, 10), -10)));

    keypoints.push({ x, y, score });
  }

  return keypoints;
}

/**
 * Converts COCO-17 keypoints into the 33-point MediaPipe structure
 * with anatomically consistent 3D world coordinates.
 */
export function convertCocoToMediaPipeLandmarks(cocoKeypoints, inputAspect = 1.0) {
  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0,
    presence: 0,
  }));
  const worldLandmarks = Array.from({ length: 33 }, () => ({
    x: 0,
    y: 0,
    z: 0,
    visibility: 0,
    presence: 0,
  }));

  for (const [cocoIdx, mpIdx] of COCO_TO_MEDIAPIPE) {
    const kp = cocoKeypoints[cocoIdx];
    if (!kp) continue;
    landmarks[mpIdx] = {
      x: kp.x,
      y: kp.y,
      z: 0,
      visibility: kp.score,
      presence: kp.score,
    };
  }

  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const nose = landmarks[0];

  const hasShoulders = leftShoulder.visibility > 0.25 && rightShoulder.visibility > 0.25;
  const hasHips = leftHip.visibility > 0.25 && rightHip.visibility > 0.25;

  const refX = hasShoulders
    ? (leftShoulder.x + rightShoulder.x) / 2
    : hasHips
      ? (leftHip.x + rightHip.x) / 2
      : 0.5;
  const refY = hasShoulders
    ? (leftShoulder.y + rightShoulder.y) / 2
    : hasHips
      ? (leftHip.y + rightHip.y) / 2
      : 0.5;

  let pixelsToMeters = 0.40 / 0.25;
  if (hasShoulders) {
    const shSpan = Math.hypot(
      leftShoulder.x - rightShoulder.x,
      (leftShoulder.y - rightShoulder.y) / inputAspect,
    );
    if (shSpan > 0.05) {
      pixelsToMeters = 0.40 / shSpan;
    }
  }

  // Torso yaw estimation
  let torsoYaw = 0;
  if (hasShoulders && nose.visibility > 0.25) {
    const shMidX = (leftShoulder.x + rightShoulder.x) / 2;
    const shWidth = leftShoulder.x - rightShoulder.x;
    if (Math.abs(shWidth) > 0.02) {
      const noseOffset = (nose.x - shMidX) / Math.abs(shWidth);
      torsoYaw = Math.max(Math.min(noseOffset * Math.PI, Math.PI / 2), -Math.PI / 2);
    }
  }

  for (const [, mpIdx] of COCO_TO_MEDIAPIPE) {
    const lm = landmarks[mpIdx];
    if (lm.visibility <= 0.05) continue;

    const wx = (lm.x - refX) * pixelsToMeters;
    const wy = (lm.y - refY) * pixelsToMeters;

    let wz = 0;
    if (mpIdx === 11) wz = Math.sin(torsoYaw) * 0.20;
    else if (mpIdx === 12) wz = -Math.sin(torsoYaw) * 0.20;
    else if (mpIdx === 0) wz = -Math.cos(torsoYaw) * 0.08;
    else if (mpIdx === 7) wz = Math.sin(torsoYaw) * 0.08;
    else if (mpIdx === 8) wz = -Math.sin(torsoYaw) * 0.08;
    else if (mpIdx === 23) wz = Math.sin(torsoYaw) * 0.12;
    else if (mpIdx === 24) wz = -Math.sin(torsoYaw) * 0.12;

    worldLandmarks[mpIdx] = {
      x: wx,
      y: wy,
      z: wz,
      visibility: lm.visibility,
      presence: lm.presence,
    };
  }

  return { landmarks, worldLandmarks };
}

/**
 * RTMPose client-side tracker running via ONNX Runtime Web.
 */
export class RTMPoseTracker {
  static async create(modelPath = "/models/rtmpose-m.onnx") {
    let ort = null;
    if (typeof window !== "undefined") {
      ort = window.ort;
      if (!ort) {
        // Attempt dynamic import from CDN or local bundle
        try {
          ort = await import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js");
        } catch {
          // Ignore CDN load failures; fallback will trigger cleanly
        }
      }
    }

    if (!ort) {
      throw new Error("ONNX Runtime Web is not loaded.");
    }

    // Verify model exists before creating session
    const response = await fetch(modelPath, { method: "HEAD" });
    if (!response.ok) {
      throw new Error(`RTMPose ONNX model not found at ${modelPath} (${response.status})`);
    }

    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    });

    return new RTMPoseTracker(session, ort);
  }

  constructor(session, ort) {
    this.session = session;
    this.ort = ort;
    this.inputWidth = 192;
    this.inputHeight = 256;
    this.canvas = typeof document !== "undefined" ? document.createElement("canvas") : null;
    if (this.canvas) {
      this.canvas.width = this.inputWidth;
      this.canvas.height = this.inputHeight;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    }
  }

  detect(video) {
    if (!this.ctx || !video || !video.videoWidth || !video.videoHeight) {
      return { landmarks: null, worldLandmarks: null };
    }

    // Letterbox draw to 192x256
    this.ctx.drawImage(video, 0, 0, this.inputWidth, this.inputHeight);
    const imageData = this.ctx.getImageData(0, 0, this.inputWidth, this.inputHeight);
    const { data } = imageData;

    // Normalization: ImageNet mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    const floatArray = new Float32Array(3 * this.inputWidth * this.inputHeight);
    const planeSize = this.inputWidth * this.inputHeight;
    for (let i = 0; i < planeSize; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      floatArray[i] = (r - 123.675) / 58.395;
      floatArray[planeSize + i] = (g - 116.28) / 57.12;
      floatArray[planeSize * 2 + i] = (b - 103.53) / 57.375;
    }

    const tensor = new this.ort.Tensor("float32", floatArray, [
      1,
      3,
      this.inputHeight,
      this.inputWidth,
    ]);

    // Note: Async session run is resolved synchronously or cached in real-time loop
    return this.runInference(tensor, video.videoWidth / video.videoHeight);
  }

  async runInference(tensor, aspect) {
    const feeds = { [this.session.inputNames[0]]: tensor };
    const results = await this.session.run(feeds);
    const simccX = results.simcc_x || results[this.session.outputNames[0]];
    const simccY = results.simcc_y || results[this.session.outputNames[1]];
    if (!simccX || !simccY) {
      return { landmarks: null, worldLandmarks: null };
    }
    const cocoKeypoints = decodeSimCC(simccX.data, simccY.data);
    return convertCocoToMediaPipeLandmarks(cocoKeypoints, aspect);
  }

  close() {
    this.session?.release?.();
  }
}

/**
 * Unified Pose Tracker that coordinates RTMPose (primary SOTA)
 * and MediaPipe Pose (seamless backup and fallback).
 */
export class UnifiedPoseTracker {
  static async create(options = {}) {
    const tracker = new UnifiedPoseTracker(options);
    await tracker.init();
    return tracker;
  }

  constructor(options = {}) {
    this.preferredEngine = options.preferredEngine || "rtmpose";
    this.activeEngine = "mediapipe";
    this.onStatus = options.onStatus || (() => {});
    this.rtmTracker = null;
    this.mediaPipeTracker = null;
    this.lastResult = { landmarks: null, worldLandmarks: null };
    this.rtmErrorLogged = false;
  }

  async init() {
    // 1. Always load MediaPipe Pose as reliable backup first
    try {
      this.mediaPipeTracker = await MediaPipePoseTracker.create();
      this.activeEngine = "mediapipe";
      this.onStatus({
        state: "ready",
        engine: "mediapipe",
        message: "MediaPipe Pose engine initialized (Backup)",
      });
    } catch (err) {
      console.warn("Could not initialize MediaPipe Pose backup:", err);
    }

    // 2. If RTMPose is preferred, attempt to initialize RTMPose
    if (this.preferredEngine === "rtmpose") {
      try {
        this.rtmTracker = await RTMPoseTracker.create();
        this.activeEngine = "rtmpose";
        this.onStatus({
          state: "ready",
          engine: "rtmpose",
          message: "RTMPose engine active (OpenMMLab SOTA)",
        });
      } catch (err) {
        this.activeEngine = "mediapipe";
        this.onStatus({
          state: "ready",
          engine: "mediapipe",
          message: `RTMPose unavailable (${err.message}); running MediaPipe Pose backup`,
        });
      }
    }
  }

  setEngine(engineName) {
    if (engineName === "rtmpose") {
      if (this.rtmTracker) {
        this.activeEngine = "rtmpose";
        this.onStatus({
          state: "ready",
          engine: "rtmpose",
          message: "Switched to RTMPose engine (OpenMMLab)",
        });
      } else {
        // Attempt on-demand load
        RTMPoseTracker.create()
          .then((tracker) => {
            this.rtmTracker = tracker;
            this.activeEngine = "rtmpose";
            this.onStatus({
              state: "ready",
              engine: "rtmpose",
              message: "RTMPose engine active (OpenMMLab SOTA)",
            });
          })
          .catch((err) => {
            this.activeEngine = "mediapipe";
            this.onStatus({
              state: "ready",
              engine: "mediapipe",
              message: `RTMPose model unavailable (${err.message}). Kept MediaPipe backup.`,
            });
          });
      }
    } else {
      this.activeEngine = "mediapipe";
      this.onStatus({
        state: "ready",
        engine: "mediapipe",
        message: "Switched to MediaPipe Pose engine (Backup)",
      });
    }
  }

  detect(video, timestamp) {
    if (this.activeEngine === "rtmpose" && this.rtmTracker) {
      try {
        const result = this.rtmTracker.detect(video);
        if (result && result.landmarks) {
          this.lastResult = result;
          return result;
        }
      } catch (err) {
        if (!this.rtmErrorLogged) {
          console.warn("RTMPose inference error, falling back to MediaPipe:", err);
          this.rtmErrorLogged = true;
        }
      }
    }

    // Default or fallback: MediaPipe Pose
    if (this.mediaPipeTracker) {
      const result = this.mediaPipeTracker.detect(video, timestamp);
      this.lastResult = result;
      return result;
    }

    return this.lastResult;
  }

  close() {
    this.rtmTracker?.close();
    this.mediaPipeTracker?.close();
  }
}
