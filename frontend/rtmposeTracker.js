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
import { RTMW3DTracker } from "./src/rtmw3dTracker.js";

export { createSyntheticPose, isLandmarkUsable, POSE_CONNECTIONS, RTMW3DTracker };

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
 * 1-Euro Adaptive Low-Pass Filter for real-time 2D landmark coordinates.
 * Eliminates micro-jitter and shivering while standing still,
 * and decreases filtering dynamically during fast movement to prevent lag.
 */
export class OneEuroPointFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = 0;
    this.yPrev = 0;
    this.dxPrev = 0;
    this.dyPrev = 0;
    this.hasValue = false;
    this.tPrev = 0;
  }

  reset() {
    this.hasValue = false;
    this.tPrev = 0;
  }

  alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, y, timestampSeconds) {
    if (!this.hasValue || this.tPrev <= 0) {
      this.xPrev = x;
      this.yPrev = y;
      this.dxPrev = 0;
      this.dyPrev = 0;
      this.tPrev = timestampSeconds;
      this.hasValue = true;
      return { x, y };
    }

    const dt = Math.min(Math.max(timestampSeconds - this.tPrev, 0.001), 0.1);
    this.tPrev = timestampSeconds;

    // Filter derivative (velocity)
    const dx = (x - this.xPrev) / dt;
    const dy = (y - this.yPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const dyHat = aD * dy + (1 - aD) * this.dyPrev;
    this.dxPrev = dxHat;
    this.dyPrev = dyHat;

    // Adaptive cutoff based on speed
    const speed = Math.hypot(dxHat, dyHat);
    const cutoff = this.minCutoff + this.beta * speed;
    const a = this.alpha(cutoff, dt);

    const xHat = a * x + (1 - a) * this.xPrev;
    const yHat = a * y + (1 - a) * this.yPrev;
    this.xPrev = xHat;
    this.yPrev = yHat;

    return { x: xHat, y: yHat };
  }
}

/**
 * 1-Euro Adaptive Low-Pass Filter for real-time 3D landmark coordinates (x, y, z).
 */
export class OneEuroPointFilter3D {
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = 0;
    this.yPrev = 0;
    this.zPrev = 0;
    this.dxPrev = 0;
    this.dyPrev = 0;
    this.dzPrev = 0;
    this.hasValue = false;
    this.tPrev = 0;
  }

  reset() {
    this.hasValue = false;
    this.tPrev = 0;
  }

  alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(x, y, z, timestampSeconds) {
    if (!this.hasValue || this.tPrev <= 0) {
      this.xPrev = x;
      this.yPrev = y;
      this.zPrev = z;
      this.dxPrev = 0;
      this.dyPrev = 0;
      this.dzPrev = 0;
      this.tPrev = timestampSeconds;
      this.hasValue = true;
      return { x, y, z };
    }

    const dt = Math.min(Math.max(timestampSeconds - this.tPrev, 0.001), 0.1);
    this.tPrev = timestampSeconds;

    // Filter derivative (velocity)
    const dx = (x - this.xPrev) / dt;
    const dy = (y - this.yPrev) / dt;
    const dz = (z - this.zPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const dyHat = aD * dy + (1 - aD) * this.dyPrev;
    const dzHat = aD * dz + (1 - aD) * this.dzPrev;
    this.dxPrev = dxHat;
    this.dyPrev = dyHat;
    this.dzPrev = dzHat;

    // Adaptive cutoff based on 3D speed
    const speed = Math.hypot(dxHat, dyHat, dzHat);
    const cutoff = this.minCutoff + this.beta * speed;
    const a = this.alpha(cutoff, dt);

    const xHat = a * x + (1 - a) * this.xPrev;
    const yHat = a * y + (1 - a) * this.yPrev;
    const zHat = a * z + (1 - a) * this.zPrev;
    this.xPrev = xHat;
    this.yPrev = yHat;
    this.zPrev = zHat;

    return { x: xHat, y: yHat, z: zHat };
  }
}

/**
 * OpenMMLab SimCC (Coordinate Classification) 1D heatmap decoder.
 * Extracts sub-pixel coordinates for all 17 keypoints from 1D representations,
 * with boundary edge clamping rejection and prominence contrast scoring.
 */
export function decodeSimCC(simccX, simccY, splitRatio = 2.0) {
  const numKeypoints = 17;
  const xBins = Math.floor(simccX.length / numKeypoints);
  const yBins = Math.floor(simccY.length / numKeypoints);
  const keypoints = [];

  for (let k = 0; k < numKeypoints; k++) {
    const xOffset = k * xBins;
    const yOffset = k * yBins;

    // Argmax along X and compute mean
    let maxValX = -Infinity;
    let maxIdxX = 0;
    let sumX = 0;
    for (let i = 0; i < xBins; i++) {
      const v = simccX[xOffset + i];
      sumX += v;
      if (v > maxValX) {
        maxValX = v;
        maxIdxX = i;
      }
    }
    const meanX = sumX / xBins;

    // Argmax along Y and compute mean
    let maxValY = -Infinity;
    let maxIdxY = 0;
    let sumY = 0;
    for (let j = 0; j < yBins; j++) {
      const v = simccY[yOffset + j];
      sumY += v;
      if (v > maxValY) {
        maxValY = v;
        maxIdxY = j;
      }
    }
    const meanY = sumY / yBins;

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

    // Reject boundary artifacts: when joint is occluded, argmax clamps to edge bins
    const isBoundary = maxIdxX <= 2 || maxIdxX >= xBins - 3 || maxIdxY <= 2 || maxIdxY >= yBins - 3;
    const prominence = Math.min(maxValX - meanX, maxValY - meanY);

    let score = 0;
    if (!isBoundary && prominence > 0.10) {
      score = Math.min(Math.max((prominence - 0.10) / 0.35, 0), 1.0);
    }

    keypoints.push({ x, y, score });
  }

  return keypoints;
}

/**
 * OpenMMLab 3D SimCC (Coordinate Classification) decoder.
 * Decodes 3D coordinates (x, y, metric depth z) for all 17 COCO keypoints.
 * simccZ represents root-relative depth in meters, mapped across [depthMin, depthMax].
 */
export function decodeSimCC3D(simccX, simccY, simccZ, depthMin = -1.2, depthMax = 1.2) {
  const numKeypoints = 17;
  const xBins = Math.floor(simccX.length / numKeypoints);
  const yBins = Math.floor(simccY.length / numKeypoints);
  const zBins = simccZ ? Math.floor(simccZ.length / numKeypoints) : 0;
  const keypoints = [];

  for (let k = 0; k < numKeypoints; k++) {
    const xOffset = k * xBins;
    const yOffset = k * yBins;

    // Argmax along X and compute mean
    let maxValX = -Infinity;
    let maxIdxX = 0;
    let sumX = 0;
    for (let i = 0; i < xBins; i++) {
      const v = simccX[xOffset + i];
      sumX += v;
      if (v > maxValX) {
        maxValX = v;
        maxIdxX = i;
      }
    }
    const meanX = sumX / xBins;

    // Argmax along Y and compute mean
    let maxValY = -Infinity;
    let maxIdxY = 0;
    let sumY = 0;
    for (let j = 0; j < yBins; j++) {
      const v = simccY[yOffset + j];
      sumY += v;
      if (v > maxValY) {
        maxValY = v;
        maxIdxY = j;
      }
    }
    const meanY = sumY / yBins;

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

    // Z decoding (root-relative metric depth in meters)
    let z = 0;
    if (simccZ && zBins > 0) {
      const zOffset = k * zBins;
      let maxValZ = -Infinity;
      let maxIdxZ = 0;
      for (let m = 0; m < zBins; m++) {
        const v = simccZ[zOffset + m];
        if (v > maxValZ) {
          maxValZ = v;
          maxIdxZ = m;
        }
      }
      let sumWeightZ = 0;
      let weightedIdxZ = 0;
      const startZ = Math.max(0, maxIdxZ - 3);
      const endZ = Math.min(zBins - 1, maxIdxZ + 3);
      for (let m = startZ; m <= endZ; m++) {
        const w = Math.exp(Math.min(simccZ[zOffset + m] - maxValZ, 20));
        sumWeightZ += w;
        weightedIdxZ += m * w;
      }
      const refinedZ = sumWeightZ > 0 ? weightedIdxZ / sumWeightZ : maxIdxZ;
      const normZ = refinedZ / (zBins - 1);
      z = normZ * (depthMax - depthMin) + depthMin;
    }

    const x = refinedX / xBins;
    const y = refinedY / yBins;

    // Reject boundary artifacts
    const isBoundary = maxIdxX <= 2 || maxIdxX >= xBins - 3 || maxIdxY <= 2 || maxIdxY >= yBins - 3;
    const prominence = Math.min(maxValX - meanX, maxValY - meanY);

    let score = 0;
    if (!isBoundary && prominence > 0.10) {
      score = Math.min(Math.max((prominence - 0.10) / 0.35, 0), 1.0);
    }

    keypoints.push({ x, y, z, hasZ: !!simccZ, score });
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
      z: kp.z || 0,
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

  // Head yaw & pitch estimation from facial triangle (nose, ears)
  const leftEar = landmarks[7];
  const rightEar = landmarks[8];
  const hasEars = leftEar && rightEar && leftEar.visibility > 0.25 && rightEar.visibility > 0.25;
  let headYaw = torsoYaw;
  let headPitch = 0;
  if (hasEars && nose.visibility > 0.25) {
    const earMidX = (leftEar.x + rightEar.x) * 0.5;
    const earMidY = (leftEar.y + rightEar.y) * 0.5;
    const earSpan = Math.hypot(leftEar.x - rightEar.x, (leftEar.y - rightEar.y) / inputAspect);
    if (earSpan > 0.02) {
      const normX = (nose.x - earMidX) / (earSpan * 0.5);
      headYaw = Math.max(Math.min(normX * (Math.PI / 2), Math.PI / 2), -Math.PI / 2);
      const normY = (nose.y - earMidY) / (earSpan * 0.5);
      headPitch = Math.max(Math.min(normY * (Math.PI / 3), Math.PI / 3), -Math.PI / 3);
    }
  } else if (leftEar && leftEar.visibility > 0.25 && nose.visibility > 0.25) {
    headYaw = Math.PI / 3;
  } else if (rightEar && rightEar.visibility > 0.25 && nose.visibility > 0.25) {
    headYaw = -Math.PI / 3;
  }

  for (const [cocoIdx, mpIdx] of COCO_TO_MEDIAPIPE) {
    const lm = landmarks[mpIdx];
    if (lm.visibility <= 0.05) continue;

    const wx = (lm.x - refX) * pixelsToMeters;
    const wy = (lm.y - refY) * pixelsToMeters;

    let wz = 0;
    const kp = cocoKeypoints[cocoIdx];
    if (kp && typeof kp.z === "number" && (kp.z !== 0 || kp.hasZ)) {
      // Direct metric depth from 3D model
      wz = kp.z;
    } else {
      // Anatomical depth approximation
      if (mpIdx === 11) wz = Math.sin(torsoYaw) * 0.20;
      else if (mpIdx === 12) wz = -Math.sin(torsoYaw) * 0.20;
      else if (mpIdx === 0) wz = -Math.cos(headYaw) * 0.12 * Math.cos(headPitch);
      else if (mpIdx === 7) wz = Math.sin(headYaw) * 0.08;
      else if (mpIdx === 8) wz = -Math.sin(headYaw) * 0.08;
      else if (mpIdx === 23) wz = Math.sin(torsoYaw) * 0.12;
      else if (mpIdx === 24) wz = -Math.sin(torsoYaw) * 0.12;
    }

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
    this.isInferencing = false;
    this.latestResult = { landmarks: null, worldLandmarks: null };
    this.filters = Array.from({ length: 17 }, () => new OneEuroPointFilter(1.2, 0.008, 1.0));
    this.retainedKeypoints = Array.from({ length: 17 }, () => null);
    this.canvas = typeof document !== "undefined" ? document.createElement("canvas") : null;
    if (this.canvas) {
      this.canvas.width = this.inputWidth;
      this.canvas.height = this.inputHeight;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    }
  }

  smoothKeypoints(rawKeypoints, timestamp) {
    const t = (timestamp || performance.now()) / 1000;
    const smoothed = [];

    for (let k = 0; k < 17; k++) {
      const raw = rawKeypoints[k];
      if (!raw) continue;

      if (raw.score > 0.25) {
        // Valid detection: apply 1-Euro adaptive low-pass filter
        const pt = this.filters[k].filter(raw.x, raw.y, t);
        const kp = { x: pt.x, y: pt.y, score: raw.score };
        smoothed.push(kp);
        this.retainedKeypoints[k] = { ...kp, lostFrames: 0 };
      } else {
        // Low confidence / boundary rejection: smoothly retain last valid position
        const prev = this.retainedKeypoints[k];
        if (prev && prev.lostFrames < 30) {
          prev.lostFrames += 1;
          const decayedScore = Math.max(prev.score * 0.94, 0.15);
          smoothed.push({ x: prev.x, y: prev.y, score: decayedScore });
        } else {
          smoothed.push(raw);
        }
      }
    }

    return smoothed;
  }

  detect(video, timestamp) {
    if (!this.ctx || !video || !video.videoWidth || !video.videoHeight) {
      return this.latestResult;
    }

    if (!this.isInferencing) {
      this.isInferencing = true;
      // Draw frame to 192x256
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

      const aspect = video.videoWidth / video.videoHeight;
      const feeds = { [this.session.inputNames[0]]: tensor };
      this.session
        .run(feeds)
        .then((results) => {
          const simccX = results.simcc_x || results[this.session.outputNames[0]];
          const simccY = results.simcc_y || results[this.session.outputNames[1]];
          if (simccX && simccY) {
            const rawKeypoints = decodeSimCC(simccX.data, simccY.data);
            const cocoKeypoints = this.smoothKeypoints(rawKeypoints, timestamp);
            this.latestResult = convertCocoToMediaPipeLandmarks(cocoKeypoints, aspect);
          }
        })
        .catch((err) => {
          console.warn("RTMPose session.run error:", err);
        })
        .finally(() => {
          this.isInferencing = false;
        });
    }

    return this.latestResult;
  }

  close() {
    this.session?.release?.();
  }
}

/**
 * RTMPose 3D client-side tracker running OpenMMLab SimCC-3D via ONNX Runtime Web.
 */
export class RTMPose3DTracker {
  static async create(candidatePaths = ["/models/rtmpose3d-m.onnx", "/models/rtmpose-3d.onnx", "/models/rtmpose3d.onnx"]) {
    let ort = null;
    if (typeof window !== "undefined") {
      ort = window.ort;
      if (!ort) {
        try {
          ort = await import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js");
        } catch {}
      }
    }

    if (!ort) {
      throw new Error("ONNX Runtime Web is not loaded.");
    }

    let foundPath = null;
    for (const path of candidatePaths) {
      try {
        const response = await fetch(path, { method: "HEAD" });
        if (response.ok) {
          foundPath = path;
          break;
        }
      } catch {}
    }

    if (!foundPath) {
      throw new Error(`RTMPose 3D ONNX model not found in ${candidatePaths.join(" or ")}`);
    }

    const session = await ort.InferenceSession.create(foundPath, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
    });

    return new RTMPose3DTracker(session, ort, foundPath);
  }

  constructor(session, ort, modelPath) {
    this.session = session;
    this.ort = ort;
    this.modelPath = modelPath;
    this.inputWidth = 256;
    this.inputHeight = 256;
    this.isInferencing = false;
    this.latestResult = { landmarks: null, worldLandmarks: null };
    this.filters = Array.from({ length: 17 }, () => new OneEuroPointFilter3D(1.2, 0.008, 1.0));
    this.retainedKeypoints = Array.from({ length: 17 }, () => null);
    this.canvas = typeof document !== "undefined" ? document.createElement("canvas") : null;
    if (this.canvas) {
      this.canvas.width = this.inputWidth;
      this.canvas.height = this.inputHeight;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    }
  }

  smoothKeypoints(rawKeypoints, timestamp) {
    const t = (timestamp || performance.now()) / 1000;
    const smoothed = [];

    for (let k = 0; k < 17; k++) {
      const raw = rawKeypoints[k];
      if (!raw) continue;

      if (raw.score > 0.25) {
        const pt = this.filters[k].filter(raw.x, raw.y, raw.z || 0, t);
        const kp = { x: pt.x, y: pt.y, z: pt.z, hasZ: true, score: raw.score };
        smoothed.push(kp);
        this.retainedKeypoints[k] = { ...kp, lostFrames: 0 };
      } else {
        const prev = this.retainedKeypoints[k];
        if (prev && prev.lostFrames < 30) {
          prev.lostFrames += 1;
          const decayedScore = Math.max(prev.score * 0.94, 0.15);
          smoothed.push({ x: prev.x, y: prev.y, z: prev.z, hasZ: true, score: decayedScore });
        } else {
          smoothed.push(raw);
        }
      }
    }

    return smoothed;
  }

  detect(video, timestamp) {
    if (!this.ctx || !video || !video.videoWidth || !video.videoHeight) {
      return this.latestResult;
    }

    if (!this.isInferencing) {
      this.isInferencing = true;
      this.ctx.drawImage(video, 0, 0, this.inputWidth, this.inputHeight);
      const imageData = this.ctx.getImageData(0, 0, this.inputWidth, this.inputHeight);
      const { data } = imageData;

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

      const aspect = video.videoWidth / video.videoHeight;
      const feeds = { [this.session.inputNames[0]]: tensor };
      this.session
        .run(feeds)
        .then((results) => {
          const simccX = results.simcc_x || results[this.session.outputNames[0]];
          const simccY = results.simcc_y || results[this.session.outputNames[1]];
          const simccZ = results.simcc_z || (this.session.outputNames.length > 2 ? results[this.session.outputNames[2]] : null);
          const keypoints3D = results.keypoints_3d || results.simcc_3d || results.joints_3d;

          let rawKeypoints = null;
          if (keypoints3D && keypoints3D.data) {
            const data3D = keypoints3D.data;
            const step = Math.floor(data3D.length / 17);
            rawKeypoints = [];
            for (let k = 0; k < 17; k++) {
              const base = k * step;
              rawKeypoints.push({
                x: data3D[base] / (data3D[base] > 1.0 ? this.inputWidth : 1.0),
                y: data3D[base + 1] / (data3D[base + 1] > 1.0 ? this.inputHeight : 1.0),
                z: data3D[base + 2],
                hasZ: true,
                score: step >= 4 ? data3D[base + 3] : 0.9,
              });
            }
          } else if (simccX && simccY) {
            rawKeypoints = decodeSimCC3D(simccX.data, simccY.data, simccZ ? simccZ.data : null);
          }

          if (rawKeypoints) {
            const cocoKeypoints = this.smoothKeypoints(rawKeypoints, timestamp);
            this.latestResult = convertCocoToMediaPipeLandmarks(cocoKeypoints, aspect);
          }
        })
        .catch((err) => {
          console.warn("RTMPose 3D session.run error:", err);
        })
        .finally(() => {
          this.isInferencing = false;
        });
    }

    return this.latestResult;
  }

  close() {
    this.session?.release?.();
  }
}

/**
 * Unified Pose Tracker that coordinates:
 * 1. MediaPipe 3D Pose (Smooth, rock-solid, fully calibrated 3D world landmarks - Default)
 * 2. RTMW3D (3D) (rtmlib-ts RTMW3D-X 133-keypoint 3D Whole-Body Pose)
 * 3. RTMPose 3D (OpenMMLab SimCC-3D metric depth estimation)
 * 4. RTMPose 2D (OpenMMLab 2D SOTA with sub-pixel SimCC)
 */
export class UnifiedPoseTracker {
  static async create(options = {}) {
    const tracker = new UnifiedPoseTracker(options);
    await tracker.init();
    return tracker;
  }

  constructor(options = {}) {
    this.preferredEngine = options.preferredEngine || "mediapipe";
    this.activeEngine = "mediapipe";
    this.onStatus = options.onStatus || (() => {});
    this.bodyPartEngine = options.bodyPartEngine || null;
    this.rtmTracker = null;
    this.rtm3DTracker = null;
    this.rtmw3dTracker = null;
    this.mediaPipeTracker = null;
    this.lastResult = { landmarks: null, worldLandmarks: null };
    this.rtmErrorLogged = false;
  }

  setBodyPartEngine(engine) {
    this.bodyPartEngine = engine;
  }

  async init() {
    // 1. Always initialize MediaPipe Pose as rock-solid engine and reliable backup
    try {
      this.mediaPipeTracker = await MediaPipePoseTracker.create();
      this.activeEngine = "mediapipe";
      this.onStatus({
        state: "ready",
        engine: "mediapipe",
        message: "MediaPipe 3D Pose engine initialized (Smooth & Stable - Recommended)",
      });
    } catch (err) {
      console.warn("Could not initialize MediaPipe Pose backup:", err);
    }

    // 2. If a different engine is preferred, attempt activation
    if (this.preferredEngine === "bodypix") {
      await this.setEngine("bodypix");
    } else if (this.preferredEngine === "rtmw3d") {
      await this.setEngine("rtmw3d");
    } else if (this.preferredEngine === "rtmpose3d") {
      await this.setEngine("rtmpose3d");
    } else if (this.preferredEngine === "rtmpose") {
      await this.setEngine("rtmpose");
    }
  }

  async setEngine(engineName) {
    if (engineName === "bodypix") {
      this.activeEngine = "bodypix";
      if (this.bodyPartEngine) {
        if (!this.bodyPartEngine.ready && !this.bodyPartEngine.loading) {
          await this.bodyPartEngine.init();
        }
        this.bodyPartEngine.enabled = true;
      }
      this.onStatus({
        state: "ready",
        engine: "bodypix",
        message: "BodyPix engine active (TensorFlow.js WebGL GPU)",
      });
      return;
    } else if (engineName === "rtmw3d") {
      if (this.rtmw3dTracker) {
        this.activeEngine = "rtmw3d";
        this.onStatus({
          state: "ready",
          engine: "rtmw3d",
          backend: this.rtmw3dTracker.activeBackend,
          message: `Switched to RTMW3D (3D) engine (${this.rtmw3dTracker.activeBackend.toUpperCase()})`,
        });
        return;
      }
      try {
        this.rtmw3dTracker = await RTMW3DTracker.create({
          onStatus: this.onStatus,
          backend: "wasm",
        });
        this.activeEngine = "rtmw3d";
        this.onStatus({
          state: "ready",
          engine: "rtmw3d",
          backend: this.rtmw3dTracker.activeBackend,
          message: `RTMW3D (3D) engine active (${this.rtmw3dTracker.activeBackend.toUpperCase()})`,
        });
      } catch (err) {
        // Do not silently claim RTMW3D is running when it encountered an error
        this.onStatus({
          state: "error",
          engine: "rtmw3d",
          message: `RTMW3D initialization failed: ${err.message}`,
        });
      }
    } else if (engineName === "rtmpose3d") {
      if (this.rtm3DTracker) {
        this.activeEngine = "rtmpose3d";
        this.onStatus({
          state: "ready",
          engine: "rtmpose3d",
          message: "Switched to RTMPose 3D engine (OpenMMLab SimCC-3D)",
        });
        return;
      }
      try {
        this.rtm3DTracker = await RTMPose3DTracker.create();
        this.activeEngine = "rtmpose3d";
        this.onStatus({
          state: "ready",
          engine: "rtmpose3d",
          message: "RTMPose 3D engine active (OpenMMLab SimCC-3D)",
        });
      } catch (err) {
        this.activeEngine = "mediapipe";
        this.onStatus({
          state: "ready",
          engine: "mediapipe",
          message: `RTMPose 3D model not found (${err.message}). Active: MediaPipe 3D Pose. Drop rtmpose3d-m.onnx in frontend/public/models/ to activate.`,
        });
      }
    } else if (engineName === "rtmpose") {
      if (this.rtmTracker) {
        this.activeEngine = "rtmpose";
        this.onStatus({
          state: "ready",
          engine: "rtmpose",
          message: "Switched to RTMPose 2D engine (OpenMMLab)",
        });
        return;
      }
      try {
        this.rtmTracker = await RTMPoseTracker.create();
        this.activeEngine = "rtmpose";
        this.onStatus({
          state: "ready",
          engine: "rtmpose",
          message: "RTMPose 2D engine active (OpenMMLab SOTA)",
        });
      } catch (err) {
        this.activeEngine = "mediapipe";
        this.onStatus({
          state: "ready",
          engine: "mediapipe",
          message: `RTMPose model unavailable (${err.message}). Kept MediaPipe 3D Pose.`,
        });
      }
    } else {
      this.activeEngine = "mediapipe";
      this.onStatus({
        state: "ready",
        engine: "mediapipe",
        message: "Switched to MediaPipe 3D Pose engine (Smooth & Stable - Recommended)",
      });
    }
  }

  detect(video, timestamp) {
    if (this.activeEngine === "bodypix" && this.bodyPartEngine) {
      try {
        const result = this.bodyPartEngine.detect(video, timestamp);
        if (result && result.landmarks) {
          this.lastResult = result;
          return result;
        }
      } catch (err) {
        console.warn("BodyPix inference error:", err);
      }
      return this.lastResult;
    } else if (this.activeEngine === "rtmw3d" && this.rtmw3dTracker) {
      try {
        const result = this.rtmw3dTracker.detect(video, timestamp);
        if (result && result.landmarks) {
          this.lastResult = result;
          return result;
        }
      } catch (err) {
        console.warn("RTMW3D inference error:", err);
      }
      return this.lastResult;
    } else if (this.activeEngine === "rtmpose3d" && this.rtm3DTracker) {
      try {
        const result = this.rtm3DTracker.detect(video, timestamp);
        if (result && result.landmarks) {
          this.lastResult = result;
          return result;
        }
      } catch (err) {
        if (!this.rtmErrorLogged) {
          console.warn("RTMPose 3D inference error, falling back to MediaPipe:", err);
          this.rtmErrorLogged = true;
        }
      }
    } else if (this.activeEngine === "rtmpose" && this.rtmTracker) {
      try {
        const result = this.rtmTracker.detect(video, timestamp);
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

    if (this.mediaPipeTracker) {
      const result = this.mediaPipeTracker.detect(video, timestamp);
      this.lastResult = result;
      return result;
    }

    return this.lastResult;
  }

  close() {
    this.rtmw3dTracker?.close?.();
    this.rtm3DTracker?.close?.();
    this.rtmTracker?.close?.();
    this.mediaPipeTracker?.close?.();
  }
}
