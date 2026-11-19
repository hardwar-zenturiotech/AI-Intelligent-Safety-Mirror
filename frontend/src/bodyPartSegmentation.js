/**
 * BodyPix Body-Part Segmentation Engine for AI Intelligent Safety Mirror
 * Powered by @tensorflow-models/body-segmentation and TensorFlow.js WebGL backend.
 *
 * Provides:
 * 1. Independent 24 body-part semantic segmentation.
 * 2. Real-time colored segmentation debug overlay aligned with webcam feed.
 * 3. Multi-mode visualization (camera overlay, segmentation-only, individual limb masks).
 * 4. Front/back anatomical labeling preservation and limb grouping.
 * 5. Non-blocking asynchronous inference scheduling and FPS monitoring.
 * 6. Clean decoupled interface for MediaPipe and RTMW3D pose coordination.
 */

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import { load as loadBodyPix } from "@tensorflow-models/body-segmentation/dist/body_pix/segmenter.js";
import {
  convertCocoToMediaPipeLandmarks,
  OneEuroPointFilter,
} from "../rtmposeTracker.js";

export const SupportedModels = {
  BodyPix: "BodyPix",
};

export async function createSegmenter(model, config) {
  if (model === SupportedModels.BodyPix || model === "BodyPix") {
    return loadBodyPix(config);
  }
  throw new Error(`Unsupported model: ${model}`);
}

/**
 * Official 24 BodyPix Part IDs
 */
export const BODYPIX_PARTS = {
  LEFT_FACE: 0,
  RIGHT_FACE: 1,
  LEFT_UPPER_ARM_FRONT: 2,
  LEFT_UPPER_ARM_BACK: 3,
  RIGHT_UPPER_ARM_FRONT: 4,
  RIGHT_UPPER_ARM_BACK: 5,
  LEFT_LOWER_ARM_FRONT: 6,
  LEFT_LOWER_ARM_BACK: 7,
  RIGHT_LOWER_ARM_FRONT: 8,
  RIGHT_LOWER_ARM_BACK: 9,
  LEFT_HAND: 10,
  RIGHT_HAND: 11,
  TORSO_FRONT: 12,
  TORSO_BACK: 13,
  LEFT_UPPER_LEG_FRONT: 14,
  LEFT_UPPER_LEG_BACK: 15,
  RIGHT_UPPER_LEG_FRONT: 16,
  RIGHT_UPPER_LEG_BACK: 17,
  LEFT_LOWER_LEG_FRONT: 18,
  LEFT_LOWER_LEG_BACK: 19,
  RIGHT_LOWER_LEG_FRONT: 20,
  RIGHT_LOWER_LEG_BACK: 21,
  LEFT_FOOT: 22,
  RIGHT_FOOT: 23,
  BACKGROUND: -1,
};

/**
 * Official Part Names Array (index = ID)
 */
export const PART_NAMES = [
  "left_face",               // 0
  "right_face",              // 1
  "left_upper_arm_front",    // 2
  "left_upper_arm_back",     // 3
  "right_upper_arm_front",   // 4
  "right_upper_arm_back",    // 5
  "left_lower_arm_front",    // 6
  "left_lower_arm_back",     // 7
  "right_lower_arm_front",   // 8
  "right_lower_arm_back",    // 9
  "left_hand",               // 10
  "right_hand",              // 11
  "torso_front",             // 12
  "torso_back",              // 13
  "left_upper_leg_front",    // 14
  "left_upper_leg_back",     // 15
  "right_upper_leg_front",   // 16
  "right_upper_leg_back",    // 17
  "left_lower_leg_front",    // 18
  "left_lower_leg_back",     // 19
  "right_lower_leg_front",   // 20
  "right_lower_leg_back",    // 21
  "left_foot",               // 22
  "right_foot",              // 23
];

/**
 * Semantic Limb and Body Groups
 */
export const LIMB_GROUPS = {
  head: [0, 1],
  leftUpperArm: [2, 3],
  leftLowerArm: [6, 7],
  leftHand: [10],
  leftArm: [2, 3, 6, 7, 10],
  rightUpperArm: [4, 5],
  rightLowerArm: [8, 9],
  rightHand: [11],
  rightArm: [4, 5, 8, 9, 11],
  torso: [12, 13],
  leftLeg: [14, 15, 18, 19],
  rightLeg: [16, 17, 20, 21],
  feet: [22, 23],
};

/**
 * High-Contrast Color Palette for Visualizing Body Parts
 * Front and back parts share their limb color base for instant visual comprehension,
 * while maintaining subtle tone contrast and preserving raw IDs internally.
 */
export const DEFAULT_PART_COLORS = {
  // Head and Face (Golden Warm Coral)
  0: { r: 255, g: 214, b: 102 }, // left_face
  1: { r: 255, g: 195, b: 77 },  // right_face
  // Left Upper Arm (Vivid Magenta / Crimson)
  2: { r: 239, g: 71, b: 111 },  // left_upper_arm_front
  3: { r: 215, g: 50, b: 90 },   // left_upper_arm_back
  // Left Lower Arm (Salmon / Peach)
  6: { r: 247, g: 140, b: 107 }, // left_lower_arm_front
  7: { r: 230, g: 120, b: 90 },  // left_lower_arm_back
  // Left Hand (Bright Rose Red)
  10: { r: 255, g: 110, b: 140 }, // left_hand
  // Right Upper Arm (Deep Blue / Cyan)
  4: { r: 17, g: 138, b: 178 },  // right_upper_arm_front
  5: { r: 12, g: 115, b: 155 },  // right_upper_arm_back
  // Right Lower Arm (Sky Blue)
  8: { r: 78, g: 168, b: 222 },  // right_lower_arm_front
  9: { r: 60, g: 145, b: 200 },  // right_lower_arm_back
  // Right Hand (Bright Electric Cyan)
  11: { r: 0, g: 180, b: 216 },   // right_hand
  // Torso (Safety Emerald Neon)
  12: { r: 6, g: 214, b: 160 },  // torso_front
  13: { r: 4, g: 180, b: 135 },  // torso_back
  // Left Upper Leg (Chartreuse Lime)
  14: { r: 167, g: 201, b: 87 }, // left_upper_leg_front
  15: { r: 145, g: 180, b: 65 }, // left_upper_leg_back
  // Left Lower Leg (Yellow Green)
  18: { r: 186, g: 218, b: 85 }, // left_lower_leg_front
  19: { r: 165, g: 195, b: 70 }, // left_lower_leg_back
  // Right Upper Leg (Vibrant Leaf Green)
  16: { r: 56, g: 176, b: 0 },   // right_upper_leg_front
  17: { r: 45, g: 145, b: 0 },   // right_upper_leg_back
  // Right Lower Leg (Medium Spring Green)
  20: { r: 80, g: 195, b: 30 },  // right_lower_leg_front
  21: { r: 70, g: 170, b: 20 },  // right_lower_leg_back
  // Feet (Violet / Royal Purple)
  22: { r: 123, g: 44, b: 191 }, // left_foot
  23: { r: 90, g: 24, b: 154 },  // right_foot
};

export class BodyPartSegmentationEngine {
  constructor(options = {}) {
    this.options = {
      architecture: options.architecture || "MobileNetV1",
      outputStride: options.outputStride || 16,
      multiplier: options.multiplier || 0.75,
      quantBytes: options.quantBytes || 2,
      internalResolution: options.internalResolution || "low",
      segmentationThreshold: options.segmentationThreshold ?? 0.5,
      targetFps: options.targetFps || 60,
      debugMode: options.debugMode || "overlay", // "overlay" | "segmentation-only" | "parts" | "off"
      selectedPartGroup: options.selectedPartGroup || "all", // "all" | "head" | "torso" | "leftArm" | "rightArm" | "leftLeg" | "rightLeg"
      opacity: options.opacity ?? 0.65,
      ...options,
    };

    this.segmenter = null;
    this.loading = false;
    this.ready = false;
    this.error = null;
    this.isInferring = false;
    this.lastInferenceTime = 0;

    // Measurement & state
    this.latestResult = null;
    this.latestMaskData = null;
    this.latestRawPartData = null;
    this.latestPoseResult = { landmarks: null, worldLandmarks: null };
    this.latestBounds = null;
    this.outputWidth = 0;
    this.outputHeight = 0;
    this.peopleCount = 0;
    this.inferenceLatencyMs = 0;
    this.fps = 0;
    this.fpsFrameCount = 0;
    this.fpsStartTime = performance.now();
    this.statusListeners = new Set();
    this.enabled = true;

    // 1-Euro adaptive filters for smooth 33-point pose keypoints
    this.landmarkFilters = Array.from(
      { length: 33 },
      () => new OneEuroPointFilter(1.2, 0.007),
    );

    // Fast 32-bit LUT cache for zero-GC single-pass pixel rendering
    this._cachedLutKey = "";
    this._cachedLut32 = null;

    // Body orientation and semantic part metrics
    this.latestMetrics = null;
    this.smoothedFacingBias = 1.0; // 1.0 = full front, -1.0 = full rear

    // Internal offscreen buffer for compositing masks without GC churn
    if (typeof document !== "undefined") {
      this.renderCanvas = document.createElement("canvas");
      this.renderCtx = this.renderCanvas.getContext("2d", { willReadFrequently: false });
    } else {
      this.renderCanvas = null;
      this.renderCtx = null;
    }
    this.cachedImageData = null;
  }

  onStatus(callback) {
    this.statusListeners.add(callback);
    return () => this.statusListeners.delete(callback);
  }

  notifyStatus(status) {
    for (const listener of this.statusListeners) {
      try {
        listener(status);
      } catch (err) {
        console.error("BodyPartSegmentation status callback error:", err);
      }
    }
  }

  /**
   * Initializes the WebGL backend with hardware GPU performance flags
   * and loads the BodyPix model.
   */
  async init() {
    if (this.ready) return true;
    if (this.loading) return false;

    this.loading = true;
    this.error = null;
    this.notifyStatus({
      state: "loading",
      message: "Initializing TensorFlow WebGL runtime for BodyPix...",
    });

    try {
      // Configure high-performance WebGL GPU flags
      try {
        tf.env().set("WEBGL_PACK", true);
        tf.env().set("WEBGL_FORCE_F16_TEXTURES", true);
        tf.env().set("WEBGL_RENDER_FLOAT32_CAPABLE", true);
        tf.env().set("WEBGL_FLUSH_THRESHOLD", 1);
        tf.env().set("WEBGL_VERSION", 2);
      } catch (envErr) {
        console.warn("Could not set WebGL performance flags:", envErr);
      }

      // 1. Set WebGL backend (or available backend in headless test environments) and wait for TF readiness
      const hasWebGL = Boolean(tf.findBackend("webgl"));
      if (hasWebGL) {
        await tf.setBackend("webgl");
      } else if (tf.getBackend()) {
        await tf.setBackend(tf.getBackend());
      }
      await tf.ready();

      this.notifyStatus({
        state: "loading",
        message: `Loading BodyPix (${this.options.architecture}, stride ${this.options.outputStride})...`,
      });

      // 2. Create the BodyPix segmenter
      this.segmenter = await createSegmenter(
        SupportedModels.BodyPix,
        {
          architecture: this.options.architecture,
          outputStride: this.options.outputStride,
          multiplier: this.options.multiplier,
          quantBytes: this.options.quantBytes,
        },
      );

      this.ready = true;
      this.loading = false;
      this.notifyStatus({
        state: "ready",
        message: "BodyPix body-part segmentation engine ready (WebGL GPU Accelerated)",
        architecture: this.options.architecture,
      });
      return true;
    } catch (err) {
      this.ready = false;
      this.loading = false;
      this.error = err;
      console.warn("BodyPix body-part segmentation failed to initialize:", err);
      this.notifyStatus({
        state: "error",
        message: `BodyPix initialization error: ${err.message}`,
      });
      return false;
    }
  }

  /**
   * Performs real-time body-part segmentation & pose extraction on the active video frame.
   * Throttles inference according to targetFps and skips overlapping requests.
   *
   * @param {HTMLVideoElement} videoElement
   * @param {number} timestamp
   * @returns {Promise<Object|null>}
   */
  async segment(videoElement, timestamp = performance.now()) {
    if (
      !this.enabled ||
      !this.ready ||
      !this.segmenter ||
      this.isInferring ||
      !videoElement ||
      videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      return this.latestResult;
    }

    // Unthrottled when targetFps >= 60; otherwise enforce minimum frame interval
    if (this.options.targetFps < 60) {
      const minInterval = 1000 / this.options.targetFps;
      if (timestamp - this.lastInferenceTime < minInterval) {
        return this.latestResult;
      }
    }

    this.isInferring = true;
    this.lastInferenceTime = timestamp;
    const tStart = performance.now();

    try {
      let width = 0;
      let height = 0;
      let rawData = null;
      let allPoses = null;

      // Fast-path: invoke underlying bodyPixModel directly to skip intermediate JS allocations
      if (
        this.segmenter.bodyPixModel &&
        typeof this.segmenter.bodyPixModel.segmentPersonParts === "function"
      ) {
        const seg = await this.segmenter.bodyPixModel.segmentPersonParts(videoElement, {
          internalResolution: this.options.internalResolution,
          segmentationThreshold: this.options.segmentationThreshold,
          maxDetections: 1,
          scoreThreshold: 0.25,
          nmsRadius: 20,
        });
        height = seg.height;
        width = seg.width;
        rawData = seg.data; // Int32Array
        allPoses = seg.allPoses;
      } else {
        // Fallback to high-level segmentPeople API
        const people = await this.segmenter.segmentPeople(videoElement, {
          multiSegmentation: false,
          segmentBodyParts: true,
          flipHorizontal: false,
          internalResolution: this.options.internalResolution,
          segmentationThreshold: this.options.segmentationThreshold,
        });

        if (people && people.length > 0 && people[0].mask) {
          const mask = people[0].mask;
          const maskImageData = await mask.toImageData();
          width = maskImageData.width;
          height = maskImageData.height;
          rawData = maskImageData.data; // Uint8ClampedArray RGBA
          this.peopleCount = people.length;
        }
      }

      const tEnd = performance.now();
      this.inferenceLatencyMs = Math.round(tEnd - tStart);

      // Calculate FPS
      this.fpsFrameCount += 1;
      const elapsed = tEnd - this.fpsStartTime;
      if (elapsed >= 1000) {
        this.fps = Math.round((this.fpsFrameCount * 1000) / elapsed);
        this.fpsFrameCount = 0;
        this.fpsStartTime = tEnd;
      }

      if (rawData && width > 0 && height > 0) {
        this.outputWidth = width;
        this.outputHeight = height;
        this.latestRawPartData = rawData;
        this.latestMaskData = { width, height, data: rawData };
        this.peopleCount =
          allPoses && allPoses.length > 0
            ? allPoses.length
            : (rawData.some ? (rawData.some((p) => p >= 0) ? 1 : 0) : 1);

        // Extract COCO-17 pose keypoints and map to MediaPipe 33-point landmarks
        if (allPoses && allPoses.length > 0 && allPoses[0].keypoints) {
          const primaryPose = allPoses[0];
          const partMap = new Map();
          for (const kp of primaryPose.keypoints) {
            if (kp.part) partMap.set(kp.part.toLowerCase(), kp);
          }

          const COCO_PART_NAMES = [
            "nose", "lefteye", "righteye", "leftear", "rightear",
            "leftshoulder", "rightshoulder", "leftelbow", "rightelbow",
            "leftwrist", "rightwrist", "lefthip", "righthip",
            "leftknee", "rightknee", "leftankle", "rightankle"
          ];

          const cocoKeypoints = [];
          for (let i = 0; i < 17; i++) {
            const pName = COCO_PART_NAMES[i];
            const kp = partMap.get(pName) || primaryPose.keypoints[i];
            if (kp && kp.position) {
              cocoKeypoints.push({
                x: kp.position.x / width,
                y: kp.position.y / height,
                score: kp.score ?? 0.8,
              });
            } else {
              cocoKeypoints.push({ x: 0.5, y: 0.5, score: 0 });
            }
          }

          const mpPose = convertCocoToMediaPipeLandmarks(cocoKeypoints, width / height);
          const tsSec = timestamp * 0.001;
          const smoothedLandmarks = mpPose.landmarks.map((lm, idx) => {
            if (lm.visibility < 0.1) return lm;
            const filter = this.landmarkFilters[idx];
            const pt = filter ? filter.filter(lm.x, lm.y, tsSec) : { x: lm.x, y: lm.y };
            return { ...lm, x: pt.x, y: pt.y };
          });

          this.latestPoseResult = {
            landmarks: smoothedLandmarks,
            worldLandmarks: mpPose.worldLandmarks,
            fps: this.fps,
            latencyMs: this.inferenceLatencyMs,
            timestamp: tEnd,
          };
        }

        // Calculate silhouette bounding box for avatar scale validation and edge alignment
        this.latestBounds = this.computeBounds();
        // Compute 24-part semantic orientation, front/back distribution, and limb metrics
        this.latestMetrics = this.computeBodyPartMetrics();

        this.latestResult = {
          peopleCount: this.peopleCount,
          maskImageData: this.latestMaskData,
          width: this.outputWidth,
          height: this.outputHeight,
          bounds: this.latestBounds,
          metrics: this.latestMetrics,
          pose: this.latestPoseResult,
          fps: this.fps,
          latencyMs: this.inferenceLatencyMs,
          timestamp: tEnd,
        };
      } else {
        this.peopleCount = 0;
        this.latestMaskData = null;
        this.latestRawPartData = null;
        this.latestBounds = null;
        this.latestMetrics = null;
        this.latestResult = {
          peopleCount: 0,
          maskImageData: null,
          width: this.outputWidth,
          height: this.outputHeight,
          bounds: null,
          metrics: null,
          pose: { landmarks: null, worldLandmarks: null },
          fps: this.fps,
          latencyMs: this.inferenceLatencyMs,
          timestamp: tEnd,
        };
      }
    } catch (err) {
      console.warn("BodyPix segmentation inference error:", err);
    } finally {
      this.isInferring = false;
    }

    return this.latestResult;
  }

  /**
   * Synchronous/cached pose detection accessor matching UnifiedPoseTracker detect signature.
   * Dispatches asynchronous inference if idle and returns latest pose result.
   */
  detect(videoElement, timestamp = performance.now()) {
    if (this.ready && !this.isInferring && videoElement) {
      void this.segment(videoElement, timestamp);
    }
    return this.latestPoseResult;
  }

  getPoseLandmarks() {
    return this.latestPoseResult;
  }

  getSilhouetteBounds() {
    return this.latestBounds;
  }

  /**
   * Returns the latest body orientation metrics (front vs rear facing bias, view mode,
   * anatomical part counts and normalized bounds).
   */
  getBodyOrientationMetrics() {
    return this.latestMetrics;
  }

  getPartMetrics() {
    return this.latestMetrics;
  }

  /**
   * Computes comprehensive 24-part semantic metrics in a single pass:
   * - Front vs. Back torso distribution (facingBias: -1.0 rear to +1.0 front)
   * - View Mode: "front" | "back" | "profile-left" | "profile-right"
   * - Head, Torso, Hand bounding boxes and normalized centroids
   * - Arm dominance for turning angle detection
   */
  computeBodyPartMetrics() {
    const data = this.latestRawPartData;
    if (!data || !this.outputWidth || !this.outputHeight) return null;
    const w = this.outputWidth;
    const h = this.outputHeight;
    const isRgba = data.length === w * h * 4;
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    const counts = new Uint32Array(24);
    let headMinX = w, headMaxX = 0, headMinY = h, headMaxY = 0, headSumX = 0, headSumY = 0;
    let torsoMinX = w, torsoMaxX = 0, torsoMinY = h, torsoMaxY = 0, torsoSumX = 0, torsoSumY = 0;
    let lHandMinX = w, lHandMaxX = 0, lHandMinY = h, lHandMaxY = 0, lHandSumX = 0, lHandSumY = 0;
    let rHandMinX = w, rHandMaxX = 0, rHandMinY = h, rHandMaxY = 0, rHandSumX = 0, rHandSumY = 0;
    let totalPersonPixels = 0;

    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        let part = -1;
        if (isRgba) {
          const j = (rowOffset + x) * 4;
          if (data[j + 3] >= alphaCutoff) {
            part = data[j];
          }
        } else {
          part = data[rowOffset + x];
        }

        if (part >= 0 && part < 24) {
          counts[part]++;
          totalPersonPixels++;

          // Head (0: left_face, 1: right_face)
          if (part === 0 || part === 1) {
            if (x < headMinX) headMinX = x;
            if (x > headMaxX) headMaxX = x;
            if (y < headMinY) headMinY = y;
            if (y > headMaxY) headMaxY = y;
            headSumX += x;
            headSumY += y;
          }
          // Torso (12: torso_front, 13: torso_back)
          else if (part === 12 || part === 13) {
            if (x < torsoMinX) torsoMinX = x;
            if (x > torsoMaxX) torsoMaxX = x;
            if (y < torsoMinY) torsoMinY = y;
            if (y > torsoMaxY) torsoMaxY = y;
            torsoSumX += x;
            torsoSumY += y;
          }
          // Left Hand (10)
          else if (part === 10) {
            if (x < lHandMinX) lHandMinX = x;
            if (x > lHandMaxX) lHandMaxX = x;
            if (y < lHandMinY) lHandMinY = y;
            if (y > lHandMaxY) lHandMaxY = y;
            lHandSumX += x;
            lHandSumY += y;
          }
          // Right Hand (11)
          else if (part === 11) {
            if (x < rHandMinX) rHandMinX = x;
            if (x > rHandMaxX) rHandMaxX = x;
            if (y < rHandMinY) rHandMinY = y;
            if (y > rHandMaxY) rHandMaxY = y;
            rHandSumX += x;
            rHandSumY += y;
          }
        }
      }
    }

    if (totalPersonPixels === 0) return null;

    const torsoFront = counts[12];
    const torsoBack = counts[13];
    const totalTorso = torsoFront + torsoBack;
    const headPixels = counts[0] + counts[1];
    const leftArmPixels = counts[2] + counts[3] + counts[6] + counts[7] + counts[10];
    const rightArmPixels = counts[4] + counts[5] + counts[8] + counts[9] + counts[11];

    let rawFacingBias = 1.0;
    if (totalTorso > 10) {
      rawFacingBias = (torsoFront - torsoBack) / totalTorso;
    } else if (headPixels > 10) {
      // If torso is occluded/missing but face is detected, infer from face parts
      rawFacingBias = 1.0;
    }

    // Exponential smoothing on facing bias to avoid flutter during turns
    this.smoothedFacingBias = this.smoothedFacingBias * 0.70 + rawFacingBias * 0.30;
    const facingBias = this.smoothedFacingBias;

    let viewMode = "front";
    if (totalTorso > 10) {
      if (facingBias > 0.22) {
        viewMode = "front";
      } else if (facingBias < -0.22) {
        viewMode = "back";
      } else {
        // In side profile: determine whether turning left or right
        viewMode = leftArmPixels > rightArmPixels ? "profile-left" : "profile-right";
      }
    }

    const headBounds = headPixels > 0 && headMaxX >= headMinX ? {
      minX: headMinX,
      maxX: headMaxX,
      minY: headMinY,
      maxY: headMaxY,
      width: headMaxX - headMinX,
      height: headMaxY - headMinY,
      normWidth: (headMaxX - headMinX) / w,
      normHeight: (headMaxY - headMinY) / h,
      centroid: { x: headSumX / (headPixels * w), y: headSumY / (headPixels * h) },
    } : null;

    const torsoBounds = totalTorso > 0 && torsoMaxX >= torsoMinX ? {
      minX: torsoMinX,
      maxX: torsoMaxX,
      minY: torsoMinY,
      maxY: torsoMaxY,
      width: torsoMaxX - torsoMinX,
      height: torsoMaxY - torsoMinY,
      normWidth: (torsoMaxX - torsoMinX) / w,
      normHeight: (torsoMaxY - torsoMinY) / h,
      centroid: { x: torsoSumX / (totalTorso * w), y: torsoSumY / (totalTorso * h) },
    } : null;

    const leftHandCentroid = counts[10] > 0 ? {
      x: lHandSumX / (counts[10] * w),
      y: lHandSumY / (counts[10] * h),
      pixels: counts[10],
    } : null;

    const rightHandCentroid = counts[11] > 0 ? {
      x: rHandSumX / (counts[11] * w),
      y: rHandSumY / (counts[11] * h),
      pixels: counts[11],
    } : null;

    return {
      facingBias,
      rawFacingBias,
      viewMode,
      isRearView: facingBias < -0.22,
      isFrontView: facingBias > 0.22,
      isProfileView: Math.abs(facingBias) <= 0.22,
      torsoFrontPixels: torsoFront,
      torsoBackPixels: torsoBack,
      totalTorsoPixels: totalTorso,
      headPixels,
      leftArmPixels,
      rightArmPixels,
      headBounds,
      torsoBounds,
      leftHandCentroid,
      rightHandCentroid,
      totalPersonPixels,
      confidence: totalTorso > 35 ? 1.0 : (totalPersonPixels > 100 ? 0.7 : 0.0),
    };
  }

  /**
   * Computes the bounding rectangle of the segmented person.
   */
  computeBounds() {
    const data = this.latestRawPartData;
    if (!data || !this.outputWidth || !this.outputHeight) return null;
    const w = this.outputWidth;
    const h = this.outputHeight;
    const isRgba = data.length === w * h * 4;
    let minX = w;
    let maxX = 0;
    let minY = h;
    let maxY = 0;
    let found = false;

    if (isRgba) {
      const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const j = (y * w + x) * 4;
          if (data[j + 3] >= alphaCutoff && data[j] >= 0 && data[j] <= 23) {
            found = true;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }
    } else {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const part = data[y * w + x];
          if (part >= 0 && part <= 23) {
            found = true;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }
    }

    if (!found) return null;
    return {
      minX,
      maxX,
      minY,
      maxY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  /**
   * Precomputes or returns the cached 32-bit little-endian color lookup table.
   * Format: (Alpha << 24) | (Blue << 16) | (Green << 8) | Red
   */
  getOrCreateLUT32(mode, selectedGroup, opacity) {
    const key = `${mode}_${selectedGroup}_${opacity}`;
    if (this._cachedLutKey === key && this._cachedLut32) {
      return this._cachedLut32;
    }

    const lut = new Uint32Array(24);
    const alpha = mode === "segmentation-only" ? 255 : Math.round(255 * opacity);
    const targetPartIds =
      selectedGroup !== "all" && LIMB_GROUPS[selectedGroup]
        ? new Set(LIMB_GROUPS[selectedGroup])
        : null;

    for (let id = 0; id < 24; id++) {
      const col = DEFAULT_PART_COLORS[id] || { r: 255, g: 255, b: 255 };
      if (targetPartIds) {
        if (targetPartIds.has(id)) {
          lut[id] = (alpha << 24) | (col.b << 16) | (col.g << 8) | col.r;
        } else {
          if (mode === "segmentation-only") {
            lut[id] = (255 << 24) | (25 << 16) | (20 << 8) | 20;
          } else {
            lut[id] = (40 << 24) | (50 << 16) | (50 << 8) | 50;
          }
        }
      } else {
        lut[id] = (alpha << 24) | (col.b << 16) | (col.g << 8) | col.r;
      }
    }

    this._cachedLutKey = key;
    this._cachedLut32 = lut;
    return lut;
  }

  /**
   * Renders the colored body-part segmentation overlay using high-speed 32-bit LUT memory stores.
   * Matches video aspect ratio and coordinates cleanly.
   *
   * @param {HTMLCanvasElement} targetCanvas
   * @param {HTMLVideoElement} videoElement
   */
  renderOverlay(targetCanvas, videoElement) {
    if (!targetCanvas || !this.enabled || this.options.debugMode === "off") {
      if (targetCanvas) {
        const ctx = targetCanvas.getContext("2d");
        ctx?.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
      }
      return;
    }

    const maskData = this.latestMaskData;
    if (!maskData) {
      const ctx = targetCanvas.getContext("2d");
      ctx?.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
      return;
    }

    const w = maskData.width;
    const h = maskData.height;

    // Ensure offscreen cache is allocated to match mask resolution
    if (
      !this.renderCanvas ||
      this.renderCanvas.width !== w ||
      this.renderCanvas.height !== h
    ) {
      this.renderCanvas = document.createElement("canvas");
      this.renderCanvas.width = w;
      this.renderCanvas.height = h;
      this.renderCtx = this.renderCanvas.getContext("2d");
      this.cachedImageData = this.renderCtx.createImageData(w, h);
    }

    const imgData = this.cachedImageData;
    const rawMask = maskData.data;
    const out32 = new Uint32Array(imgData.data.buffer);
    const mode = this.options.debugMode;
    const isRgba = rawMask.length === w * h * 4;
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    const lut = this.getOrCreateLUT32(mode, this.options.selectedPartGroup, this.options.opacity);
    const bgWord = mode === "segmentation-only" ? ((255 << 24) | (18 << 16) | (15 << 8) | 12) : 0;
    const totalPixels = w * h;

    if (isRgba) {
      for (let i = 0; i < totalPixels; i++) {
        const j = i * 4;
        const part = rawMask[j];
        const alpha = rawMask[j + 3];
        if (alpha >= alphaCutoff && part >= 0 && part < 24) {
          out32[i] = lut[part];
        } else {
          out32[i] = bgWord;
        }
      }
    } else {
      for (let i = 0; i < totalPixels; i++) {
        const part = rawMask[i];
        if (part >= 0 && part < 24) {
          out32[i] = lut[part];
        } else {
          out32[i] = bgWord;
        }
      }
    }

    this.renderCtx.putImageData(imgData, 0, 0);

    const ctx = targetCanvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    ctx.drawImage(this.renderCanvas, 0, 0, targetCanvas.width, targetCanvas.height);
  }

  /**
   * Returns a 1-channel Uint8Array binary mask for a designated body part.
   * 1 = part pixel, 0 = outside.
   *
   * @param {number|string} partIdOrName
   * @returns {Uint8Array|null}
   */
  getPartMask(partIdOrName) {
    if (!this.latestMaskData) return null;
    const targetId =
      typeof partIdOrName === "string"
        ? PART_NAMES.indexOf(partIdOrName)
        : partIdOrName;
    if (targetId < 0 || targetId > 23) return null;

    const { width, height, data } = this.latestMaskData;
    const binary = new Uint8Array(width * height);
    const isRgba = data.length === width * height * 4;
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    for (let i = 0; i < binary.length; i++) {
      const part = isRgba ? data[i * 4] : data[i];
      const alpha = isRgba ? data[i * 4 + 3] : (part >= 0 ? 255 : 0);
      if (alpha >= alphaCutoff && part === targetId) {
        binary[i] = 1;
      }
    }
    return binary;
  }

  /**
   * Returns an aggregated 1-channel binary mask for a designated limb group.
   *
   * @param {"head"|"leftArm"|"rightArm"|"torso"|"leftLeg"|"rightLeg"|"feet"} limbName
   * @returns {Uint8Array|null}
   */
  getLimbMask(limbName) {
    if (!this.latestMaskData || !LIMB_GROUPS[limbName]) return null;
    const partSet = new Set(LIMB_GROUPS[limbName]);
    const { width, height, data } = this.latestMaskData;
    const binary = new Uint8Array(width * height);
    const isRgba = data.length === width * height * 4;
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    for (let i = 0; i < binary.length; i++) {
      const part = isRgba ? data[i * 4] : data[i];
      const alpha = isRgba ? data[i * 4 + 3] : (part >= 0 ? 255 : 0);
      if (alpha >= alphaCutoff && partSet.has(part)) {
        binary[i] = 1;
      }
    }
    return binary;
  }

  /**
   * Returns all limb masks organized as an object for modular consumption.
   */
  getMasksByLimb() {
    return {
      head: this.getLimbMask("head"),
      torso: this.getLimbMask("torso"),
      leftArm: this.getLimbMask("leftArm"),
      rightArm: this.getLimbMask("rightArm"),
      leftLeg: this.getLimbMask("leftLeg"),
      rightLeg: this.getLimbMask("rightLeg"),
      feet: this.getLimbMask("feet"),
    };
  }

  /**
   * Queries which body part label exists at normalized camera image coordinates (0.0 to 1.0).
   *
   * @param {number} normX
   * @param {number} normY
   * @returns {{ id: number, name: string, isPerson: boolean }}
   */
  getPartAtNormalizedCoord(normX, normY) {
    if (!this.latestMaskData) {
      return { id: -1, name: "background", isPerson: false };
    }
    const { width, height, data } = this.latestMaskData;
    const px = Math.min(Math.max(Math.floor(normX * width), 0), width - 1);
    const py = Math.min(Math.max(Math.floor(normY * height), 0), height - 1);
    const isRgba = data.length === width * height * 4;
    const idx = isRgba ? (py * width + px) * 4 : (py * width + px);
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    if (isRgba) {
      if (data[idx + 3] >= alphaCutoff) {
        const id = data[idx];
        return {
          id,
          name: PART_NAMES[id] || "unknown",
          isPerson: true,
        };
      }
    } else {
      const id = data[idx];
      if (id >= 0 && id <= 23) {
        return {
          id,
          name: PART_NAMES[id] || "unknown",
          isPerson: true,
        };
      }
    }
    return { id: -1, name: "background", isPerson: false };
  }

  setDebugMode(mode) {
    this.options.debugMode = mode;
  }

  setSelectedPartGroup(group) {
    this.options.selectedPartGroup = group;
  }

  setTargetFps(fps) {
    this.options.targetFps = Math.max(1, Math.min(fps, 60));
  }

  setInternalResolution(resolution) {
    const valid = ["low", "medium", "high", "full"];
    if (valid.includes(resolution)) {
      this.options.internalResolution = resolution;
    }
  }

  setOpacity(opacity) {
    this.options.opacity = Math.max(0, Math.min(opacity, 1));
  }

  close() {
    this.enabled = false;
    this.segmenter?.dispose?.();
    this.segmenter = null;
    this.ready = false;
    this.statusListeners.clear();
  }
}
