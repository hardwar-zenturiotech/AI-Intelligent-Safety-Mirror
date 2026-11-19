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
import { load as loadBodyPix } from "@tensorflow-models/body-segmentation/dist/body_pix/segmenter.js";

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
      internalResolution: options.internalResolution || "medium",
      segmentationThreshold: options.segmentationThreshold ?? 0.5,
      targetFps: options.targetFps || 15,
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
    this.outputWidth = 0;
    this.outputHeight = 0;
    this.peopleCount = 0;
    this.inferenceLatencyMs = 0;
    this.fps = 0;
    this.fpsFrameCount = 0;
    this.fpsStartTime = performance.now();
    this.statusListeners = new Set();
    this.enabled = true;

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
   * Initializes the WebGL backend and loads the BodyPix model.
   * Handles initialization failures gracefully without crashing other systems.
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
      // 1. Set WebGL backend and wait for TF readiness
      await tf.setBackend("webgl");
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
        message: "BodyPix body-part segmentation engine ready (WebGL)",
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
   * Performs real-time body-part segmentation on the active video frame.
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

    // Throttle inference frequency independently from rendering rate
    const minInterval = 1000 / this.options.targetFps;
    if (timestamp - this.lastInferenceTime < minInterval) {
      return this.latestResult;
    }

    this.isInferring = true;
    this.lastInferenceTime = timestamp;
    const tStart = performance.now();

    try {
      // Segment body parts with single-person mode
      const people = await this.segmenter.segmentPeople(videoElement, {
        multiSegmentation: false,
        segmentBodyParts: true,
        flipHorizontal: false,
        internalResolution: this.options.internalResolution,
        segmentationThreshold: this.options.segmentationThreshold,
      });

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

      if (people && people.length > 0 && people[0].mask) {
        const mask = people[0].mask;
        const maskImageData = await mask.toImageData();

        this.latestMaskData = maskImageData;
        this.outputWidth = maskImageData.width;
        this.outputHeight = maskImageData.height;
        this.peopleCount = people.length;

        this.latestResult = {
          peopleCount: this.peopleCount,
          maskImageData,
          width: this.outputWidth,
          height: this.outputHeight,
          fps: this.fps,
          latencyMs: this.inferenceLatencyMs,
          timestamp: tEnd,
        };
      } else {
        this.peopleCount = 0;
        this.latestMaskData = null;
        this.latestResult = {
          peopleCount: 0,
          maskImageData: null,
          width: this.outputWidth,
          height: this.outputHeight,
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
   * Renders the colored body-part segmentation overlay to a destination canvas.
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
    const rawMask = maskData.data; // RGBA buffer from toImageData()
    const outPixels = imgData.data;
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);
    const mode = this.options.debugMode; // "overlay" | "segmentation-only" | "parts"
    const opacityAlpha = Math.round(255 * this.options.opacity);
    const selectedGroup = this.options.selectedPartGroup;
    const targetPartIds =
      selectedGroup !== "all" && LIMB_GROUPS[selectedGroup]
        ? new Set(LIMB_GROUPS[selectedGroup])
        : null;

    const totalPixels = w * h;
    for (let i = 0; i < totalPixels; i++) {
      const j = i * 4;
      const maskVal = rawMask[j];
      const personAlpha = rawMask[j + 3];

      if (personAlpha >= alphaCutoff && maskVal >= 0 && maskVal <= 23) {
        const color = DEFAULT_PART_COLORS[maskVal] || { r: 255, g: 255, b: 255 };

        if (targetPartIds) {
          // Individual parts filtering mode
          if (targetPartIds.has(maskVal)) {
            outPixels[j] = color.r;
            outPixels[j + 1] = color.g;
            outPixels[j + 2] = color.b;
            outPixels[j + 3] = mode === "segmentation-only" ? 255 : opacityAlpha;
          } else {
            // Demote unselected parts to faint ghost/dark
            if (mode === "segmentation-only") {
              outPixels[j] = 20;
              outPixels[j + 1] = 20;
              outPixels[j + 2] = 25;
              outPixels[j + 3] = 255;
            } else {
              outPixels[j] = 50;
              outPixels[j + 1] = 50;
              outPixels[j + 2] = 50;
              outPixels[j + 3] = 40;
            }
          }
        } else {
          // Full 24-part segmentation display
          outPixels[j] = color.r;
          outPixels[j + 1] = color.g;
          outPixels[j + 2] = color.b;
          outPixels[j + 3] = mode === "segmentation-only" ? 255 : opacityAlpha;
        }
      } else {
        // Background
        if (mode === "segmentation-only") {
          outPixels[j] = 12;
          outPixels[j + 1] = 15;
          outPixels[j + 2] = 18;
          outPixels[j + 3] = 255;
        } else {
          outPixels[j] = 0;
          outPixels[j + 1] = 0;
          outPixels[j + 2] = 0;
          outPixels[j + 3] = 0;
        }
      }
    }

    this.renderCtx.putImageData(imgData, 0, 0);

    // Draw composited mask onto destination targetCanvas
    const targetCtx = targetCanvas.getContext("2d");
    if (!targetCtx) return;

    targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    targetCtx.imageSmoothingEnabled = true;
    targetCtx.drawImage(
      this.renderCanvas,
      0,
      0,
      targetCanvas.width,
      targetCanvas.height,
    );
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
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    for (let i = 0; i < binary.length; i++) {
      const j = i * 4;
      if (data[j + 3] >= alphaCutoff && data[j] === targetId) {
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
    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);

    for (let i = 0; i < binary.length; i++) {
      const j = i * 4;
      if (data[j + 3] >= alphaCutoff && partSet.has(data[j])) {
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
    const idx = (py * width + px) * 4;

    const alphaCutoff = Math.round(255 * this.options.segmentationThreshold);
    if (data[idx + 3] >= alphaCutoff) {
      const id = data[idx];
      return {
        id,
        name: PART_NAMES[id] || "unknown",
        isPerson: true,
      };
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
