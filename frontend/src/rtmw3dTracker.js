/**
 * RTMW3D (3D Pose Tracking) Engine for AI Intelligent Safety Mirror
 * Powered by rtmlib-ts and ONNX Runtime Web.
 *
 * Implements:
 * - Real-time 3D pose estimation with RTMW3D-X (COCO-WholeBody 133-keypoints)
 * - Three.js Diagnostic 3D Skeleton
 * - 2D camera-image overlay using keypoints2d
 * - Adapter converting root-relative 3D metric coordinates into retargeting-compatible format
 * - WebGPU support with graceful WASM baseline fallback
 */

import { Pose3DDetector } from "rtmlib-ts";
import * as THREE from "three";

export const COCO17_KEYPOINTS = {
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

// 17 COCO keypoint IDs -> 33 MediaPipe landmark IDs
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

// COCO17 bone connections for 2D and 3D skeleton visualization
export const COCO17_CONNECTIONS = [
  [5, 6],   // left shoulder to right shoulder
  [11, 12], // left hip to right hip
  [5, 11],  // left shoulder to left hip (torso lateral)
  [6, 12],  // right shoulder to right hip (torso lateral)
  [5, 7],   // left upper arm
  [7, 9],   // left forearm
  [6, 8],   // right upper arm
  [8, 10],  // right forearm
  [11, 13], // left thigh
  [13, 15], // left shin
  [12, 14], // right thigh
  [14, 16], // right shin
  [0, 1],   // nose to left eye
  [0, 2],   // nose to right eye
  [1, 3],   // left eye to left ear
  [2, 4],   // right eye to right ear
  [0, 5],   // nose to left shoulder
  [0, 6],   // nose to right shoulder
];

/**
 * Checks whether WebGPU is genuinely supported in the current browser.
 */
export async function isWebGPUSupported() {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

/**
 * Three.js Diagnostic 3D Skeleton
 * Renders the detected 3D keypoints and bone connections directly in 3D scene space.
 * Joints are color-coded by prediction confidence.
 */
export class RTMW3DDiagnosticSkeleton {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = "RTMW3D_Diagnostic_Skeleton";

    // 17 joint spheres
    this.jointMeshes = [];
    const sphereGeo = new THREE.SphereGeometry(0.024, 12, 12);
    for (let i = 0; i < 17; i++) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x00ff88,
        wireframe: false,
        depthTest: false,
      });
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.renderOrder = 999;
      mesh.visible = false;
      this.group.add(mesh);
      this.jointMeshes.push(mesh);
    }

    // Line segments connecting bones
    const numLines = COCO17_CONNECTIONS.length;
    const linePositions = new Float32Array(numLines * 2 * 3);
    const lineColors = new Float32Array(numLines * 2 * 3);
    this.lineGeometry = new THREE.BufferAttribute(linePositions, 3);
    this.lineColorAttribute = new THREE.BufferAttribute(lineColors, 3);

    const bufferGeo = new THREE.BufferGeometry();
    bufferGeo.setAttribute("position", this.lineGeometry);
    bufferGeo.setAttribute("color", this.lineColorAttribute);

    const lineMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      linewidth: 3,
      depthTest: false,
      transparent: true,
      opacity: 0.85,
    });
    this.lineSegments = new THREE.LineSegments(bufferGeo, lineMat);
    this.lineSegments.renderOrder = 998;
    this.lineSegments.visible = false;
    this.group.add(this.lineSegments);

    // Root offset in 3D space: placed slightly in front or to the side for clear inspection
    this.offset = new THREE.Vector3(0, 0.95, 0);
    this.scale = 1.0;
  }

  update(keypoints3D, scores) {
    if (!keypoints3D || keypoints3D.length < 17) {
      this.setVisible(false);
      return;
    }

    this.setVisible(true);
    const posArr = this.lineGeometry.array;
    const colArr = this.lineColorAttribute.array;

    // 1. Update joints
    for (let i = 0; i < 17; i++) {
      const kp = keypoints3D[i];
      const score = scores ? (scores[i] ?? 1.0) : 1.0;
      const mesh = this.jointMeshes[i];

      if (!kp || score < 0.20) {
        mesh.visible = false;
        continue;
      }

      mesh.visible = true;

      // In RTMW3D:
      // x: lateral (right positive)
      // y: vertical (downward positive)
      // z: depth in meters (away positive)
      // In Three.js: +X right, +Y up, +Z towards camera
      const tx = kp[0] * this.scale + this.offset.x;
      const ty = -kp[1] * this.scale + this.offset.y;
      const tz = -kp[2] * this.scale + this.offset.z;

      mesh.position.set(tx, ty, tz);

      // Color-code by confidence
      if (score >= 0.65) {
        mesh.material.color.setHex(0x00ff88); // High confidence: emerald green
      } else if (score >= 0.35) {
        mesh.material.color.setHex(0xffcc00); // Medium confidence: amber
      } else {
        mesh.material.color.setHex(0xff4444); // Low confidence: red
      }
    }

    // 2. Update bone lines
    for (let c = 0; c < COCO17_CONNECTIONS.length; c++) {
      const [from, to] = COCO17_CONNECTIONS[c];
      const kpA = keypoints3D[from];
      const kpB = keypoints3D[to];
      const scA = scores ? (scores[from] ?? 1.0) : 1.0;
      const scB = scores ? (scores[to] ?? 1.0) : 1.0;

      const idx = c * 6;
      if (!kpA || !kpB || scA < 0.20 || scB < 0.20) {
        posArr[idx] = 0; posArr[idx + 1] = 0; posArr[idx + 2] = 0;
        posArr[idx + 3] = 0; posArr[idx + 4] = 0; posArr[idx + 5] = 0;
        continue;
      }

      const ax = kpA[0] * this.scale + this.offset.x;
      const ay = -kpA[1] * this.scale + this.offset.y;
      const az = -kpA[2] * this.scale + this.offset.z;

      const bx = kpB[0] * this.scale + this.offset.x;
      const by = -kpB[1] * this.scale + this.offset.y;
      const bz = -kpB[2] * this.scale + this.offset.z;

      posArr[idx] = ax;
      posArr[idx + 1] = ay;
      posArr[idx + 2] = az;
      posArr[idx + 3] = bx;
      posArr[idx + 4] = by;
      posArr[idx + 5] = bz;

      // Color segment
      const r = Math.min(scA, scB) >= 0.5 ? 0.2 : 0.9;
      const g = Math.min(scA, scB) >= 0.5 ? 0.9 : 0.6;
      const b = 0.4;
      colArr[idx] = r; colArr[idx + 1] = g; colArr[idx + 2] = b;
      colArr[idx + 3] = r; colArr[idx + 4] = g; colArr[idx + 5] = b;
    }

    this.lineGeometry.needsUpdate = true;
    this.lineColorAttribute.needsUpdate = true;
    this.lineSegments.visible = true;
  }

  setVisible(visible) {
    this.group.visible = visible;
    if (!visible) {
      for (const mesh of this.jointMeshes) mesh.visible = false;
      if (this.lineSegments) this.lineSegments.visible = false;
    }
  }

  dispose() {
    for (const mesh of this.jointMeshes) {
      mesh.geometry?.dispose();
      mesh.material?.dispose();
    }
    this.lineSegments?.geometry?.dispose();
    this.lineSegments?.material?.dispose();
    if (this.group.parent) {
      this.group.parent.remove(this.group);
    }
  }
}

/**
 * Converts RTMW3D multi-person 3D output into the internal pose format
 * expected by retargeting.js, avatar.js, and main.js.
 */
export function convertRtmw3dToMediaPipe(result, videoWidth, videoHeight) {
  if (
    !result ||
    !result.keypoints ||
    !result.keypoints.length ||
    !result.keypoints[0] ||
    !result.keypoints[0].length
  ) {
    return {
      landmarks: null,
      worldLandmarks: null,
      keypoints3D: null,
      keypoints2D: null,
      scores: null,
      personCount: 0,
      stats: result?.stats || null,
    };
  }

  // Use primary (first) detected person
  const personK3D = result.keypoints[0]; // [133][3] in meters
  const personScores = result.scores ? result.scores[0] : []; // [133]
  const personK2D = result.keypoints2d ? result.keypoints2d[0] : []; // [133][2] in source pixels

  const vw = Math.max(videoWidth || 640, 1);
  const vh = Math.max(videoHeight || 480, 1);

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

  // Determine root in 2D image coordinates (midpoint of hips or shoulders)
  const lHip2D = personK2D[11];
  const rHip2D = personK2D[12];
  const lSh2D = personK2D[5];
  const rSh2D = personK2D[6];

  let refX = vw * 0.5;
  let refY = vh * 0.5;
  if (lHip2D && rHip2D && (personScores[11] || 0) > 0.2 && (personScores[12] || 0) > 0.2) {
    refX = (lHip2D[0] + rHip2D[0]) * 0.5;
    refY = (lHip2D[1] + rHip2D[1]) * 0.5;
  } else if (lSh2D && rSh2D && (personScores[5] || 0) > 0.2 && (personScores[6] || 0) > 0.2) {
    refX = (lSh2D[0] + rSh2D[0]) * 0.5;
    refY = (lSh2D[1] + rSh2D[1]) * 0.5;
  }

  // Pixel to meter scale based on shoulder span
  let pixelsToMeters = 0.40 / 160.0;
  if (lSh2D && rSh2D && (personScores[5] || 0) > 0.2 && (personScores[6] || 0) > 0.2) {
    const shSpanPx = Math.hypot(lSh2D[0] - rSh2D[0], lSh2D[1] - rSh2D[1]);
    if (shSpanPx > 20) {
      pixelsToMeters = 0.40 / shSpanPx;
    }
  }

  // Map COCO17 keypoints to MediaPipe 33 landmark slots
  for (const [cocoIdx, mpIdx] of COCO_TO_MEDIAPIPE) {
    const k3 = personK3D[cocoIdx];
    const k2 = personK2D[cocoIdx];
    const score = personScores[cocoIdx] ?? 0.0;

    if (!k3) continue;

    // 2D Screen-space landmark normalized to [0, 1]
    const px = k2 ? k2[0] : (k3[0] + refX);
    const py = k2 ? k2[1] : (k3[1] + refY);
    const normX = Math.min(Math.max(px / vw, 0), 1);
    const normY = Math.min(Math.max(py / vh, 0), 1);
    const metricZ = k3[2] ?? 0.0;

    landmarks[mpIdx] = {
      x: normX,
      y: normY,
      z: metricZ,
      visibility: score,
      presence: score,
    };

    // 3D Metric World Landmark
    // X: right positive, Y: downward positive, Z: depth away from camera positive
    let wx = 0;
    let wy = 0;
    if (k2) {
      wx = (k2[0] - refX) * pixelsToMeters;
      wy = (k2[1] - refY) * pixelsToMeters;
    } else {
      wx = k3[0];
      wy = k3[1];
    }
    const wz = metricZ;

    worldLandmarks[mpIdx] = {
      x: wx,
      y: wy,
      z: wz,
      visibility: score,
      presence: score,
    };
  }

  return {
    landmarks,
    worldLandmarks,
    keypoints3D: personK3D,
    keypoints2D: personK2D,
    scores: personScores,
    personCount: result.keypoints.length,
    stats: result.stats || null,
  };
}

/**
 * RTMW3DTracker
 * Coordinates the rtmlib-ts Pose3DDetector pipeline, diagnostic visualizers,
 * and retargeting adapter with seamless status reporting.
 */
export class RTMW3DTracker {
  static async create(options = {}) {
    const tracker = new RTMW3DTracker(options);
    await tracker.init();
    return tracker;
  }

  constructor(options = {}) {
    this.options = options;
    this.onStatus = options.onStatus || (() => {});
    this.requestedBackend = options.backend || "wasm";
    this.activeBackend = "wasm";
    this.detector = null;
    this.isInferencing = false;
    this.latestResult = {
      landmarks: null,
      worldLandmarks: null,
      keypoints3D: null,
      keypoints2D: null,
      scores: null,
      personCount: 0,
      stats: null,
      inferenceTimeMs: 0,
      inferenceFps: 0,
      backend: "wasm",
    };
    this.fpsFrameCount = 0;
    this.fpsStartTime = performance.now();
    this.diagnosticSkeleton = new RTMW3DDiagnosticSkeleton();
    this.diagnosticAttached = false;
  }

  async init() {
    this.onStatus({
      state: "loading",
      engine: "rtmw3d",
      message: "Initialising person detector...",
    });

    // Check if WebGPU was requested and is genuinely available
    let useWebGPU = false;
    if (this.requestedBackend === "webgpu") {
      const gpuAvailable = await isWebGPUSupported();
      if (gpuAvailable) {
        useWebGPU = true;
      } else {
        console.warn("[RTMW3D] WebGPU not supported on this browser; falling back to WASM.");
      }
    }

    // Check if local bundled/cached RTMW3D ONNX model is available
    const localModelPath = "/models/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx";
    let poseModelUrl = undefined;
    if (typeof window !== "undefined") {
      try {
        const res = await fetch(localModelPath, { method: "HEAD" });
        if (res.ok) {
          poseModelUrl = localModelPath;
          console.log("[RTMW3D] Found local ONNX model at", localModelPath);
        }
      } catch {}
    }

    const config = useWebGPU
      ? {
          objectModel: "yolov8n",
          pose3dModel: "rtmw3d",
          backend: "webgpu",
          profile: true,
          ...(poseModelUrl ? { poseModel: poseModelUrl } : {}),
          onInitProgress: (stage, detail) => {
            this.handleInitProgress(stage, detail);
          },
        }
      : {
          objectModel: "mediapipe",
          pose3dModel: "rtmw3d",
          backend: "wasm",
          profile: true,
          ...(poseModelUrl ? { poseModel: poseModelUrl } : {}),
          onInitProgress: (stage, detail) => {
            this.handleInitProgress(stage, detail);
          },
        };

    try {
      this.detector = new Pose3DDetector(config);
      await this.detector.init();
      this.activeBackend = useWebGPU ? "webgpu" : "wasm";
      this.onStatus({
        state: "ready",
        engine: "rtmw3d",
        backend: this.activeBackend,
        message: `RTMW3D-X model ready (Backend: ${this.activeBackend.toUpperCase()})`,
      });
    } catch (err) {
      if (useWebGPU) {
        console.warn("[RTMW3D] WebGPU initialization failed, attempting WASM fallback...", err);
        this.onStatus({
          state: "loading",
          engine: "rtmw3d",
          message: "WebGPU failed, falling back to WASM baseline...",
        });
        const wasmConfig = {
          objectModel: "mediapipe",
          pose3dModel: "rtmw3d",
          backend: "wasm",
          profile: true,
          onInitProgress: (stage, detail) => {
            this.handleInitProgress(stage, detail);
          },
        };
        this.detector = new Pose3DDetector(wasmConfig);
        await this.detector.init();
        this.activeBackend = "wasm";
        this.onStatus({
          state: "ready",
          engine: "rtmw3d",
          backend: "wasm",
          message: "RTMW3D-X model ready (WASM baseline)",
        });
      } else {
        this.onStatus({
          state: "error",
          engine: "rtmw3d",
          message: `RTMW3D initialization failed: ${err.message}`,
        });
        throw err;
      }
    }
  }

  handleInitProgress(stage, detail) {
    let msg = "Loading RTMW3D...";
    switch (stage) {
      case "start":
        msg = "Initialising RTMW3D pipeline...";
        break;
      case "mp-init":
        msg = "Initialising person detector (MediaPipe)...";
        break;
      case "mp-ready":
        msg = "Person detector ready.";
        break;
      case "yolo-load":
        msg = "Downloading YOLO person detector...";
        break;
      case "pose-load":
        msg = "Downloading RTMW3D-X model & initialising ONNX Runtime...";
        break;
      case "pose-ready":
        msg = "RTMW3D model ready.";
        break;
      case "ready":
        msg = "RTMW3D (3D) tracking engine ready.";
        break;
      default:
        if (detail) msg = `RTMW3D: ${detail}`;
    }

    this.onStatus({
      state: "loading",
      engine: "rtmw3d",
      message: msg,
    });
  }

  attachToScene(scene) {
    if (scene && !this.diagnosticAttached) {
      scene.add(this.diagnosticSkeleton.group);
      this.diagnosticAttached = true;
    }
  }

  setDiagnosticVisibility(visible) {
    this.diagnosticSkeleton.setVisible(visible);
  }

  detect(video, timestamp) {
    if (!this.detector || !video || !video.videoWidth || !video.videoHeight) {
      return this.latestResult;
    }

    // Process only one inference frame at a time; do not allow overlapping inference
    if (!this.isInferencing) {
      this.isInferencing = true;
      const tStart = performance.now();

      this.detector
        .detectFromVideo(video)
        .then((poseResult) => {
          const tEnd = performance.now();
          const inferTime = Math.round(tEnd - tStart);

          // Update FPS counter
          this.fpsFrameCount += 1;
          const elapsed = tEnd - this.fpsStartTime;
          let currentFps = this.latestResult.inferenceFps;
          if (elapsed >= 1000) {
            currentFps = Math.round((this.fpsFrameCount * 1000) / elapsed);
            this.fpsFrameCount = 0;
            this.fpsStartTime = tEnd;
          }

          // Convert to retargeting format
          const converted = convertRtmw3dToMediaPipe(
            poseResult,
            video.videoWidth,
            video.videoHeight,
          );

          // Update Three.js 3D Diagnostic Skeleton
          if (converted.keypoints3D && this.diagnosticSkeleton) {
            this.diagnosticSkeleton.update(converted.keypoints3D, converted.scores);
          }

          this.latestResult = {
            ...converted,
            inferenceTimeMs: inferTime,
            inferenceFps: currentFps,
            backend: this.activeBackend,
          };
        })
        .catch((err) => {
          console.warn("[RTMW3D] Detection error:", err);
          this.onStatus({
            state: "error",
            engine: "rtmw3d",
            message: `RTMW3D inference error: ${err.message}`,
          });
        })
        .finally(() => {
          this.isInferencing = false;
        });
    }

    return this.latestResult;
  }

  /**
   * Draws 2D detected body skeleton, joints, confidence, person count,
   * inference FPS, and tracking backend directly onto the camera preview debug canvas.
   */
  draw2DOverlay(canvas, video) {
    if (!canvas || !this.latestResult.keypoints2D) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * pixelRatio);
    const height = Math.round(rect.height * pixelRatio);

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const k2d = this.latestResult.keypoints2D;
    const scores = this.latestResult.scores;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scaleX = rect.width / vw;
    const scaleY = rect.height / vh;

    // Draw bone connection lines
    ctx.lineWidth = 2.5;
    for (const [from, to] of COCO17_CONNECTIONS) {
      const ptA = k2d[from];
      const ptB = k2d[to];
      const scA = scores ? (scores[from] ?? 1.0) : 1.0;
      const scB = scores ? (scores[to] ?? 1.0) : 1.0;

      if (!ptA || !ptB || scA < 0.25 || scB < 0.25) {
        continue;
      }

      ctx.beginPath();
      ctx.moveTo(ptA[0] * scaleX, ptA[1] * scaleY);
      ctx.lineTo(ptB[0] * scaleX, ptB[1] * scaleY);
      ctx.strokeStyle = Math.min(scA, scB) >= 0.5 ? "#00ff88" : "#ffb700";
      ctx.stroke();
    }

    // Draw joint circles
    for (let i = 0; i < 17; i++) {
      const pt = k2d[i];
      const score = scores ? (scores[i] ?? 1.0) : 1.0;
      if (!pt || score < 0.25) continue;

      ctx.beginPath();
      ctx.arc(pt[0] * scaleX, pt[1] * scaleY, 4, 0, Math.PI * 2);
      ctx.fillStyle = score >= 0.65 ? "#00ff88" : score >= 0.4 ? "#ffdd00" : "#ff4444";
      ctx.fill();
    }

    // Draw HUD diagnostics badge
    const stats = this.latestResult.stats;
    const backendLabel = this.activeBackend.toUpperCase();
    const fps = this.latestResult.inferenceFps || "--";
    const inferMs = this.latestResult.inferenceTimeMs || "--";
    const personCount = this.latestResult.personCount || 0;

    let meanConf = 0;
    if (scores && scores.length) {
      let sum = 0;
      for (let i = 0; i < 17; i++) sum += (scores[i] || 0);
      meanConf = Math.round((sum / 17) * 100);
    }

    const hudText = `RTMW3D [${backendLabel}] | Persons: ${personCount} | ${inferMs}ms (${fps} FPS) | Conf: ${meanConf}%`;
    ctx.font = "600 11px Segoe UI, system-ui, sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    const textWidth = ctx.measureText(hudText).width;
    ctx.fillRect(8, 8, textWidth + 14, 22);

    ctx.fillStyle = "#00ffcc";
    ctx.fillText(hudText, 15, 23);
  }

  dispose() {
    this.diagnosticSkeleton?.dispose();
    this.detector?.dispose?.();
    this.detector = null;
    this.diagnosticAttached = false;
  }

  close() {
    this.dispose();
  }
}
