import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { Avatar } from "./avatar.js";
import {
  createSCBACylinderApparatus,
  createSCBAFaceMask,
  createSafetyHelmet,
  createSafetyVest,
  createProtectiveGloves,
} from "./scbaApparatus.js";
import { mapLandmarkToStage, requestCameraStream } from "./camera.js";
import {
  UnifiedPoseTracker,
  POSE_CONNECTIONS,
  createSyntheticPose,
  isLandmarkUsable,
} from "./rtmposeTracker.js";
import { RTMW3DTracker } from "./src/rtmw3dTracker.js";
import { BodyPixSegmenter } from "./bodyPixSegmenter.js";
import { BodyPartSegmentationEngine } from "./src/bodyPartSegmentation.js";

const cameraPreview = document.querySelector("#cameraPreview");
const cameraPlaceholder = document.querySelector("#cameraPlaceholder");
const cameraStage = document.querySelector("#cameraStage");
const maskOverlay = document.querySelector("#maskOverlay");
const cameraToggle = document.querySelector("#cameraToggle");
const cameraMessage = document.querySelector("#cameraMessage");
const recordToggle = document.querySelector("#recordToggle");
const recordingMessage = document.querySelector("#recordingMessage");
const feedStatus = document.querySelector("#feedStatus");
const checkCount = document.querySelector("#checkCount");
const equipmentItems = document.querySelectorAll(".equipment-item");
const currentDate = document.querySelector("#currentDate");
const modelViewer = document.querySelector("#modelViewer");
const modelMessage = document.querySelector("#modelMessage");
const modelStatus = document.querySelector("#modelStatus");
const resetModelView = document.querySelector("#resetModelView");
const maskSizeInput = document.querySelector("#maskSize");
const maskSizeValue = document.querySelector("#maskSizeValue");
const maskOffsetXInput = document.querySelector("#maskOffsetX");
const maskOffsetXValue = document.querySelector("#maskOffsetXValue");
const maskOffsetYInput = document.querySelector("#maskOffsetY");
const maskOffsetYValue = document.querySelector("#maskOffsetYValue");
const maskDepthInput = document.querySelector("#maskDepth");
const maskDepthValue = document.querySelector("#maskDepthValue");
const resetCalibration = document.querySelector("#resetCalibration");
const avatarSizeInput = document.querySelector("#avatarSize");
const avatarSizeValue = document.querySelector("#avatarSizeValue");
const avatarDepthInput = document.querySelector("#avatarDepth");
const avatarDepthValue = document.querySelector("#avatarDepthValue");
const resetAvatarSliders = document.querySelector("#resetAvatarSliders");
const helmetToggle = document.querySelector("#helmetToggle");
const vestToggle = document.querySelector("#vestToggle");
const glovesToggle = document.querySelector("#glovesToggle");
const respiratorToggle = document.querySelector("#respiratorToggle");
const trackingEngine = document.querySelector("#trackingEngine");
const renderMode = document.querySelector("#renderMode");
const poseStatus = document.querySelector("#poseStatus");
const boneReport = document.querySelector("#boneReport");
const poseFps = document.querySelector("#poseFps");
const poseConfidence = document.querySelector("#poseConfidence");
const poseDebugCanvas = document.querySelector("#poseDebugCanvas");
const showPoseSkeleton = document.querySelector("#showPoseSkeleton");
const show3dDiagnosticSkeleton = document.querySelector("#show3dDiagnosticSkeleton");
const showAvatarSkeleton = document.querySelector("#showAvatarSkeleton");
const showJointPositions = document.querySelector("#showJointPositions");
const syntheticPoseInput = document.querySelector("#syntheticPoseInput");
const showRetargetingDebug = document.querySelector("#showRetargetingDebug");
const retargetingReadout = document.querySelector("#retargetingReadout");
const resetAvatarCalibration = document.querySelector("#resetAvatarCalibration");
const bodyPixEdgeCanvas = document.querySelector("#bodyPixEdgeCanvas");
const enableBodyPixSegmentation = document.querySelector("#enableBodyPixSegmentation");
const showBodyPixEdges = document.querySelector("#showBodyPixEdges");
const bodyPixFps = document.querySelector("#bodyPixFps");

// TensorFlow BodyPix 24 Body-Part Segmentation
const bodyPartSegmentationCanvas = document.querySelector("#bodyPartSegmentationCanvas");
const enableBodyPartSegmentation = document.querySelector("#enableBodyPartSegmentation");
const bodyPartResolution = document.querySelector("#bodyPartResolution");
const bodyPartDebugMode = document.querySelector("#bodyPartDebugMode");
const bodyPartFilter = document.querySelector("#bodyPartFilter");
const bodyPartFps = document.querySelector("#bodyPartFps");
const bodyPartDims = document.querySelector("#bodyPartDims");
const bodyPartPeople = document.querySelector("#bodyPartPeople");

const bodyPartEngine = new BodyPartSegmentationEngine({
  architecture: "MobileNetV1",
  outputStride: 16,
  multiplier: 0.75,
  quantBytes: 2,
  internalResolution: bodyPartResolution ? bodyPartResolution.value : "medium",
  targetFps: 60,
  debugMode: bodyPartDebugMode ? bodyPartDebugMode.value : "overlay",
  selectedPartGroup: bodyPartFilter ? bodyPartFilter.value : "all",
  opacity: 0.65,
});

bodyPartEngine.onStatus((status) => {
  if (bodyPartFps) {
    bodyPartFps.textContent = status.message || `BodyPix Parts: ${status.state}`;
  }
});

bodyPartResolution?.addEventListener("change", (e) => {
  bodyPartEngine.setInternalResolution(e.target.value);
});

enableBodyPartSegmentation?.addEventListener("change", (e) => {
  bodyPartEngine.enabled = e.target.checked;
  if (!e.target.checked && bodyPartSegmentationCanvas) {
    bodyPartSegmentationCanvas.hidden = true;
    const ctx = bodyPartSegmentationCanvas.getContext("2d");
    ctx?.clearRect(0, 0, bodyPartSegmentationCanvas.width, bodyPartSegmentationCanvas.height);
  }
});

bodyPartDebugMode?.addEventListener("change", (e) => {
  bodyPartEngine.setDebugMode(e.target.value);
  if (e.target.value === "off" && bodyPartSegmentationCanvas) {
    bodyPartSegmentationCanvas.hidden = true;
    const ctx = bodyPartSegmentationCanvas.getContext("2d");
    ctx?.clearRect(0, 0, bodyPartSegmentationCanvas.width, bodyPartSegmentationCanvas.height);
  }
});

bodyPartFilter?.addEventListener("change", (e) => {
  bodyPartEngine.setSelectedPartGroup(e.target.value);
});

const bodyPixSegmenter = new BodyPixSegmenter();
bodyPixSegmenter.onStatus(({ message }) => {
  if (bodyPixFps) bodyPixFps.textContent = message;
});

enableBodyPixSegmentation?.addEventListener("change", (e) => {
  bodyPixSegmenter.enabled = e.target.checked;
  if (!e.target.checked && bodyPixEdgeCanvas) {
    bodyPixEdgeCanvas.hidden = true;
  }
});

showBodyPixEdges?.addEventListener("change", (e) => {
  bodyPixSegmenter.showEdges = e.target.checked;
  if (!e.target.checked && bodyPixEdgeCanvas) {
    bodyPixEdgeCanvas.hidden = true;
  }
});

const OVERLAY_CAMERA_DISTANCE = 600;
const VIDEO_BACKGROUND_Z = -400;
const FACE_TESSELATION_INDICES = new Uint16Array(
  FaceLandmarker.FACE_LANDMARKS_TESSELATION.map((connection) => connection.start),
);
const FACE_VERTEX_COUNT = FACE_TESSELATION_INDICES.reduce(
  (count, index) => Math.max(count, index + 1),
  0,
);
let cameraStream = null;
let faceLandmarker = null;
let faceLandmarkerPromise = null;
let poseTracker = null;
let poseTrackerPromise = null;
let maskModelPromise = null;
let trackingFrameId = null;
let syntheticFrameId = null;
let lastVideoTime = -1;
let lastFaceDetected = null;
let poseFpsFrameCount = 0;
let poseFpsStartTime = performance.now();
let syntheticStartTime = 0;
let modelCamera;
let modelControls;
let maskOverlayRenderer;
let maskOverlayCamera;
let maskOverlayScene;
let maskOverlayModel;
let faceOccluder;
let headOccluder;
let maskLostFrames = 0;
let videoTexture = null;
let videoBackground = null;
let mediaRecorder = null;
let recordingChunks = [];
let recordingSaving = false;
const syntheticVideo = { videoWidth: 640, videoHeight: 480 };
const avatar = new Avatar(cameraStage, updateBoneReport);
avatar.video = cameraPreview;
avatar.onPostRender = () => {
  if (mediaRecorder?.state === "recording") {
    renderCompositeRecordingFrame();
  }
};
const overlayCalibration = { size: 0.82, offsetX: 0, offsetY: 0, depth: 0 };

function setPoseStatus(message, state = "idle") {
  poseStatus.textContent = message;
  poseStatus.dataset.state = state;
}

function setCameraState(isRunning) {
  cameraPreview.hidden = !isRunning;
  if (isRunning) {
    cameraPreview.style.display = "block";
    cameraPreview.style.opacity = "1";
    cameraPreview.style.visibility = "visible";
  } else {
    cameraPreview.style.display = "none";
  }
  maskOverlay.hidden = true; // Mask overlay kept hidden to eliminate duplicate respirator
  cameraPlaceholder.hidden = isRunning || syntheticPoseInput.checked;
  feedStatus.classList.toggle("is-live", isRunning);
  feedStatus.lastChild.textContent = isRunning ? " CAMERA LIVE" : " CAMERA OFF";
  cameraToggle.querySelector("span").textContent = isRunning
    ? "Stop camera"
    : "Start camera";
  cameraToggle.classList.toggle("is-active", isRunning);
  updateRecordingControls();
}

async function startCamera() {
  cameraToggle.disabled = true;
  cameraMessage.textContent = "Requesting camera access...";

  try {
    cameraStream = await requestCameraStream();
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    cameraStage.classList.remove("is-ar-active");
    setCameraState(true);
    cameraMessage.textContent = "Loading pose tracking...";
    startPoseTrackingLoop();
    void bodyPixSegmenter.init();
    void bodyPartEngine.init();
    try {
      poseTracker = await loadPoseTracker();
      setPoseStatus("Pose tracker ready; waiting for a visible person.", "ready");
      cameraMessage.textContent = "Camera live. 3D safety PPE tracking is active.";
    } catch (error) {
      console.error("Could not start pose tracking.", error);
      setPoseStatus(`Pose tracking unavailable: ${error.message}`, "error");
    }
    updateRecordingControls();
    recordingMessage.textContent = "Ready to record session to recordings/.";
  } catch (error) {
    if (!cameraStream && (error.name === "NotAllowedError" || error.name === "PermissionDeniedError")) {
      cameraMessage.textContent =
        "Camera permission was denied. Allow camera access in your browser settings and try again.";
    } else if (!cameraStream && (error.name === "NotFoundError" || error.name === "DevicesNotFoundError")) {
      cameraMessage.textContent = "No camera was found on this device.";
    } else {
      console.error("Could not start the camera.", error);
      stopCamera();
      cameraMessage.textContent = `Could not start the camera: ${error.message}`;
    }
  } finally {
    cameraToggle.disabled = false;
  }
}

function stopCamera() {
  stopPoseTrackingLoop();
  avatar.update(null, null);
  avatar.setSegmentationBounds(null);
  if (bodyPixEdgeCanvas) {
    bodyPixEdgeCanvas.hidden = true;
  }
  if (bodyPartSegmentationCanvas) {
    bodyPartSegmentationCanvas.hidden = true;
    const ctx = bodyPartSegmentationCanvas.getContext("2d");
    ctx?.clearRect(0, 0, bodyPartSegmentationCanvas.width, bodyPartSegmentationCanvas.height);
  }
  if (bodyPartFps) bodyPartFps.textContent = "BodyPix Parts: Ready";
  if (bodyPartDims) bodyPartDims.textContent = "Dims: --";
  if (bodyPartPeople) bodyPartPeople.textContent = "People: --";
  if (!syntheticPoseInput.checked) {
    poseConfidence.textContent = "Tracking: --";
    setPoseStatus("Camera stopped. Start it again for live pose tracking.", "idle");
  }
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraPreview.srcObject = null;
  if (maskOverlayModel) {
    maskOverlayModel.visible = false;
  }
  if (faceOccluder) {
    faceOccluder.visible = false;
  }
  if (headOccluder) {
    headOccluder.visible = false;
  }
  if (videoBackground) {
    maskOverlayScene.remove(videoBackground);
    videoBackground.geometry.dispose();
    videoBackground.material.dispose();
    videoBackground = null;
  }
  videoTexture?.dispose();
  videoTexture = null;
  cameraStage.classList.remove("is-ar-active");
  setCameraState(false);
  cameraMessage.textContent = "Camera and face tracking run locally in this browser.";
  if (!recordingSaving) {
    recordingMessage.textContent = "Start the camera to record a clip to recordings/.";
  }
}

cameraToggle.addEventListener("click", () => {
  if (cameraStream) {
    stopCamera();
  } else {
    startCamera();
  }
});

let recordingCanvas = null;
let recordingCtx = null;
let recordingFrameId = null;

function renderCompositeRecordingFrame() {
  if (mediaRecorder?.state !== "recording" || !recordingCtx || !recordingCanvas) {
    return;
  }

  const w = recordingCanvas.width;
  const h = recordingCanvas.height;

  // Draw Camera Feed with cover-fit matching live preview exactly
  if (cameraPreview && cameraPreview.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const videoW = cameraPreview.videoWidth || w;
    const videoH = cameraPreview.videoHeight || h;
    const videoScale = Math.max(w / videoW, h / videoH);
    const drawW = videoW * videoScale;
    const drawH = videoH * videoScale;
    const drawX = (w - drawW) / 2;
    const drawY = (h - drawH) / 2;

    recordingCtx.save();
    if (renderMode.value !== "environment") {
      recordingCtx.translate(w, 0);
      recordingCtx.scale(-1, 1);
    }
    recordingCtx.drawImage(cameraPreview, drawX, drawY, drawW, drawH);
    recordingCtx.restore();
  } else {
    recordingCtx.fillStyle = "#141b1c";
    recordingCtx.fillRect(0, 0, w, h);
  }

  // Draw 3D Avatar (if loaded and active)
  const avatarCanvas = avatar?.renderer?.domElement;
  if (avatarCanvas && avatarCanvas.width > 0 && avatarCanvas.height > 0) {
    if (renderMode.value !== "environment") {
      recordingCtx.save();
      recordingCtx.translate(w, 0);
      recordingCtx.scale(-1, 1);
      recordingCtx.drawImage(avatarCanvas, 0, 0, w, h);
      recordingCtx.restore();
    } else {
      recordingCtx.drawImage(avatarCanvas, 0, 0, w, h);
    }
  }

  // Draw Pose Debug skeleton (if visible)
  if (poseDebugCanvas && !poseDebugCanvas.hidden && poseDebugCanvas.width > 0 && poseDebugCanvas.height > 0) {
    recordingCtx.drawImage(poseDebugCanvas, 0, 0, w, h);
  }

  // Draw BodyPix segmentation edge overlay (if active)
  if (bodyPixEdgeCanvas && !bodyPixEdgeCanvas.hidden && bodyPixEdgeCanvas.width > 0 && bodyPixEdgeCanvas.height > 0) {
    recordingCtx.save();
    if (renderMode.value !== "environment") {
      recordingCtx.translate(w, 0);
      recordingCtx.scale(-1, 1);
    }
    recordingCtx.drawImage(bodyPixEdgeCanvas, 0, 0, w, h);
    recordingCtx.restore();
  }

  // Draw BodyPix 24 body-part segmentation overlay (if active)
  if (
    bodyPartSegmentationCanvas &&
    !bodyPartSegmentationCanvas.hidden &&
    bodyPartSegmentationCanvas.width > 0 &&
    bodyPartSegmentationCanvas.height > 0
  ) {
    recordingCtx.save();
    if (renderMode.value !== "environment") {
      recordingCtx.translate(w, 0);
      recordingCtx.scale(-1, 1);
    }
    recordingCtx.drawImage(bodyPartSegmentationCanvas, 0, 0, w, h);
    recordingCtx.restore();
  }
}

function updateRecordingControls() {
  const isRecording = mediaRecorder?.state === "recording";
  const canRecord =
    Boolean(cameraStream) &&
    typeof MediaRecorder !== "undefined" &&
    !recordingSaving;

  recordToggle.disabled = !isRecording && !canRecord;
  recordToggle.textContent = recordingSaving
    ? "Saving clip..."
    : isRecording
      ? "Stop and save"
      : "Record clip";
  recordToggle.classList.toggle("is-recording", isRecording);
}

async function saveRecording(blob) {
  try {
    recordingMessage.textContent = "Saving clip to recordings/...";
    const response = await fetch("/api/recordings", {
      method: "POST",
      headers: { "Content-Type": blob.type || "video/webm" },
      body: blob,
    });
    const result = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(result?.error || `Recording save failed (${response.status}).`);
    }
    if (!result?.filename) {
      throw new Error("Recording server returned no filename.");
    }
    recordingMessage.textContent = `Saved to recordings/${result.filename}`;
  } catch (error) {
    console.error("Could not save the browser recording to server.", error);
    // Automatic fallback: download locally in browser so recorded clip is never lost
    try {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const ext = (blob.type || "").includes("mp4") ? "mp4" : "webm";
      const filename = `safety_mirror_${new Date().toISOString().replace(/[:.]/g, "-")}.${ext}`;
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 3000);
      recordingMessage.textContent = `Server save failed (${error.message}); saved via browser download as ${filename}`;
    } catch (fallbackError) {
      recordingMessage.textContent = `Could not save the clip: ${error.message}`;
    }
  } finally {
    recordingSaving = false;
    cameraToggle.disabled = false;
    updateRecordingControls();
  }
}

function startRecording() {
  if (!cameraStream) {
    throw new Error("Start the camera before recording.");
  }
  if (typeof MediaRecorder === "undefined") {
    throw new Error("This browser does not support video recording.");
  }

  const mimeType = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4;codecs=avc1",
    "video/mp4",
  ].find((candidate) => MediaRecorder.isTypeSupported(candidate)) || "";

  // Set up composite recording canvas matching live stage aspect ratio with even dimensions for HW encoding
  const stage = cameraStage.getBoundingClientRect();
  const stageWidth = Math.round(stage.width) || 1280;
  const stageHeight = Math.round(stage.height) || 720;
  const aspect = stageWidth / stageHeight;
  const targetH = Math.max(720, Math.min(1080, stageHeight));
  const height = Math.round(targetH) & ~1;
  const width = Math.round(height * aspect) & ~1;

  if (!recordingCanvas) {
    recordingCanvas = document.createElement("canvas");
  }
  recordingCanvas.width = width;
  recordingCanvas.height = height;
  recordingCtx = recordingCanvas.getContext("2d", { alpha: false, desynchronized: true }) || recordingCanvas.getContext("2d");
  if (recordingCtx) {
    recordingCtx.imageSmoothingEnabled = true;
    recordingCtx.imageSmoothingQuality = "medium";
  }

  let recordingStream;
  try {
    recordingStream = recordingCanvas.captureStream(30);
  } catch (error) {
    console.warn("Canvas captureStream failed; falling back to direct camera stream.", error);
    recordingStream = cameraStream.clone();
  }

  const recorderOptions = {
    videoBitsPerSecond: 4_500_000,
  };
  if (mimeType) {
    recorderOptions.mimeType = mimeType;
  }

  let recorder;
  try {
    recorder = new MediaRecorder(recordingStream, recorderOptions);
  } catch {
    try {
      recorder = mimeType
        ? new MediaRecorder(recordingStream, { mimeType })
        : new MediaRecorder(recordingStream);
    } catch (error) {
      if (recordingFrameId) {
        cancelAnimationFrame(recordingFrameId);
        recordingFrameId = null;
      }
      recordingStream.getTracks().forEach((track) => track.stop());
      throw new Error(`Could not start recording: ${error.message}`);
    }
  }

  let recordingError = null;
  mediaRecorder = recorder;
  recordingChunks = [];

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      recordingChunks.push(event.data);
    }
  });

  recorder.addEventListener("error", (event) => {
    recordingError = event.error || new Error("The browser recorder failed.");
    console.error("Browser recording failed.", recordingError);
    recordingMessage.textContent = `Recording failed: ${recordingError.message}`;
    if (recorder.state === "recording") {
      recorder.stop();
    }
  });

  recorder.addEventListener("stop", () => {
    if (recordingFrameId) {
      cancelAnimationFrame(recordingFrameId);
      recordingFrameId = null;
    }
    recordingStream.getTracks().forEach((track) => track.stop());
    mediaRecorder = null;
    updateRecordingControls();

    if (recordingError) {
      recordingChunks = [];
      cameraToggle.disabled = false;
      return;
    }

    const recordedType = recorder.mimeType || mimeType || "video/webm";
    const blob = new Blob(recordingChunks, { type: recordedType });
    recordingChunks = [];

    if (blob.size === 0) {
      recordingMessage.textContent = "No video data was captured; the clip was not saved.";
      cameraToggle.disabled = false;
      updateRecordingControls();
      return;
    }

    recordingSaving = true;
    cameraToggle.disabled = true;
    recordingMessage.textContent = "Saving clip to recordings/...";
    updateRecordingControls();
    void saveRecording(blob);
  });

  try {
    recorder.start(1000);
    renderCompositeRecordingFrame();
  } catch (error) {
    if (recordingFrameId) {
      cancelAnimationFrame(recordingFrameId);
      recordingFrameId = null;
    }
    mediaRecorder = null;
    recordingStream.getTracks().forEach((track) => track.stop());
    updateRecordingControls();
    throw new Error(`Could not start recording: ${error.message}`);
  }

  cameraToggle.disabled = true;
  recordingMessage.textContent = "Recording safety mirror session...";
  updateRecordingControls();
}

recordToggle.addEventListener("click", () => {
  if (mediaRecorder?.state === "recording") {
    recordingMessage.textContent = "Finishing clip...";
    mediaRecorder.stop();
    return;
  }

  try {
    startRecording();
  } catch (error) {
    recordingMessage.textContent = error.message;
    console.error("Could not start the browser recording.", error);
  }
});

updateRecordingControls();

function updateCalibration() {
  const size = Number(maskSizeInput.value) / 100;
  const offsetX = Number(maskOffsetXInput.value);
  const offsetY = Number(maskOffsetYInput.value);
  const depth = Number(maskDepthInput.value);
  overlayCalibration.size = size;
  overlayCalibration.offsetX = offsetX;
  overlayCalibration.offsetY = offsetY;
  overlayCalibration.depth = depth;
  maskSizeValue.value = `${maskSizeInput.value}%`;
  maskOffsetXValue.value = `${offsetX} px`;
  maskOffsetYValue.value = `${offsetY} px`;
  maskDepthValue.value = `${depth} px`;
  if (avatar) {
    avatar.setMaskCalibration({ size, offsetX, offsetY, depth });
  }
}

[maskSizeInput, maskOffsetXInput, maskOffsetYInput, maskDepthInput].forEach((input) => {
  input.addEventListener("input", updateCalibration);
});

resetCalibration.addEventListener("click", () => {
  maskSizeInput.value = "100";
  maskOffsetXInput.value = "0";
  maskOffsetYInput.value = "0";
  maskDepthInput.value = "0";
  updateCalibration();
});

function updateBoneReport(report) {
  if (report.state === "inspecting") {
    const detected = report.detected.length ? report.detected.join(", ") : "none";
    boneReport.textContent =
      `Detected bones: ${detected}. Mapped: ${report.mapped.join(", ") || "none"}. ` +
      `Missing: ${report.missing.join(", ") || "none"}. Skinned meshes: ${report.skinnedMeshCount}.`;
    return;
  }
  if (report.state === "ready") {
    avatar.setDebug({ avatarSkeleton: showAvatarSkeleton.checked });
    avatar.setCalibrationDebug(showRetargetingDebug.checked);
    boneReport.textContent =
      `Detected bones: ${report.detected.join(", ") || "none"}. ` +
      `Mapped: ${report.mapped.join(", ") || "none"}. ` +
      `Missing: ${report.missing.join(", ") || "none"}.`;
    return;
  }
  const detected = report.detected?.join(", ") || "none";
  const mapped = report.mapped?.join(", ") || "none";
  const missing = report.missing?.join(", ") || "none";
  boneReport.textContent =
    `Avatar unavailable: ${report.message} Detected: ${detected}. ` +
    `Mapped: ${mapped}. Missing: ${missing}.`;
}

function drawPoseDebug(landmarks, video = cameraPreview) {
  const showSkeleton = showPoseSkeleton.checked;
  const showJoints = showJointPositions.checked;
  const showCalibration = showRetargetingDebug.checked;
  const mirrorDebug = !cameraStage.classList.contains("is-environment");

  if (poseTracker?.activeEngine === "rtmw3d" && poseTracker.rtmw3dTracker) {
    poseDebugCanvas.hidden = !showSkeleton && !showJoints;
    if (!poseDebugCanvas.hidden) {
      poseTracker.rtmw3dTracker.draw2DOverlay(poseDebugCanvas, video);
    }
    return;
  }

  poseDebugCanvas.hidden = !showSkeleton && !showJoints && !showCalibration;
  if (poseDebugCanvas.hidden) {
    return;
  }

  const rect = cameraStage.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * pixelRatio);
  const height = Math.round(rect.height * pixelRatio);
  if (poseDebugCanvas.width !== width || poseDebugCanvas.height !== height) {
    poseDebugCanvas.width = width;
    poseDebugCanvas.height = height;
  }
  const context = poseDebugCanvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  context.lineWidth = 2;
  context.strokeStyle = "#75e69a";
  context.fillStyle = "#c5f36b";

  if (showSkeleton) {
    for (const [from, to] of POSE_CONNECTIONS) {
      if (!isLandmarkUsable(landmarks[from]) || !isLandmarkUsable(landmarks[to])) {
        continue;
      }
      const start = mapLandmarkToStage(landmarks[from], video, rect, mirrorDebug);
      const end = mapLandmarkToStage(landmarks[to], video, rect, mirrorDebug);
      if (!start || !end) {
        continue;
      }
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }
  }

  if (showJoints) {
    for (const landmark of landmarks) {
      if (!isLandmarkUsable(landmark)) {
        continue;
      }
      const point = mapLandmarkToStage(landmark, video, rect, mirrorDebug);
      if (point) {
        context.beginPath();
        context.arc(point.x, point.y, 3, 0, Math.PI * 2);
        context.fill();
      }
    }
  }

  if (showCalibration) {
    const labels = [
      [11, "L shoulder", "#66e7ff"],
      [13, "L elbow", "#66e7ff"],
      [15, "L wrist", "#66e7ff"],
      [12, "R shoulder", "#ffae75"],
      [14, "R elbow", "#ffae75"],
      [16, "R wrist", "#ffae75"],
    ];
    context.font = "600 11px Segoe UI, sans-serif";
    for (const [index, label, color] of labels) {
      if (!isLandmarkUsable(landmarks[index])) continue;
      const point = mapLandmarkToStage(landmarks[index], video, rect, mirrorDebug);
      if (!point) continue;
      context.fillStyle = color;
      context.beginPath();
      context.arc(point.x, point.y, 4, 0, Math.PI * 2);
      context.fill();
      context.fillText(label, point.x + 7, point.y - 6);
    }

    const leftShoulder = isLandmarkUsable(landmarks[11])
      ? mapLandmarkToStage(landmarks[11], video, rect, mirrorDebug)
      : null;
    const rightShoulder = isLandmarkUsable(landmarks[12])
      ? mapLandmarkToStage(landmarks[12], video, rect, mirrorDebug)
      : null;
    const origin = leftShoulder && rightShoulder
      ? {
          x: (leftShoulder.x + rightShoulder.x) / 2,
          y: (leftShoulder.y + rightShoulder.y) / 2,
        }
      : null;
    const diagnostics = avatar.retargeter?.diagnostics;
    if (origin && diagnostics) {
      const drawAxis = (vector, label, color) => {
        const x = mirrorDebug ? -vector.x : vector.x;
        const y = -vector.y;
        const length = Math.hypot(x, y);
        context.strokeStyle = color;
        context.fillStyle = color;
        if (length < 0.15) {
          context.beginPath();
          context.arc(origin.x + 14, origin.y - 12, 5, 0, Math.PI * 2);
          context.stroke();
          context.fillText(`${label} (depth)`, origin.x + 22, origin.y - 10);
          return;
        }
        const endX = origin.x + (x / length) * 48;
        const endY = origin.y + (y / length) * 48;
        context.beginPath();
        context.moveTo(origin.x, origin.y);
        context.lineTo(endX, endY);
        context.stroke();
        context.fillText(label, endX + 4, endY - 4);
      };
      drawAxis(diagnostics.torsoUp, "Torso up", "#c5f36b");
      drawAxis(diagnostics.torsoForward, "Torso forward", "#f7e36d");
      drawAxis(diagnostics.headForward, "Head forward", "#66e7ff");
    }

    const avatarPoints = avatar.getDebugBonePoints(rect);
    const comparisons = [
      ["leftShoulder", 11, "L shoulder", "#66e7ff"],
      ["leftForearm", 13, "L elbow", "#66e7ff"],
      ["leftHand", 15, "L wrist", "#66e7ff"],
      ["rightShoulder", 12, "R shoulder", "#ffae75"],
      ["rightForearm", 14, "R elbow", "#ffae75"],
      ["rightHand", 16, "R wrist", "#ffae75"],
      ["hips", null, "hips", "#c5f36b"],
      ["leftUpperLeg", 23, "L hip", "#66e7ff"],
      ["leftLeg", 25, "L knee", "#66e7ff"],
      ["rightUpperLeg", 24, "R hip", "#ffae75"],
      ["rightLeg", 26, "R knee", "#ffae75"],
    ];
    let totalError = 0;
    let comparedJoints = 0;
    const jointOffsets = {};
    for (const [role, index, label, color] of comparisons) {
      const avatarPoint = avatarPoints[role];
      let posePoint = null;
      if (index === null && isLandmarkUsable(landmarks[23]) && isLandmarkUsable(landmarks[24])) {
        const leftHip = mapLandmarkToStage(landmarks[23], video, rect, mirrorDebug);
        const rightHip = mapLandmarkToStage(landmarks[24], video, rect, mirrorDebug);
        if (leftHip && rightHip) {
          posePoint = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2,
          };
        }
      } else if (index !== null && isLandmarkUsable(landmarks[index])) {
        posePoint = mapLandmarkToStage(landmarks[index], video, rect, mirrorDebug);
      }
      if (!avatarPoint || !posePoint) continue;

      const error = Math.hypot(avatarPoint.x - posePoint.x, avatarPoint.y - posePoint.y);
      totalError += error;
      comparedJoints += 1;
      jointOffsets[role] =
        `${Math.round(avatarPoint.x - posePoint.x)},${Math.round(avatarPoint.y - posePoint.y)}`;
      context.strokeStyle = color;
      context.globalAlpha = 0.65;
      context.beginPath();
      context.moveTo(posePoint.x, posePoint.y);
      context.lineTo(avatarPoint.x, avatarPoint.y);
      context.stroke();
      context.globalAlpha = 1;
      context.strokeStyle = color;
      context.beginPath();
      context.arc(avatarPoint.x, avatarPoint.y, 5, 0, Math.PI * 2);
      context.stroke();
      context.fillStyle = color;
      context.fillText(`avatar ${label}`, avatarPoint.x + 7, avatarPoint.y + 12);
    }
    avatar.lastJointErrorPx = comparedJoints ? totalError / comparedJoints : null;
    avatar.lastJointOffsets = jointOffsets;
  }
}

function updateRetargetingReadout() {
  retargetingReadout.hidden = !showRetargetingDebug.checked;
  if (!showRetargetingDebug.checked) return;
  const diagnostics = avatar.retargeter?.diagnostics;
  if (!diagnostics) {
    retargetingReadout.textContent = "Waiting for the avatar rig to load.";
    return;
  }
  const degrees = (radians) => Math.round(THREE.MathUtils.radToDeg(radians));
  const vector = (value) =>
    `${value.x.toFixed(2)}, ${value.y.toFixed(2)}, ${value.z.toFixed(2)}`;
  retargetingReadout.textContent =
    `Torso yaw/pitch/roll: ${degrees(diagnostics.torsoYaw)}° / ` +
    `${degrees(diagnostics.torsoPitch)}° / ${degrees(diagnostics.torsoRoll)}°. ` +
    `Head relative yaw/pitch/roll: ${degrees(diagnostics.headYaw)}° / ` +
    `${degrees(diagnostics.headPitch)}° / ${degrees(diagnostics.headRoll)}°. ` +
    `Torso forward ${vector(diagnostics.torsoForward)}, up ${vector(diagnostics.torsoUp)}; ` +
    `tracked L/R shoulders ${vector(diagnostics.trackedLeftShoulder)} / ${vector(diagnostics.trackedRightShoulder)}; ` +
    `model forward ${vector(diagnostics.modelForward)}, head forward ${vector(diagnostics.headForward)}. ` +
    `Scale refs x/y: ${avatar.lastHorizontalScaleReference} / ${avatar.lastScaleReference}; ` +
    `avatar shoulder/hip widths ${avatar.modelShoulderWidth.toFixed(3)} / ${avatar.modelHipWidth.toFixed(3)} ` +
    `(${avatar.modelShoulderAnchors}), detected shoulder width ${avatar.lastDetectedShoulderWidth.toFixed(1)} px, ` +
    `body/torso/head-shoulder heights ${avatar.lastDetectedBodyHeight.toFixed(1)} / ` +
    `${avatar.lastDetectedTorsoHeight.toFixed(1)} / ${avatar.lastDetectedHeadShoulderHeight.toFixed(1)} px, ` +
    `model torso/head-shoulder ${avatar.modelTorsoHeight.toFixed(3)} / ` +
    `${avatar.modelHeadShoulderHeight.toFixed(3)}, scale x/y ${avatar.group.scale.x.toFixed(1)} / ` +
    `${avatar.group.scale.y.toFixed(1)} (target ${avatar.scaleTarget.x.toFixed(1)} / ` +
    `${avatar.scaleTarget.y.toFixed(1)}), depth Z ${avatar.lastDetectedDepth?.toFixed(1) ?? "0.0"} px; ` +
    `mean joint offset ${avatar.lastJointErrorPx?.toFixed(1) ?? "--"} px; offsets ` +
    `${Object.entries(avatar.lastJointOffsets).map(([role, error]) => `${role}:${error}`).join(", ") || "--"} px.`;
}

function recordPoseFrame(now, source) {
  poseFpsFrameCount += 1;
  const elapsed = now - poseFpsStartTime;
  if (elapsed < 1000) {
    return;
  }
  const inferenceFps = Math.round((poseFpsFrameCount * 1000) / elapsed);
  poseFps.textContent = `FPS: ${inferenceFps} ${source} / ${avatar.fps || "--"} render`;
  poseFpsFrameCount = 0;
  poseFpsStartTime = now;
}

function updatePoseFrame(frameTime = performance.now()) {
  if (syntheticPoseInput.checked) {
    return;
  }
  if (!poseTracker || cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    avatar.update(null, null);
    return;
  }
  try {
    const now = frameTime;
    const pose = poseTracker.detect(cameraPreview, now);
    recordPoseFrame(now, "pose");
    avatar.update(pose.landmarks, pose.worldLandmarks, cameraPreview);

    // BodyPix Segmentation & Pose processing
    const isBodyPixEngine = poseTracker.activeEngine === "bodypix";
    const shouldRunBodyParts = isBodyPixEngine || (enableBodyPartSegmentation?.checked && bodyPartEngine.ready);

    // Primary BodyPix 24 Body-Part pass (single GPU inference serving both pose and segmentation)
    if (shouldRunBodyParts && bodyPartEngine.ready) {
      bodyPartEngine
        .segment(cameraPreview, now)
        .then((partResult) => {
          if (!partResult) return;

          // Provide silhouette bounds to 3D avatar for occlusion and scale calibration
          if (partResult.bounds) {
            avatar.setSegmentationBounds(partResult.bounds);
          }

          if (
            bodyPartSegmentationCanvas &&
            (bodyPartEngine.options.debugMode !== "off" || isBodyPixEngine)
          ) {
            bodyPartSegmentationCanvas.hidden = false;
            const rect = cameraStage.getBoundingClientRect();
            const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
            const w = Math.round(rect.width * pixelRatio);
            const h = Math.round(rect.height * pixelRatio);
            if (
              bodyPartSegmentationCanvas.width !== w ||
              bodyPartSegmentationCanvas.height !== h
            ) {
              bodyPartSegmentationCanvas.width = w;
              bodyPartSegmentationCanvas.height = h;
            }
            bodyPartEngine.renderOverlay(bodyPartSegmentationCanvas, cameraPreview);
          } else if (bodyPartSegmentationCanvas) {
            bodyPartSegmentationCanvas.hidden = true;
          }

          if (bodyPartFps) {
            bodyPartFps.textContent = `BodyPix Parts: ${bodyPartEngine.fps || "--"} FPS (${bodyPartEngine.inferenceLatencyMs}ms)`;
          }
          if (bodyPartDims) {
            bodyPartDims.textContent = `Dims: ${partResult.width}×${partResult.height}`;
          }
          if (bodyPartPeople) {
            bodyPartPeople.textContent = `People: ${partResult.peopleCount}`;
          }

          if (isBodyPixEngine && poseFps) {
            poseFps.textContent = `FPS: ${bodyPartEngine.fps || "--"}`;
          }
        })
        .catch((e) => console.warn("BodyPix body-part segmentation error:", e));
    } else if (bodyPartSegmentationCanvas) {
      bodyPartSegmentationCanvas.hidden = true;
    }

    // Edge contours pass: only run secondary model if bodyPartEngine is NOT running, preventing GPU contention
    if (!shouldRunBodyParts && enableBodyPixSegmentation?.checked && bodyPixSegmenter.ready) {
      bodyPixSegmenter
        .segment(cameraPreview)
        .then((segResult) => {
          if (!segResult) return;
          if (segResult.bounds) {
            avatar.setSegmentationBounds(segResult.bounds);
          }
          if (showBodyPixEdges?.checked && bodyPixSegmenter.enabled) {
            bodyPixEdgeCanvas.hidden = false;
            const rect = cameraStage.getBoundingClientRect();
            const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
            const w = Math.round(rect.width * pixelRatio);
            const h = Math.round(rect.height * pixelRatio);
            if (bodyPixEdgeCanvas.width !== w || bodyPixEdgeCanvas.height !== h) {
              bodyPixEdgeCanvas.width = w;
              bodyPixEdgeCanvas.height = h;
            }
            const ctx = bodyPixEdgeCanvas.getContext("2d");
            ctx.clearRect(0, 0, w, h);
            ctx.drawImage(segResult.edgeCanvas, 0, 0, w, h);
          } else {
            bodyPixEdgeCanvas.hidden = true;
          }
          if (bodyPixFps) {
            bodyPixFps.textContent = `BodyPix: ${bodyPixSegmenter.fps || "--"} FPS`;
          }
        })
        .catch((e) => console.warn("BodyPix frame error:", e));
    } else if (!showBodyPixEdges?.checked || shouldRunBodyParts) {
      if (bodyPixEdgeCanvas) bodyPixEdgeCanvas.hidden = true;
    }

    updateRetargetingReadout();
    const landmarks = pose.landmarks;
    if (!landmarks || !pose.worldLandmarks) {
      drawPoseDebug([]);
      poseConfidence.textContent = "Tracking: 0%";
      setPoseStatus("No pose detected. Stand back so your body is visible.", "lost");
      return;
    }

    if (poseTracker.activeEngine === "rtmw3d" && poseTracker.rtmw3dTracker && avatar?.scene) {
      poseTracker.rtmw3dTracker.attachToScene(avatar.scene);
      if (show3dDiagnosticSkeleton) {
        poseTracker.rtmw3dTracker.setDiagnosticVisibility(show3dDiagnosticSkeleton.checked);
      }
    }

    drawPoseDebug(landmarks);
    if (poseTracker.activeEngine === "rtmw3d" && pose.inferenceFps) {
      poseFps.textContent = `FPS: ${pose.inferenceFps} [${(pose.backend || "WASM").toUpperCase()}] / ${avatar.fps || "--"} render`;
    } else if (poseTracker.activeEngine === "bodypix" && bodyPartEngine) {
      poseFps.textContent = `FPS: ${bodyPartEngine.fps || "--"} [WebGL GPU] (${bodyPartEngine.inferenceLatencyMs}ms)`;
    }
    const trackedIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    const visible = trackedIndices.filter((index) =>
      isLandmarkUsable(landmarks[index]),
    );
    const confidence = visible.length / trackedIndices.length;
    const meanConfidence = visible.length
      ? visible.reduce(
          (sum, index) =>
            sum +
            (landmarks[index].visibility ?? landmarks[index].presence ?? 1),
          0,
        ) / visible.length
      : 0;
    poseConfidence.textContent =
      `Tracking: ${Math.round(confidence * 100)}% coverage · ` +
      `${Math.round(meanConfidence * 100)}% confidence`;
    if (visible.length < 10) {
      setPoseStatus(
        "Partial pose detected. Keep more of your body in frame; visible joints still animate.",
        "partial",
      );
    } else if (avatar.retargeter) {
      setPoseStatus("Live webcam pose is being retargeted to the avatar.", "tracking");
    } else {
      setPoseStatus("Pose detected; a usable rigged humanoid GLB is required to animate.", "tracking");
    }
  } catch (error) {
    console.error("Pose tracking failed.", error);
    setPoseStatus(`Pose tracking stopped: ${error.message}`, "error");
    const failedTracker = poseTracker;
    poseTracker = null;
    poseTrackerPromise = null;
    try {
      failedTracker.close();
    } catch (closeError) {
      console.error("Could not release the failed pose tracker.", closeError);
    }
    avatar.update(null, null);
  }
}

function updateSyntheticPoseFrame(timestamp) {
  if (!syntheticPoseInput.checked) {
    syntheticFrameId = null;
    return;
  }
  const pose = createSyntheticPose(timestamp - syntheticStartTime);
  avatar.update(pose.landmarks, pose.worldLandmarks, syntheticVideo);
  updateRetargetingReadout();
  drawPoseDebug(pose.landmarks || [], syntheticVideo);
  recordPoseFrame(performance.now(), "synthetic");
  if (pose.landmarks) {
    const trackedIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    const visibleCount = trackedIndices.filter((index) =>
      isLandmarkUsable(pose.landmarks[index]),
    ).length;
    poseConfidence.textContent = `Synthetic coverage: ${Math.round(
      (visibleCount / trackedIndices.length) * 100,
    )}%`;
    setPoseStatus(
      `Synthetic debug only: ${pose.phase}; no webcam pose is being verified.`,
      "synthetic",
    );
  } else {
    poseConfidence.textContent = "Synthetic tracking: 0%";
    setPoseStatus(
      `Synthetic debug only: ${pose.phase}; avatar is easing to rest.`,
      "synthetic",
    );
  }
  syntheticFrameId = requestAnimationFrame(updateSyntheticPoseFrame);
}

let selectedTrackingEngine = trackingEngine ? trackingEngine.value : "mediapipe";

function loadPoseTracker() {
  if (!poseTrackerPromise) {
    const preferredEngine = selectedTrackingEngine || (trackingEngine ? trackingEngine.value : "mediapipe");
    poseTrackerPromise = UnifiedPoseTracker.create({
      preferredEngine,
      bodyPartEngine,
      onStatus: (status) => {
        setPoseStatus(status.message, status.state);
      },
    }).catch((error) => {
      poseTrackerPromise = null;
      throw error;
    });
  }
  return poseTrackerPromise;
}

trackingEngine?.addEventListener("change", async (event) => {
  const engine = event.target.value;
  selectedTrackingEngine = engine;

  if (!poseTracker) {
    // Pose tracker has not yet started (e.g. camera is stopped)
    setPoseStatus(`Engine set to ${engine.toUpperCase()}. Will start when camera begins.`, "idle");
    return;
  }

  // Live switching while camera is active
  const previousEngine = poseTracker.activeEngine;
  trackingEngine.disabled = true;
  setPoseStatus(`Switching tracking engine to ${engine}...`, "loading");

  try {
    if (engine !== "rtmw3d" && poseTracker.rtmw3dTracker) {
      poseTracker.rtmw3dTracker.setDiagnosticVisibility(false);
    }
    // Clear 2D overlay to avoid frozen frames between engines
    drawPoseDebug([]);

    if (engine === "bodypix") {
      poseTracker.setBodyPartEngine(bodyPartEngine);
      if (!bodyPartEngine.ready && !bodyPartEngine.loading) {
        await bodyPartEngine.init();
      }
      bodyPartEngine.enabled = true;
      if (enableBodyPartSegmentation) enableBodyPartSegmentation.checked = true;
      if (bodyPartDebugMode && bodyPartDebugMode.value === "off") {
        bodyPartDebugMode.value = "overlay";
        bodyPartEngine.setDebugMode("overlay");
      }
    }

    await poseTracker.setEngine(engine);

    if (engine === "rtmw3d" && poseTracker.rtmw3dTracker && avatar?.scene) {
      poseTracker.rtmw3dTracker.attachToScene(avatar.scene);
      if (show3dDiagnosticSkeleton) {
        poseTracker.rtmw3dTracker.setDiagnosticVisibility(show3dDiagnosticSkeleton.checked);
      }
    }
  } catch (err) {
    console.error("Failed to switch tracking engine:", err);
    setPoseStatus(`Engine switch failed: ${err.message}. Retained ${previousEngine}.`, "error");
    trackingEngine.value = previousEngine;
    selectedTrackingEngine = previousEngine;
  } finally {
    trackingEngine.disabled = false;
  }
});

show3dDiagnosticSkeleton?.addEventListener("change", () => {
  if (poseTracker?.rtmw3dTracker) {
    poseTracker.rtmw3dTracker.setDiagnosticVisibility(show3dDiagnosticSkeleton.checked);
  }
});

[showPoseSkeleton, showJointPositions].forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) {
      drawPoseDebug([]);
    }
  });
});

syntheticPoseInput.addEventListener("change", () => {
  poseFpsFrameCount = 0;
  poseFpsStartTime = performance.now();
  if (syntheticPoseInput.checked) {
    cameraPlaceholder.hidden = true;
    cameraMessage.textContent = "Synthetic motion test active; this is not live webcam tracking.";
    syntheticStartTime = performance.now();
    syntheticFrameId = requestAnimationFrame(updateSyntheticPoseFrame);
    return;
  }
  if (syntheticFrameId !== null) {
    cancelAnimationFrame(syntheticFrameId);
    syntheticFrameId = null;
  }
  avatar.update(null, null);
  drawPoseDebug([]);
  poseConfidence.textContent = "Tracking: --";
  cameraMessage.textContent = cameraStream
    ? "Synthetic motion test stopped; live camera remains available."
    : "Camera and face tracking run locally in this browser.";
  setPoseStatus(
    cameraStream
      ? "Synthetic test stopped; waiting for live webcam pose."
      : "Synthetic test stopped. Start the camera for live pose tracking.",
    cameraStream ? "ready" : "idle",
  );
  cameraPlaceholder.hidden = Boolean(cameraStream);
});

showAvatarSkeleton.addEventListener("change", () => {
  avatar.setDebug({ avatarSkeleton: showAvatarSkeleton.checked });
});

showRetargetingDebug.addEventListener("change", () => {
  avatar.setCalibrationDebug(showRetargetingDebug.checked);
  drawPoseDebug([]);
  updateRetargetingReadout();
});

renderMode.addEventListener("change", () => {
  const mode = renderMode.value;
  const environmentMode = mode === "environment";
  cameraStage.classList.toggle("is-environment", environmentMode);
  avatar.setRenderMode(mode);
});

function isRespiratorActive() {
  return respiratorToggle?.getAttribute("aria-pressed") === "true";
}

const mainScaleSlider = document.querySelector("#mainScaleSlider");
const liveScaleBadge = document.querySelector("#liveScaleBadge");
const stepScaleDown = document.querySelector("#stepScaleDown");
const stepScaleUp = document.querySelector("#stepScaleUp");
const quickResetScale = document.querySelector("#quickResetScale");
const presetButtons = document.querySelectorAll(".preset-btn[data-scale]");

function setAvatarScalePercent(val) {
  const clamped = Math.min(Math.max(Math.round(val), 40), 250);
  if (mainScaleSlider) mainScaleSlider.value = String(clamped);
  if (avatarSizeInput) avatarSizeInput.value = String(clamped);
  if (liveScaleBadge) liveScaleBadge.textContent = `${clamped}%`;
  if (avatarSizeValue) avatarSizeValue.value = `${clamped}%`;

  presetButtons.forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.scale) === clamped);
  });

  const depth = avatarDepthInput ? Number(avatarDepthInput.value) : 0;
  avatar.setUserCalibration({ size: clamped / 100, depth });
}

mainScaleSlider?.addEventListener("input", (e) => {
  setAvatarScalePercent(Number(e.target.value));
});

avatarSizeInput?.addEventListener("input", (e) => {
  setAvatarScalePercent(Number(e.target.value));
});

stepScaleDown?.addEventListener("click", () => {
  const current = Number(mainScaleSlider?.value || 150);
  setAvatarScalePercent(current - 5);
});

stepScaleUp?.addEventListener("click", () => {
  const current = Number(mainScaleSlider?.value || 150);
  setAvatarScalePercent(current + 5);
});

quickResetScale?.addEventListener("click", () => {
  setAvatarScalePercent(150);
});

presetButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const scale = Number(btn.dataset.scale);
    if (scale) {
      setAvatarScalePercent(scale);
    }
  });
});

function updateAvatarSliders() {
  if (!avatarSizeInput || !avatarDepthInput) return;
  const size = Number(avatarSizeInput.value) / 100;
  const depth = Number(avatarDepthInput.value);
  if (avatarSizeValue) avatarSizeValue.value = `${avatarSizeInput.value}%`;
  if (avatarDepthValue) avatarDepthValue.value = `${depth} px`;
  if (liveScaleBadge) liveScaleBadge.textContent = `${avatarSizeInput.value}%`;
  if (mainScaleSlider) mainScaleSlider.value = avatarSizeInput.value;
  avatar.setUserCalibration({ size, depth });
}

avatarDepthInput?.addEventListener("input", updateAvatarSliders);

resetAvatarSliders?.addEventListener("click", () => {
  if (avatarSizeInput) avatarSizeInput.value = "150";
  if (avatarDepthInput) avatarDepthInput.value = "0";
  setAvatarScalePercent(150);
  avatar.resetCalibration();
});

resetAvatarCalibration.addEventListener("click", () => {
  if (avatarSizeInput) avatarSizeInput.value = "150";
  if (avatarDepthInput) avatarDepthInput.value = "0";
  setAvatarScalePercent(150);
  avatar.resetCalibration();
  setPoseStatus("Avatar pose reset; position will be reacquired from tracking.", "ready");
});

updateCalibration();
updateAvatarSliders();
setAvatarScalePercent(150);

// 3D PPE items map for both Avatar AR and Model Viewer preview
const viewerPPEItems = {
  helmet: null,
  vest: null,
  gloves: [],
  respirator: [],
};

function setViewerPPEItemVisible(type, isVis) {
  const item = viewerPPEItems[type];
  if (!item) return;
  if (Array.isArray(item)) {
    item.forEach((i) => {
      if (i) i.visible = isVis;
    });
  } else {
    item.visible = isVis;
  }
}

// Initialize all 3D PPE items based on checklist state
equipmentItems.forEach((item) => {
  const eqType = item.dataset.equipment;
  if (eqType) {
    const isPressed = item.getAttribute("aria-pressed") === "true";
    avatar.setPPEItemVisible(eqType, isPressed);
    setViewerPPEItemVisible(eqType, isPressed);
  }

  item.addEventListener("click", () => {
    const isSelected = item.getAttribute("aria-pressed") === "true";
    const nextState = !isSelected;
    item.setAttribute("aria-pressed", String(nextState));

    const selectedCount = document.querySelectorAll(
      '.equipment-item[aria-pressed="true"]',
    ).length;
    checkCount.textContent = `${selectedCount} / ${equipmentItems.length}`;

    const equipmentType = item.dataset.equipment;
    if (equipmentType) {
      avatar.setPPEItemVisible(equipmentType, nextState);
      setViewerPPEItemVisible(equipmentType, nextState);
    }

    const label = item.querySelector("strong")?.textContent || equipmentType;
    if (cameraStream) {
      cameraMessage.textContent = nextState
        ? `${label} overlay enabled.`
        : `${label} overlay disabled.`;
    }
  });
});

currentDate.textContent = new Intl.DateTimeFormat(undefined, {
  weekday: "short",
  month: "short",
  day: "numeric",
}).format(new Date());

function showModelError(message, error) {
  if (error) {
    console.error("Could not load the SCBA mask model.", error);
  }
  modelStatus.textContent = "MODEL UNAVAILABLE";
  modelMessage.textContent = message;
  modelMessage.hidden = false;
}

function loadMaskModel() {
  if (!maskModelPromise) {
    maskModelPromise = new FBXLoader().loadAsync("/models/scba_mask.fbx");
  }
  return maskModelPromise;
}

async function loadFaceLandmarker() {
  if (!faceLandmarkerPromise) {
    faceLandmarkerPromise = (async () => {
      const visionFileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
      return FaceLandmarker.createFromOptions(visionFileset, {
        baseOptions: { modelAssetPath: "/models/face_landmarker.task" },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    })();
  }

  try {
    return await faceLandmarkerPromise;
  } catch (error) {
    faceLandmarkerPromise = null;
    throw error;
  }
}

function updateVideoBackground() {
  if (!maskOverlayRenderer || !cameraPreview.videoWidth || !cameraPreview.videoHeight) {
    return;
  }

  const { width, height } = cameraStage.getBoundingClientRect();
  if (width === 0 || height === 0) {
    return;
  }

  if (!videoTexture) {
    videoTexture = new THREE.VideoTexture(cameraPreview);
    videoTexture.colorSpace = THREE.SRGBColorSpace;
    videoTexture.minFilter = THREE.LinearFilter;
    videoTexture.magFilter = THREE.LinearFilter;
    videoTexture.generateMipmaps = false;

    const material = new THREE.MeshBasicMaterial({
      map: videoTexture,
      side: THREE.DoubleSide,
    });
    videoBackground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
    videoBackground.position.z = VIDEO_BACKGROUND_Z;
    videoBackground.renderOrder = -1;
    maskOverlayScene.add(videoBackground);
  }

  const videoAspect = cameraPreview.videoWidth / cameraPreview.videoHeight;
  const stageAspect = width / height;
  const repeatX = videoAspect > stageAspect ? stageAspect / videoAspect : 1;
  const repeatY = videoAspect < stageAspect ? videoAspect / stageAspect : 1;
  videoTexture.repeat.set(-repeatX, repeatY);
  videoTexture.offset.set((1 + repeatX) / 2, (1 - repeatY) / 2);

  const planeDistance = OVERLAY_CAMERA_DISTANCE - VIDEO_BACKGROUND_Z;
  const planeScale = planeDistance / OVERLAY_CAMERA_DISTANCE;
  videoBackground.geometry.dispose();
  videoBackground.geometry = new THREE.PlaneGeometry(
    width * planeScale,
    height * planeScale,
  );
  // Keep cameraStage clean so native HTML5 video feed remains visible in AR mode
  cameraStage.classList.remove("is-ar-active");
}

async function initializeMaskOverlay() {
  if (maskOverlayModel) {
    updateVideoBackground();
    return;
  }

  const model = await loadMaskModel();
  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3());
  if (!Number.isFinite(size.x) || size.x <= 0) {
    throw new Error("The SCBA mask model has no measurable width.");
  }

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
  } catch (error) {
    throw new Error("3D overlay is unavailable in this browser.", { cause: error });
  }

  const center = bounds.getCenter(new THREE.Vector3());
  maskOverlayRenderer = renderer;
  maskOverlayScene = new THREE.Scene();
  maskOverlayCamera = new THREE.PerspectiveCamera(1, 1, 1, 2000);
  maskOverlayCamera.position.z = OVERLAY_CAMERA_DISTANCE;

  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  maskOverlay.replaceChildren(renderer.domElement);

  maskOverlayScene.add(new THREE.HemisphereLight(0xffffff, 0x353535, 2.5));

  const keyLight = new THREE.DirectionalLight(0xffffff, 3);
  keyLight.position.set(1, 3, 5);
  maskOverlayScene.add(keyLight);

  const overlayContent = model.clone(true);
  overlayContent.position.sub(center);
  const faceGeometry = new THREE.BufferGeometry();
  faceGeometry.setIndex(new THREE.BufferAttribute(FACE_TESSELATION_INDICES, 1));
  faceGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(new Float32Array(FACE_VERTEX_COUNT * 3), 3),
  );
  faceOccluder = new THREE.Mesh(
    faceGeometry,
    new THREE.MeshBasicMaterial({
      colorWrite: false,
      depthWrite: true,
      side: THREE.DoubleSide,
    }),
  );
  faceOccluder.renderOrder = 0;
  faceOccluder.frustumCulled = false;
  faceOccluder.visible = false;
  maskOverlayScene.add(faceOccluder);
  // Keep the front mask visible through an opening while hiding the rear harness.
  const headGeometry = new THREE.SphereGeometry(1, 48, 32);
  const headPositions = headGeometry.getAttribute("position");
  const headSourceIndices = headGeometry.index.array;
  const headIndices = [];
  for (let index = 0; index < headSourceIndices.length; index += 3) {
    const triangle = [
      headSourceIndices[index],
      headSourceIndices[index + 1],
      headSourceIndices[index + 2],
    ];
    const center = triangle.reduce(
      (sum, vertexIndex) => sum.add(
        new THREE.Vector3().fromBufferAttribute(headPositions, vertexIndex),
      ),
      new THREE.Vector3(),
    ).multiplyScalar(1 / 3);
    const faceOpening =
      center.z > 0.25 &&
      Math.abs(center.x) < 0.5 &&
      center.y < 0.08;
    if (!faceOpening) {
      headIndices.push(...triangle);
    }
  }
  headGeometry.setIndex(headIndices);
  headOccluder = new THREE.Mesh(
    headGeometry,
    faceOccluder.material,
  );
  headOccluder.renderOrder = 0;
  headOccluder.frustumCulled = false;
  headOccluder.visible = false;
  maskOverlayScene.add(headOccluder);

  maskOverlayModel = new THREE.Group();
  maskOverlayModel.add(overlayContent);
  maskOverlayModel.scale.setScalar(1 / size.x);
  maskOverlayModel.renderOrder = 1;
  maskOverlayModel.visible = false;
  maskOverlayScene.add(maskOverlayModel);

  const resizeObserver = new ResizeObserver(() => {
    const { width, height } = cameraStage.getBoundingClientRect();
    if (width === 0 || height === 0) {
      return;
    }

    maskOverlayCamera.aspect = width / height;
    maskOverlayCamera.fov =
      (2 * Math.atan(height / (2 * OVERLAY_CAMERA_DISTANCE)) * 180) / Math.PI;
    maskOverlayCamera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    updateVideoBackground();
    renderer.render(maskOverlayScene, maskOverlayCamera);
  });
  resizeObserver.observe(cameraStage);
  updateVideoBackground();
  renderer.render(maskOverlayScene, maskOverlayCamera);
}

function mapFaceLandmark(landmark) {
  const { width, height } = cameraStage.getBoundingClientRect();
  const scale = Math.max(
    width / cameraPreview.videoWidth,
    height / cameraPreview.videoHeight,
  );
  const offsetX = (width - cameraPreview.videoWidth * scale) / 2;
  const offsetY = (height - cameraPreview.videoHeight * scale) / 2;

  return {
    x: width - (landmark.x * cameraPreview.videoWidth * scale + offsetX),
    y: landmark.y * cameraPreview.videoHeight * scale + offsetY,
  };
}

function updateFaceOccluder(
  landmarks,
  stage,
  targetDepth,
  targetWidth,
  cheekWidth,
  yaw,
  roll,
) {
  const positions = faceOccluder.geometry.getAttribute("position");
  const videoScale = Math.max(
    stage.width / cameraPreview.videoWidth,
    stage.height / cameraPreview.videoHeight,
  );
  const cropX = (stage.width - cameraPreview.videoWidth * videoScale) / 2;
  const cropY = (stage.height - cameraPreview.videoHeight * videoScale) / 2;
  const depthScale =
    cameraPreview.videoWidth * videoScale * (targetWidth / cheekWidth);

  for (let index = 0; index < positions.count; index += 1) {
    const landmark = landmarks[index];
    if (
      !landmark ||
      !Number.isFinite(landmark.x) ||
      !Number.isFinite(landmark.y) ||
      !Number.isFinite(landmark.z)
    ) {
      faceOccluder.visible = false;
      headOccluder.visible = false;
      return false;
    }

    const screenX =
      stage.width -
      (landmark.x * cameraPreview.videoWidth * videoScale + cropX);
    const screenY =
      landmark.y * cameraPreview.videoHeight * videoScale + cropY;
    const depth = targetDepth - landmark.z * depthScale;
    const perspective = (OVERLAY_CAMERA_DISTANCE - depth) / OVERLAY_CAMERA_DISTANCE;
    positions.setXYZ(
      index,
      (screenX - stage.width / 2) * perspective,
      (stage.height / 2 - screenY) * perspective,
      depth,
    );
  }

  positions.needsUpdate = true;
  faceOccluder.visible = true;

  const forehead = mapFaceLandmark(landmarks[10]);
  const chin = mapFaceLandmark(landmarks[152]);
  const faceHeight = Math.hypot(chin.x - forehead.x, chin.y - forehead.y);
  const headCenterX = (forehead.x + chin.x) / 2;
  const headCenterY = (forehead.y + chin.y) / 2 - faceHeight * 0.12;
  const headDepth = targetDepth + targetWidth * 0.45;
  const headPerspective =
    (OVERLAY_CAMERA_DISTANCE - headDepth) / OVERLAY_CAMERA_DISTANCE;
  headOccluder.position.set(
    (headCenterX - stage.width / 2) * headPerspective,
    (stage.height / 2 - headCenterY) * headPerspective,
    headDepth,
  );
  headOccluder.scale.set(
    cheekWidth * 0.75 * headPerspective,
    faceHeight * 0.78 * headPerspective,
    targetWidth * 0.38,
  );
  headOccluder.rotation.set(0, yaw, roll);
  headOccluder.visible = true;
  return true;
}

function updateMaskOverlay(landmarks) {
  // In unified AR mode, the single 3D respirator is attached to the avatar's head bone.
  // We keep maskOverlayModel hidden so only ONE respirator is rendered.
  if (maskOverlayModel) {
    maskOverlayModel.visible = false;
  }
  if (faceOccluder) {
    faceOccluder.visible = false;
  }
  if (headOccluder) {
    headOccluder.visible = false;
  }
  if (maskOverlay) {
    maskOverlay.hidden = true;
  }

  if (!landmarks || landmarks.length <= 454) {
    return false;
  }

  const leftCheek = mapFaceLandmark(landmarks[234]);
  const rightCheek = mapFaceLandmark(landmarks[454]);
  const cheekWidth = Math.hypot(
    rightCheek.x - leftCheek.x,
    rightCheek.y - leftCheek.y,
  );
  return cheekWidth >= 16;
}

let poseTrackingActive = false;
let isTrackingFrameInProgress = false;
let lastVideoFrameTime = -1;

function startPoseTrackingLoop() {
  poseTrackingActive = true;
  scheduleNextPoseFrame();
}

function stopPoseTrackingLoop() {
  poseTrackingActive = false;
  isTrackingFrameInProgress = false;
  lastVideoFrameTime = -1;
  if (trackingFrameId !== null) {
    cancelAnimationFrame(trackingFrameId);
    trackingFrameId = null;
  }
}

function scheduleNextPoseFrame() {
  if (!poseTrackingActive || !cameraStream) return;

  if ("requestVideoFrameCallback" in cameraPreview) {
    cameraPreview.requestVideoFrameCallback((now) => {
      runPoseDetectionStep(now);
      scheduleNextPoseFrame();
    });
  } else {
    trackingFrameId = requestAnimationFrame((now) => {
      runPoseDetectionStep(now);
      scheduleNextPoseFrame();
    });
  }
}

function runPoseDetectionStep(now) {
  if (
    !poseTrackingActive ||
    isTrackingFrameInProgress ||
    !poseTracker ||
    cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    return;
  }

  // Avoid running on identical duplicate video frames
  if (cameraPreview.currentTime === lastVideoFrameTime) {
    return;
  }
  lastVideoFrameTime = cameraPreview.currentTime;

  isTrackingFrameInProgress = true;
  try {
    updatePoseFrame(now);
  } catch (err) {
    console.error("Pose tracking update error:", err);
  } finally {
    isTrackingFrameInProgress = false;
  }
}

function initModelViewer() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
  } catch (error) {
    showModelError("3D preview is unavailable in this browser.", error);
    return;
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#141b1c");
  modelCamera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  modelCamera.position.set(0, 0, 4);

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  modelViewer.replaceChildren(renderer.domElement, modelMessage);

  scene.add(new THREE.HemisphereLight(0xe4f2dc, 0x26302f, 2.2));

  const keyLight = new THREE.DirectionalLight(0xffffff, 3);
  keyLight.position.set(3, 5, 6);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xc5f36b, 1.2);
  fillLight.position.set(-4, 1, -3);
  scene.add(fillLight);

  modelControls = new OrbitControls(modelCamera, renderer.domElement);
  modelControls.enableDamping = true;
  modelControls.target.set(0, 0, 0);

  const resizeObserver = new ResizeObserver(() => {
    const { width, height } = modelViewer.getBoundingClientRect();
    if (width === 0 || height === 0) {
      return;
    }

    modelCamera.aspect = width / height;
    modelCamera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  });
  resizeObserver.observe(modelViewer);

  function resetView() {
    modelCamera.position.set(0, 0, 4);
    modelControls.target.set(0, 0, 0);
    modelControls.update();
  }

  resetModelView.addEventListener("click", resetView);

  loadMaskModel()
    .then((model) => {
      const bounds = new THREE.Box3().setFromObject(model);
      const size = bounds.getSize(new THREE.Vector3());
      const largestDimension = Math.max(size.x, size.y, size.z);

      if (!Number.isFinite(largestDimension) || largestDimension <= 0) {
        throw new Error("The FBX file contains no visible 3D geometry.");
      }

      const center = bounds.getCenter(new THREE.Vector3());
      const viewerModel = new THREE.Group();

      // Safety Helmet (top)
      const viewerHelmet = createSafetyHelmet();
      viewerHelmet.scale.setScalar(2.0);
      viewerHelmet.position.set(0, 0.72, 0);
      viewerHelmet.rotation.y = -Math.PI * 0.12;
      viewerModel.add(viewerHelmet);

      // SCBA Face Mask (center-left)
      const viewerMask = new THREE.Group();
      const viewerContent = model.clone(true);
      viewerContent.position.sub(center);
      viewerMask.add(viewerContent);
      viewerMask.scale.setScalar(1.5 / largestDimension);
      viewerMask.position.set(-0.72, 0.05, 0);
      viewerModel.add(viewerMask);

      // SCBA Back Cylinder Apparatus (center-right)
      const viewerCylinder = createSCBACylinderApparatus();
      viewerCylinder.scale.setScalar(1.9);
      viewerCylinder.position.set(0.72, 0.05, 0);
      viewerCylinder.rotation.y = Math.PI * 0.85;
      viewerModel.add(viewerCylinder);

      // High-Visibility Safety Vest (center)
      const viewerVest = createSafetyVest();
      viewerVest.scale.setScalar(1.5);
      viewerVest.position.set(0, -0.28, 0);
      viewerVest.rotation.y = Math.PI * 0.08;
      viewerModel.add(viewerVest);

      // Protective Gloves (bottom left & right)
      const viewerGloveL = createProtectiveGloves("left");
      viewerGloveL.scale.setScalar(2.6);
      viewerGloveL.position.set(-0.9, -0.68, 0);
      viewerGloveL.rotation.y = Math.PI * 0.25;
      viewerModel.add(viewerGloveL);

      const viewerGloveR = createProtectiveGloves("right");
      viewerGloveR.scale.setScalar(2.6);
      viewerGloveR.position.set(0.9, -0.68, 0);
      viewerGloveR.rotation.y = -Math.PI * 0.25;
      viewerModel.add(viewerGloveR);

      viewerPPEItems.helmet = viewerHelmet;
      viewerPPEItems.vest = viewerVest;
      viewerPPEItems.gloves = [viewerGloveL, viewerGloveR];
      viewerPPEItems.respirator = [viewerMask, viewerCylinder];

      // Synchronize with current checklist selection state
      equipmentItems.forEach((item) => {
        const eqType = item.dataset.equipment;
        if (eqType) {
          const isPressed = item.getAttribute("aria-pressed") === "true";
          setViewerPPEItemVisible(eqType, isPressed);
        }
      });

      scene.add(viewerModel);

      modelStatus.textContent = "PPE GEAR LOADED";
      modelMessage.hidden = true;
      resetModelView.disabled = false;
      resetView();
    })
    .catch((error) => {
      showModelError(
        "Could not load models/scba_mask.fbx. Check that the file is present and valid.",
        error,
      );
    });

  function render() {
    modelControls.update();
    renderer.render(scene, modelCamera);
    requestAnimationFrame(render);
  }
  render();
}

initModelViewer();
renderMode.dispatchEvent(new Event("change"));
window.addEventListener("pagehide", () => {
  stopCamera();
  faceLandmarker?.close();
  poseTracker?.close();
  avatar.close();
});
