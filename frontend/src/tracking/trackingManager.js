/**
 * Unified Tracking Engine Manager
 * Coordinates all 4 tracking & reconstruction engines for the AI Intelligent Safety Mirror:
 * - Engine A: MediaPipe 3D Pose
 * - Engine B: RTMW3D (SimCC-3D)
 * - Engine C: TensorFlow BodyPix (24-part segmentation)
 * - Engine D: EMOSH-Tracker (Full 3D human body mesh reconstruction)
 *
 * Implements independent scheduling, visual debug modes (1-6), and unified telemetry.
 */

export const TRACKING_MODES = {
  MEDIAPIPE: "mediapipe",
  RTMW3D: "rtmw3d",
  BODYPIX: "bodypix",
  MEDIAPIPE_BODYPIX: "mediapipe+bodypix",
  RTMW3D_BODYPIX: "rtmw3d+bodypix",
  EMOSH: "emosh",
  COMBINED: "combined",
};

export const VISUAL_DEBUG_MODES = {
  CAMERA_ONLY: 1,       // Mode 1: Original webcam feed only
  MEDIAPIPE_SKELETON: 2,// Mode 2: MediaPipe skeleton landmarks
  RTMW3D_SKELETON: 3,   // Mode 3: RTMW3D 3D skeleton & joints
  BODYPIX_SEGMENT: 4,   // Mode 4: Coloured body-part segmentation overlay
  EMOSH_RECON: 5,       // Mode 5: EMOSH reconstructed 3D human body surface mesh
  COMBINED_VIEW: 6,     // Mode 6: All compatible overlays active
};

export class TrackingManager {
  constructor({
    mediaPipeTracker,
    rtmw3dTracker,
    bodyPartEngine,
    emoshClient,
    bodyMeshViewer,
  } = {}) {
    this.engines = {
      mediaPipe: mediaPipeTracker || null,
      rtmw3d: rtmw3dTracker || null,
      bodyPix: bodyPartEngine || null,
      emosh: emoshClient || null,
    };
    this.bodyMeshViewer = bodyMeshViewer || null;

    this.activeMode = TRACKING_MODES.MEDIAPIPE;
    this.visualMode = VISUAL_DEBUG_MODES.COMBINED_VIEW;

    // Independent enablement flags
    this.enabledEngines = {
      mediaPipe: true,
      rtmw3d: false,
      bodyPix: true,
      emosh: false,
    };

    // Telemetry & diagnostics cache
    this.diagnostics = {
      activeEngine: "MediaPipe",
      activeMode: this.activeMode,
      visualMode: this.visualMode,
      mediaPipeFps: 0,
      rtmw3dFps: 0,
      bodyPixFps: 0,
      emoshStatus: "idle",
      emoshProcessingTime: 0,
      gpuAvailable: false,
      gpuDevice: "N/A",
      gpuMemoryTotalMb: 0,
      gpuMemoryFreeMb: 0,
      personDetected: false,
      landmarkConfidence: 0.0,
      segmentationAvailable: false,
      viewOrientation: "front",
    };

    this.listeners = new Set();
  }

  onDiagnostics(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notifyDiagnostics() {
    for (const listener of this.listeners) {
      try {
        listener(this.diagnostics);
      } catch (err) {
        console.error("TrackingManager listener error:", err);
      }
    }
  }

  /**
   * Sets active tracking mode and configures independent engine flags.
   */
  setMode(mode) {
    this.activeMode = mode;
    this.diagnostics.activeMode = mode;

    switch (mode) {
      case TRACKING_MODES.MEDIAPIPE:
        this.enabledEngines.mediaPipe = true;
        this.enabledEngines.rtmw3d = false;
        this.enabledEngines.bodyPix = false;
        this.enabledEngines.emosh = false;
        this.diagnostics.activeEngine = "MediaPipe";
        break;

      case TRACKING_MODES.RTMW3D:
        this.enabledEngines.mediaPipe = false;
        this.enabledEngines.rtmw3d = true;
        this.enabledEngines.bodyPix = false;
        this.enabledEngines.emosh = false;
        this.diagnostics.activeEngine = "RTMW3D (SimCC-3D)";
        break;

      case TRACKING_MODES.BODYPIX:
        this.enabledEngines.mediaPipe = false;
        this.enabledEngines.rtmw3d = false;
        this.enabledEngines.bodyPix = true;
        this.enabledEngines.emosh = false;
        this.diagnostics.activeEngine = "TensorFlow BodyPix";
        break;

      case TRACKING_MODES.MEDIAPIPE_BODYPIX:
        this.enabledEngines.mediaPipe = true;
        this.enabledEngines.rtmw3d = false;
        this.enabledEngines.bodyPix = true;
        this.enabledEngines.emosh = false;
        this.diagnostics.activeEngine = "MediaPipe + BodyPix";
        break;

      case TRACKING_MODES.RTMW3D_BODYPIX:
        this.enabledEngines.mediaPipe = false;
        this.enabledEngines.rtmw3d = true;
        this.enabledEngines.bodyPix = true;
        this.enabledEngines.emosh = false;
        this.diagnostics.activeEngine = "RTMW3D + BodyPix";
        break;

      case TRACKING_MODES.EMOSH:
        this.enabledEngines.mediaPipe = false;
        this.enabledEngines.rtmw3d = false;
        this.enabledEngines.bodyPix = false;
        this.enabledEngines.emosh = true;
        this.diagnostics.activeEngine = "EMOSH-Tracker (Reconstruction)";
        if (this.bodyMeshViewer) {
          this.bodyMeshViewer.setVisible(true);
          this.bodyMeshViewer.play();
        }
        break;

      case TRACKING_MODES.COMBINED:
      default:
        this.enabledEngines.mediaPipe = true;
        this.enabledEngines.rtmw3d = true;
        this.enabledEngines.bodyPix = true;
        this.enabledEngines.emosh = true;
        this.diagnostics.activeEngine = "Combined Multi-Engine Mode";
        break;
    }

    if (mode !== TRACKING_MODES.EMOSH && this.bodyMeshViewer && this.visualMode !== VISUAL_DEBUG_MODES.EMOSH_RECON) {
      this.bodyMeshViewer.setVisible(false);
      this.bodyMeshViewer.pause();
    }

    this.notifyDiagnostics();
  }

  /**
   * Sets visual debugging mode (Modes 1 to 6).
   */
  setVisualMode(modeNumber) {
    this.visualMode = Number(modeNumber);
    this.diagnostics.visualMode = this.visualMode;

    if (this.bodyMeshViewer) {
      if (this.visualMode === VISUAL_DEBUG_MODES.EMOSH_RECON || this.visualMode === VISUAL_DEBUG_MODES.COMBINED_VIEW) {
        this.bodyMeshViewer.setVisible(this.enabledEngines.emosh || this.visualMode === VISUAL_DEBUG_MODES.EMOSH_RECON);
      } else {
        this.bodyMeshViewer.setVisible(false);
      }
    }

    this.notifyDiagnostics();
  }

  /**
   * Updates telemetry measurements from live tracking pipelines.
   */
  updateTelemetry(telemetry = {}) {
    Object.assign(this.diagnostics, telemetry);
    this.notifyDiagnostics();
  }
}
