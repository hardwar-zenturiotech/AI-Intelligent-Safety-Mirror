/**
 * EMOSH Integration & Unified Tracking Manager Test Suite
 */

import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { EMOSHClient } from "./src/tracking/emoshClient.js";
import {
  TrackingManager,
  TRACKING_MODES,
  VISUAL_DEBUG_MODES,
} from "./src/tracking/trackingManager.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let passed = true;
function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    passed = false;
  } else {
    console.log(`PASS: ${msg}`);
  }
}

console.log("=== EMOSH-Tracker & Unified Tracking Manager Test Suite ===");

// 1. Verify EMOSHClient defaults
const client = new EMOSHClient({ serviceUrl: "http://127.0.0.1:8000" });
assert(client.serviceUrl === "http://127.0.0.1:8000", "EMOSHClient service URL configured.");
assert(client.status === "idle", "EMOSHClient initial status is idle.");

// 2. Verify TrackingManager Engine Modes
const manager = new TrackingManager({ emoshClient: client });
assert(manager.activeMode === TRACKING_MODES.MEDIAPIPE, "Default active mode is MediaPipe.");
assert(manager.enabledEngines.mediaPipe === true, "MediaPipe enabled in default mode.");
assert(manager.enabledEngines.emosh === false, "EMOSH disabled in default mode.");

// Mode: RTMW3D
manager.setMode(TRACKING_MODES.RTMW3D);
assert(manager.enabledEngines.rtmw3d === true && !manager.enabledEngines.mediaPipe, "Mode RTMW3D correctly switches active engine.");

// Mode: BodyPix
manager.setMode(TRACKING_MODES.BODYPIX);
assert(manager.enabledEngines.bodyPix === true && !manager.enabledEngines.rtmw3d, "Mode BodyPix correctly switches active engine.");

// Mode: MediaPipe + BodyPix
manager.setMode(TRACKING_MODES.MEDIAPIPE_BODYPIX);
assert(manager.enabledEngines.mediaPipe === true && manager.enabledEngines.bodyPix === true, "Mode MediaPipe + BodyPix enables both engines.");

// Mode: RTMW3D + BodyPix
manager.setMode(TRACKING_MODES.RTMW3D_BODYPIX);
assert(manager.enabledEngines.rtmw3d === true && manager.enabledEngines.bodyPix === true, "Mode RTMW3D + BodyPix enables both engines.");

// Mode: EMOSH Reconstruction
manager.setMode(TRACKING_MODES.EMOSH);
assert(manager.enabledEngines.emosh === true && !manager.enabledEngines.mediaPipe, "Mode EMOSH switches to reconstruction engine.");

// Mode: Combined
manager.setMode(TRACKING_MODES.COMBINED);
assert(
  manager.enabledEngines.mediaPipe && manager.enabledEngines.rtmw3d && manager.enabledEngines.bodyPix && manager.enabledEngines.emosh,
  "Mode Combined enables all compatible engines simultaneously.",
);

// 3. Verify Visual Debugging Modes (Modes 1 to 6)
assert(VISUAL_DEBUG_MODES.CAMERA_ONLY === 1, "Mode 1 is Camera Only.");
assert(VISUAL_DEBUG_MODES.MEDIAPIPE_SKELETON === 2, "Mode 2 is MediaPipe Skeleton.");
assert(VISUAL_DEBUG_MODES.RTMW3D_SKELETON === 3, "Mode 3 is RTMW3D 3D Skeleton.");
assert(VISUAL_DEBUG_MODES.BODYPIX_SEGMENT === 4, "Mode 4 is BodyPix Segmentation.");
assert(VISUAL_DEBUG_MODES.EMOSH_RECON === 5, "Mode 5 is EMOSH 3D Reconstruction.");
assert(VISUAL_DEBUG_MODES.COMBINED_VIEW === 6, "Mode 6 is Combined View.");

manager.setVisualMode(VISUAL_DEBUG_MODES.EMOSH_RECON);
assert(manager.visualMode === 5, "setVisualMode(5) correctly updates visual debug mode.");

// 4. Verify EMOSH Output Artifacts & Three.js Mesh Sequence Contract
const sampleMeshPath = resolve(__dirname, "public/models/emosh_sample.json");
assert(existsSync(sampleMeshPath), "EMOSH sample mesh exists in frontend public bundle.");

const sampleData = JSON.parse(readFileSync(sampleMeshPath, "utf-8"));
assert(sampleData.engine === "EMOSH-Tracker", "Mesh sequence engine is EMOSH-Tracker.");
assert(sampleData.totalFrames > 0, `Mesh sequence contains ${sampleData.totalFrames} frames.`);
assert(sampleData.vertexCount > 0, `Humanoid mesh contains ${sampleData.vertexCount} vertices.`);
assert(sampleData.faceCount > 0, `Humanoid topology contains ${sampleData.faceCount} triangular faces.`);
assert(Array.isArray(sampleData.indices) && sampleData.indices.length === sampleData.faceCount * 3, "Indices array matches face topology.");

// Verify Frame 0 Schema
const frame0 = sampleData.frames[0];
assert(frame0.frameIndex === 0, "Frame 0 index valid.");
assert(Array.isArray(frame0.vertices) && frame0.vertices.length === sampleData.vertexCount * 3, "Frame vertices buffer matches vertexCount * 3 ($x,y,z$).");
assert(Array.isArray(frame0.normals) && frame0.normals.length === sampleData.vertexCount * 3, "Frame normals buffer matches vertexCount * 3 ($nx,ny,nz$).");

// Verify Anatomical SCBA Joint Anchors
assert(frame0.joints && typeof frame0.joints === "object", "Joint anchors present in frame data.");
assert(Array.isArray(frame0.joints.head) && frame0.joints.head.length === 3, "Head anchor ($x,y,z$) present for SCBA helmet & mask.");
assert(Array.isArray(frame0.joints.chest) && frame0.joints.chest.length === 3, "Chest anchor ($x,y,z$) present for harness & back cylinder.");
assert(Array.isArray(frame0.joints.leftHand) && frame0.joints.leftHand.length === 3, "Left hand anchor ($x,y,z$) present for gauntlet gloves.");
assert(Array.isArray(frame0.joints.rightHand) && frame0.joints.rightHand.length === 3, "Right hand anchor ($x,y,z$) present for gauntlet gloves.");

// 5. Verify Metadata Structure
const metaPath = resolve(__dirname, "../emosh_service/outputs/sample_test/metadata.json");
if (existsSync(metaPath)) {
  const meta = JSON.parse(readFileSync(metaPath, "utf-8"));
  assert(meta.engine === "EMOSH-Tracker", "Metadata engine tag verified.");
  assert(meta.device !== undefined, `Metadata device detected: ${meta.device}`);
  assert(meta.models_status !== undefined, "Models licensing status documented in metadata.");
  assert(meta.models_status.smplx && meta.models_status.smplx.license_url, "SMPL-X MPI registration URL documented.");
  assert(meta.models_status.flame && meta.models_status.flame.license_url, "FLAME MPI registration URL documented.");
  assert(meta.models_status.mano && meta.models_status.mano.license_url, "MANO MPI registration URL documented.");
}

if (!passed) {
  process.exit(1);
}
console.log("ALL EMOSH & TRACKING MANAGER UNIT TESTS PASSED!");
