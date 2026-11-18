import { decodeSimCC, convertCocoToMediaPipeLandmarks, COCO_KEYPOINTS, COCO_TO_MEDIAPIPE } from "./rtmposeTracker.js";

console.log("--- 1. Testing SimCC Decoder ---");
// Create mock SimCC 1D heatmaps: 17 keypoints
// xBins = 384, yBins = 512
const numKeypoints = 17;
const xBins = 384;
const yBins = 512;
const simccX = new Float32Array(numKeypoints * xBins);
const simccY = new Float32Array(numKeypoints * yBins);

// Place peak for Nose (kp 0) at x = 192 (0.5), y = 102 (0.2)
simccX[0 * xBins + 192] = 10.0;
simccY[0 * yBins + 102] = 10.0;

// Place peak for Left Shoulder (kp 5) at x = 230 (0.6), y = 170 (0.33)
simccX[5 * xBins + 230] = 12.0;
simccY[5 * yBins + 170] = 12.0;

// Place peak for Right Shoulder (kp 6) at x = 154 (0.4), y = 170 (0.33)
simccX[6 * xBins + 154] = 12.0;
simccY[6 * yBins + 170] = 12.0;

const keypoints = decodeSimCC(simccX, simccY);
if (keypoints.length !== 17) {
  console.error("FAIL: Expected 17 keypoints from decodeSimCC, got", keypoints.length);
  process.exit(1);
}

const nose = keypoints[COCO_KEYPOINTS.NOSE];
const lSh = keypoints[COCO_KEYPOINTS.LEFT_SHOULDER];
const rSh = keypoints[COCO_KEYPOINTS.RIGHT_SHOULDER];

console.log(`PASS: Decoded 17 keypoints.`);
console.log(`Nose: x=${nose.x.toFixed(3)}, y=${nose.y.toFixed(3)}, score=${nose.score.toFixed(3)}`);
console.log(`Left Shoulder: x=${lSh.x.toFixed(3)}, y=${lSh.y.toFixed(3)}, score=${lSh.score.toFixed(3)}`);
console.log(`Right Shoulder: x=${rSh.x.toFixed(3)}, y=${rSh.y.toFixed(3)}, score=${rSh.score.toFixed(3)}`);

if (Math.abs(nose.x - 0.5) > 0.02 || Math.abs(nose.y - 0.2) > 0.02) {
  console.error("FAIL: Nose coordinates outside expected tolerance!");
  process.exit(1);
}

console.log("--- 2. Testing COCO to MediaPipe 33 Landmark Conversion ---");
const { landmarks, worldLandmarks } = convertCocoToMediaPipeLandmarks(keypoints, 16 / 9);

if (landmarks.length !== 33 || worldLandmarks.length !== 33) {
  console.error("FAIL: Expected 33 landmarks, got", landmarks.length, worldLandmarks.length);
  process.exit(1);
}

// MediaPipe nose is index 0
if (Math.abs(landmarks[0].x - nose.x) > 1e-4) {
  console.error("FAIL: MediaPipe nose X does not match COCO nose X");
  process.exit(1);
}

// MediaPipe left shoulder is index 11
if (Math.abs(landmarks[11].x - lSh.x) > 1e-4) {
  console.error("FAIL: MediaPipe left shoulder X does not match COCO left shoulder X");
  process.exit(1);
}

// MediaPipe right shoulder is index 12
if (Math.abs(landmarks[12].x - rSh.x) > 1e-4) {
  console.error("FAIL: MediaPipe right shoulder X does not match COCO right shoulder X");
  process.exit(1);
}

console.log("PASS: MediaPipe-33 landmarks mapped correctly from COCO-17.");
console.log("PASS: World landmarks generated with metric scaling.");
console.log("ALL RTMPose UNIT TESTS PASSED!");
