import {
  convertRtmw3dToMediaPipe,
  COCO17_KEYPOINTS,
  COCO_TO_MEDIAPIPE,
  COCO17_CONNECTIONS,
} from "./src/rtmw3dTracker.js";

console.log("=== RTMW3D 3D Pose Tracking Verification Suite ===");

function createMockPerson(options = {}) {
  const k3d = Array.from({ length: 133 }, () => [0, 0, 0]);
  const k2d = Array.from({ length: 133 }, () => [320, 240]);
  const scores = Array.from({ length: 133 }, () => 0.95);

  // Default neutral standing
  // Shoulders (span = 80px -> ~0.40m)
  k2d[COCO17_KEYPOINTS.LEFT_SHOULDER] = [360, 160];
  k2d[COCO17_KEYPOINTS.RIGHT_SHOULDER] = [280, 160];
  // Hips
  k2d[COCO17_KEYPOINTS.LEFT_HIP] = [345, 280];
  k2d[COCO17_KEYPOINTS.RIGHT_HIP] = [295, 280];
  // Elbows (down by side)
  k2d[COCO17_KEYPOINTS.LEFT_ELBOW] = [365, 220];
  k2d[COCO17_KEYPOINTS.RIGHT_ELBOW] = [275, 220];
  // Wrists (down by hips)
  k2d[COCO17_KEYPOINTS.LEFT_WRIST] = [365, 275];
  k2d[COCO17_KEYPOINTS.RIGHT_WRIST] = [275, 275];
  // Knees
  k2d[COCO17_KEYPOINTS.LEFT_KNEE] = [345, 360];
  k2d[COCO17_KEYPOINTS.RIGHT_KNEE] = [295, 360];
  // Ankles
  k2d[COCO17_KEYPOINTS.LEFT_ANKLE] = [345, 440];
  k2d[COCO17_KEYPOINTS.RIGHT_ANKLE] = [295, 440];
  // Head
  k2d[COCO17_KEYPOINTS.NOSE] = [320, 115];
  k2d[COCO17_KEYPOINTS.LEFT_EYE] = [328, 108];
  k2d[COCO17_KEYPOINTS.RIGHT_EYE] = [312, 108];
  k2d[COCO17_KEYPOINTS.LEFT_EAR] = [338, 112];
  k2d[COCO17_KEYPOINTS.RIGHT_EAR] = [302, 112];

  // 3D coordinates (root-relative in meters)
  k3d[COCO17_KEYPOINTS.LEFT_SHOULDER] = [0.20, -0.32, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_SHOULDER] = [-0.20, -0.32, 0];
  k3d[COCO17_KEYPOINTS.LEFT_ELBOW] = [0.22, -0.05, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_ELBOW] = [-0.22, -0.05, 0];
  k3d[COCO17_KEYPOINTS.LEFT_WRIST] = [0.22, 0.20, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_WRIST] = [-0.22, 0.20, 0];
  k3d[COCO17_KEYPOINTS.LEFT_HIP] = [0.12, 0.0, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_HIP] = [-0.12, 0.0, 0];
  k3d[COCO17_KEYPOINTS.LEFT_KNEE] = [0.12, 0.40, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_KNEE] = [-0.12, 0.40, 0];
  k3d[COCO17_KEYPOINTS.LEFT_ANKLE] = [0.12, 0.80, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_ANKLE] = [-0.12, 0.80, 0];
  k3d[COCO17_KEYPOINTS.NOSE] = [0.0, -0.55, -0.05];
  k3d[COCO17_KEYPOINTS.LEFT_EAR] = [0.09, -0.55, 0];
  k3d[COCO17_KEYPOINTS.RIGHT_EAR] = [-0.09, -0.55, 0];

  // Apply scenario overrides
  if (options.leftArmRaised) {
    k2d[COCO17_KEYPOINTS.LEFT_ELBOW] = [360, 100];
    k2d[COCO17_KEYPOINTS.LEFT_WRIST] = [360, 40];
    k3d[COCO17_KEYPOINTS.LEFT_ELBOW] = [0.22, -0.55, 0];
    k3d[COCO17_KEYPOINTS.LEFT_WRIST] = [0.22, -0.80, 0];
  }
  if (options.rightArmRaised) {
    k2d[COCO17_KEYPOINTS.RIGHT_ELBOW] = [280, 100];
    k2d[COCO17_KEYPOINTS.RIGHT_WRIST] = [280, 40];
    k3d[COCO17_KEYPOINTS.RIGHT_ELBOW] = [-0.22, -0.55, 0];
    k3d[COCO17_KEYPOINTS.RIGHT_WRIST] = [-0.22, -0.80, 0];
  }
  if (options.elbowBending) {
    // Left elbow bent forward (z negative in camera-relative coords)
    k2d[COCO17_KEYPOINTS.LEFT_WRIST] = [360, 210];
    k3d[COCO17_KEYPOINTS.LEFT_WRIST] = [0.22, -0.05, -0.25];
  }
  if (options.headTurnLeft) {
    k2d[COCO17_KEYPOINTS.NOSE] = [340, 115];
    k3d[COCO17_KEYPOINTS.NOSE] = [0.08, -0.55, -0.03];
  }
  if (options.torsoTurnLeft) {
    k2d[COCO17_KEYPOINTS.LEFT_SHOULDER] = [350, 160];
    k2d[COCO17_KEYPOINTS.RIGHT_SHOULDER] = [290, 160];
    k3d[COCO17_KEYPOINTS.LEFT_SHOULDER] = [0.15, -0.32, 0.15];
    k3d[COCO17_KEYPOINTS.RIGHT_SHOULDER] = [-0.15, -0.32, -0.15];
  }
  if (options.sideProfile) {
    // Right shoulder and right limbs partially occluded
    scores[COCO17_KEYPOINTS.RIGHT_SHOULDER] = 0.25;
    scores[COCO17_KEYPOINTS.RIGHT_ELBOW] = 0.15;
    scores[COCO17_KEYPOINTS.RIGHT_WRIST] = 0.10;
  }
  if (options.partialOcclusion) {
    scores[COCO17_KEYPOINTS.LEFT_WRIST] = 0.05;
    scores[COCO17_KEYPOINTS.RIGHT_WRIST] = 0.05;
  }

  return {
    keypoints: [k3d],
    scores: [scores],
    keypoints2d: [k2d],
    stats: {
      personCount: 1,
      detTime: 12,
      poseTime: 22,
      totalTime: 34,
    },
  };
}

// 1. Neutral standing position
console.log("--- 1. Testing Neutral Standing Position ---");
const neutralResult = convertRtmw3dToMediaPipe(createMockPerson(), 640, 480);
const lUpperArmVecY = neutralResult.worldLandmarks[13].y - neutralResult.worldLandmarks[11].y;
if (lUpperArmVecY <= 0) {
  console.error("FAIL: In neutral pose, elbow should be below shoulder (positive Y in MediaPipe coords)!");
  process.exit(1);
}
console.log(`PASS: Neutral arm hangs downward (delta Y = +${lUpperArmVecY.toFixed(3)}m).`);

// 2. Left arm raised
console.log("--- 2. Testing Left Arm Raised ---");
const leftRaisedResult = convertRtmw3dToMediaPipe(createMockPerson({ leftArmRaised: true }), 640, 480);
const lRaisedVecY = leftRaisedResult.worldLandmarks[13].y - leftRaisedResult.worldLandmarks[11].y;
if (lRaisedVecY >= 0) {
  console.error("FAIL: Raised arm should have elbow above shoulder (negative Y in MediaPipe coords)!");
  process.exit(1);
}
console.log(`PASS: Left arm raises upward (delta Y = ${lRaisedVecY.toFixed(3)}m).`);

// 3. Right arm raised
console.log("--- 3. Testing Right Arm Raised ---");
const rightRaisedResult = convertRtmw3dToMediaPipe(createMockPerson({ rightArmRaised: true }), 640, 480);
const rRaisedVecY = rightRaisedResult.worldLandmarks[14].y - rightRaisedResult.worldLandmarks[12].y;
if (rRaisedVecY >= 0) {
  console.error("FAIL: Raised right arm should have elbow above shoulder!");
  process.exit(1);
}
console.log(`PASS: Right arm raises upward (delta Y = ${rRaisedVecY.toFixed(3)}m).`);

// 4. Both arms raised
console.log("--- 4. Testing Both Arms Raised ---");
const bothRaisedResult = convertRtmw3dToMediaPipe(createMockPerson({ leftArmRaised: true, rightArmRaised: true }), 640, 480);
if (bothRaisedResult.worldLandmarks[13].y >= bothRaisedResult.worldLandmarks[11].y ||
    bothRaisedResult.worldLandmarks[14].y >= bothRaisedResult.worldLandmarks[12].y) {
  console.error("FAIL: Both arms must be raised!");
  process.exit(1);
}
console.log("PASS: Both arms raised simultaneously.");

// 5. Elbow bending
console.log("--- 5. Testing Elbow Bending ---");
const bentResult = convertRtmw3dToMediaPipe(createMockPerson({ elbowBending: true }), 640, 480);
const wristDepthZ = bentResult.worldLandmarks[15].z;
if (wristDepthZ >= 0) {
  console.error("FAIL: Bent forward wrist should have negative depth Z relative to root, got", wristDepthZ);
  process.exit(1);
}
console.log(`PASS: Elbow bends forward (wrist metric depth Z = ${wristDepthZ.toFixed(3)}m).`);

// 6. Head turning
console.log("--- 6. Testing Head Turning ---");
const headTurnResult = convertRtmw3dToMediaPipe(createMockPerson({ headTurnLeft: true }), 640, 480);
const noseOffsetX = headTurnResult.worldLandmarks[0].x;
if (noseOffsetX <= 0) {
  console.error("FAIL: Head turning left should offset nose X towards positive!");
  process.exit(1);
}
console.log(`PASS: Head turn left detected (nose lateral X = +${noseOffsetX.toFixed(3)}m).`);

// 7. Torso turning
console.log("--- 7. Testing Torso Turning ---");
const torsoTurnResult = convertRtmw3dToMediaPipe(createMockPerson({ torsoTurnLeft: true }), 640, 480);
const lShZ = torsoTurnResult.worldLandmarks[11].z;
const rShZ = torsoTurnResult.worldLandmarks[12].z;
if (lShZ <= rShZ) {
  console.error("FAIL: Torso turning left must make left shoulder deeper (+Z) than right shoulder (-Z)!");
  process.exit(1);
}
console.log(`PASS: Torso yaw detected (L Shoulder Z: +${lShZ.toFixed(2)}m, R Shoulder Z: ${rShZ.toFixed(2)}m).`);

// 8. Side profile
console.log("--- 8. Testing Side Profile ---");
const sideProfileResult = convertRtmw3dToMediaPipe(createMockPerson({ sideProfile: true }), 640, 480);
if (sideProfileResult.worldLandmarks[16].visibility >= 0.20) {
  console.error("FAIL: Occluded far arm should have low visibility in side profile!");
  process.exit(1);
}
console.log("PASS: Side profile correctly demotes occluded joints.");

// 9. Partial body occlusion
console.log("--- 9. Testing Partial Body Occlusion ---");
const occlusionResult = convertRtmw3dToMediaPipe(createMockPerson({ partialOcclusion: true }), 640, 480);
if (occlusionResult.landmarks[15].visibility >= 0.25 || occlusionResult.landmarks[16].visibility >= 0.25) {
  console.error("FAIL: Occluded joints must have low visibility score!");
  process.exit(1);
}
console.log("PASS: Partial occlusion filtered out unreliable joints.");

// 10. Tracking loss and reacquisition
console.log("--- 10. Testing Tracking Loss and Reacquisition ---");
const lostResult = convertRtmw3dToMediaPipe({ keypoints: [] }, 640, 480);
if (lostResult.landmarks !== null || lostResult.worldLandmarks !== null) {
  console.error("FAIL: Tracking loss must yield null landmarks!");
  process.exit(1);
}
console.log("PASS: Tracking loss correctly yields null landmarks.");
const reacqResult = convertRtmw3dToMediaPipe(createMockPerson(), 640, 480);
if (!reacqResult.landmarks || !reacqResult.worldLandmarks) {
  console.error("FAIL: Reacquisition must yield valid landmarks!");
  process.exit(1);
}
console.log("PASS: Reacquisition restores full 3D landmarks immediately.");

console.log("\nALL 10 TEST CASES VERIFIED SUCCESSFULLY FOR RTMW3D!");
