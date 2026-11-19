import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import fs from "fs";
import { MotionRetargeter, inspectHumanoidRig, convertMediaPipeWorldVector } from "./retargeting.js";
import { createSyntheticPose } from "./poseTracker.js";

const buffer = fs.readFileSync("../models/avatar.glb");
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

const loader = new GLTFLoader();
loader.parse(arrayBuffer, "", (gltf) => {
  const scene = gltf.scene;
  scene.updateMatrixWorld(true);
  const rig = inspectHumanoidRig(scene);
  const retargeter = new MotionRetargeter(rig.mapping, scene, rig.anchors);

  const getDir = (b1, b2) => {
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();
    b1.getWorldPosition(v1);
    b2.getWorldPosition(v2);
    return v2.sub(v1).normalize();
  };

  let allPassed = true;
  function assert(condition, message) {
    if (!condition) {
      console.error(`FAIL: ${message}`);
      allPassed = false;
    } else {
      console.log(`PASS: ${message}`);
    }
  }

  console.log("--- 1. Neutral Upright Pose ---");
  const neutral = createSyntheticPose(0);
  retargeter.update(neutral.landmarks, neutral.worldLandmarks, 1);
  const lUpper0 = getDir(rig.mapping.leftUpperArm, rig.mapping.leftForearm);
  const rUpper0 = getDir(rig.mapping.rightUpperArm, rig.mapping.rightForearm);
  assert(lUpper0.y < -0.6, `Left upper arm points down in neutral (y=${lUpper0.y.toFixed(3)})`);
  assert(rUpper0.y < -0.6, `Right upper arm points down in neutral (y=${rUpper0.y.toFixed(3)})`);
  assert(Math.abs(retargeter.diagnostics.torsoYaw) < 0.1, `Torso yaw near zero (${retargeter.diagnostics.torsoYaw.toFixed(3)})`);
  assert(Math.abs(retargeter.diagnostics.headYaw) < 0.1, `Head yaw near zero (${retargeter.diagnostics.headYaw.toFixed(3)})`);

  console.log("--- 2. Raise Left Arm Only ---");
  const p1 = createSyntheticPose(2500);
  retargeter.update(p1.landmarks, p1.worldLandmarks, 1);
  const lUpper1 = getDir(rig.mapping.leftUpperArm, rig.mapping.leftForearm);
  const rUpper1 = getDir(rig.mapping.rightUpperArm, rig.mapping.rightForearm);
  assert(lUpper1.y > 0.2, `Left arm raises up (y=${lUpper1.y.toFixed(3)})`);
  assert(rUpper1.y < -0.6, `Right arm remains down (y=${rUpper1.y.toFixed(3)})`);

  console.log("--- 3. Raise Right Arm Only ---");
  retargeter.reset();
  const p2 = createSyntheticPose(4500);
  retargeter.update(p2.landmarks, p2.worldLandmarks, 1);
  const lUpper2 = getDir(rig.mapping.leftUpperArm, rig.mapping.leftForearm);
  const rUpper2 = getDir(rig.mapping.rightUpperArm, rig.mapping.rightForearm);
  assert(lUpper2.y < -0.6, `Left arm remains down (y=${lUpper2.y.toFixed(3)})`);
  assert(rUpper2.y > 0.2, `Right arm raises up (y=${rUpper2.y.toFixed(3)})`);

  console.log("--- 4. Raise Both Arms ---");
  retargeter.reset();
  const p3 = createSyntheticPose(6500);
  retargeter.update(p3.landmarks, p3.worldLandmarks, 1);
  const lUpper3 = getDir(rig.mapping.leftUpperArm, rig.mapping.leftForearm);
  const rUpper3 = getDir(rig.mapping.rightUpperArm, rig.mapping.rightForearm);
  assert(lUpper3.y > 0.2, `Left arm raises up (y=${lUpper3.y.toFixed(3)})`);
  assert(rUpper3.y > 0.2, `Right arm raises up (y=${rUpper3.y.toFixed(3)})`);

  console.log("--- 5. Bend Each Elbow ---");
  retargeter.reset();
  const p4 = createSyntheticPose(8500);
  retargeter.update(p4.landmarks, p4.worldLandmarks, 1);
  const lForearm4 = getDir(rig.mapping.leftForearm, rig.mapping.leftHand);
  assert(lForearm4.z > 0.3 || lForearm4.y > 0, `Left elbow bends forward/up (z=${lForearm4.z.toFixed(3)}, y=${lForearm4.y.toFixed(3)})`);

  retargeter.reset();
  const p5 = createSyntheticPose(10500);
  retargeter.update(p5.landmarks, p5.worldLandmarks, 1);
  const rForearm5 = getDir(rig.mapping.rightForearm, rig.mapping.rightHand);
  assert(rForearm5.z > 0.3 || rForearm5.y > 0, `Right elbow bends forward/up (z=${rForearm5.z.toFixed(3)}, y=${rForearm5.y.toFixed(3)})`);

  console.log("--- 6. Tilt Head Left and Right ---");
  retargeter.reset();
  const p6 = createSyntheticPose(12500);
  retargeter.update(p6.landmarks, p6.worldLandmarks, 1);
  assert(retargeter.diagnostics.headRoll < -0.15, `Head rolls left (roll=${retargeter.diagnostics.headRoll.toFixed(3)})`);

  retargeter.reset();
  const p7 = createSyntheticPose(14500);
  retargeter.update(p7.landmarks, p7.worldLandmarks, 1);
  assert(retargeter.diagnostics.headRoll > 0.15, `Head rolls right (roll=${retargeter.diagnostics.headRoll.toFixed(3)})`);

  console.log("--- 7. Turn Head Left and Right ---");
  retargeter.reset();
  const p8 = createSyntheticPose(16500);
  retargeter.update(p8.landmarks, p8.worldLandmarks, 1);
  assert(retargeter.diagnostics.headYaw > 0.2, `Head turns left (yaw=${retargeter.diagnostics.headYaw.toFixed(3)})`);

  retargeter.reset();
  const p9 = createSyntheticPose(18500);
  retargeter.update(p9.landmarks, p9.worldLandmarks, 1);
  assert(retargeter.diagnostics.headYaw < -0.2, `Head turns right (yaw=${retargeter.diagnostics.headYaw.toFixed(3)})`);

  console.log("--- 8. Turn Entire Torso Left and Right ---");
  retargeter.reset();
  const p10 = createSyntheticPose(20500);
  retargeter.update(p10.landmarks, p10.worldLandmarks, 1);
  assert(retargeter.diagnostics.torsoYaw > 0.2, `Torso turns left (yaw=${retargeter.diagnostics.torsoYaw.toFixed(3)})`);

  retargeter.reset();
  const p11 = createSyntheticPose(22500);
  retargeter.update(p11.landmarks, p11.worldLandmarks, 1);
  assert(retargeter.diagnostics.torsoYaw < -0.2, `Torso turns right (yaw=${retargeter.diagnostics.torsoYaw.toFixed(3)})`);

  console.log("--- 9. Lean Forward and Backward ---");
  retargeter.reset();
  const p12 = createSyntheticPose(24500);
  retargeter.update(p12.landmarks, p12.worldLandmarks, 1);
  assert(retargeter.diagnostics.torsoPitch > 0.15, `Torso leans forward (pitch=${retargeter.diagnostics.torsoPitch.toFixed(3)})`);

  retargeter.reset();
  const p13 = createSyntheticPose(26500);
  retargeter.update(p13.landmarks, p13.worldLandmarks, 1);
  assert(retargeter.diagnostics.torsoPitch < -0.15, `Torso leans backward (pitch=${retargeter.diagnostics.torsoPitch.toFixed(3)})`);

  console.log("--- 10. Return to Neutral ---");
  retargeter.update(neutral.landmarks, neutral.worldLandmarks, 1);
  assert(Math.abs(retargeter.diagnostics.torsoYaw) < 0.1, `Torso yaw returns to neutral (${retargeter.diagnostics.torsoYaw.toFixed(3)})`);
  assert(Math.abs(retargeter.diagnostics.headYaw) < 0.1, `Head yaw returns to neutral (${retargeter.diagnostics.headYaw.toFixed(3)})`);

  console.log("--- 11. Tracking Loss and Reacquisition ---");
  const lost = createSyntheticPose(32500);
  retargeter.update(lost.landmarks, lost.worldLandmarks, 0.1);
  assert(retargeter.tracked === false, "Tracking status is false when lost");
  retargeter.update(neutral.landmarks, neutral.worldLandmarks, 1);
  assert(retargeter.tracked === true, "Tracking status is true when reacquired");

  console.log("--- 12. Rear View Retargeting with BodyPix Orientation (FR-05 SCBA Cylinder Inspection) ---");
  retargeter.reset();
  const rearMetrics = {
    facingBias: -0.85,
    viewMode: "back",
    confidence: 1.0,
    isRearView: true,
    isFrontView: false,
  };
  retargeter.update(neutral.landmarks, neutral.worldLandmarks, 1, rearMetrics);
  assert(
    retargeter.diagnostics.torsoForward.z < 0,
    `Torso forward normal faces backwards in rear view (z=${retargeter.diagnostics.torsoForward.z.toFixed(3)} < 0)`,
  );
  assert(
    retargeter.diagnostics.viewMode === "back",
    "Torso diagnostics viewMode reflects rear view for SCBA cylinder inspection",
  );

  if (!allPassed) {
    process.exit(1);
  }
  console.log("ALL 12 RETARGETING REQUIREMENTS VERIFIED SUCCESSFULLY!");
});
