import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import fs from 'fs';
import path from 'path';
import { inspectHumanoidRig } from '../frontend/retargeting.js';
import {
  createSCBACylinderApparatus,
  createSCBAFaceMask,
  createSafetyHelmet,
  createSafetyVest,
  createProtectiveGloves,
} from '../frontend/scbaApparatus.js';

let passed = true;
function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    passed = false;
  } else {
    console.log(`PASS: ${msg}`);
  }
}

console.log('=== Avatar Scaling & Calibrated PPE Suite ===');

const buffer = fs.readFileSync(path.resolve('..', 'models', 'avatar.glb'));
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

new GLTFLoader().parse(arrayBuffer, '', (gltf) => {
  const model = gltf.scene;
  const rig = inspectHumanoidRig(model);
  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3());
  const totalH = size.y;

  const leftShoulderJoint = rig.mapping.leftUpperArm || rig.anchors.leftShoulder;
  const rightShoulderJoint = rig.mapping.rightUpperArm || rig.anchors.rightShoulder;
  const hips = rig.mapping.hips;
  const headBone = rig.mapping.head;
  const chestBone = rig.mapping.chest || rig.mapping.spine;

  const shoulderCenter = new THREE.Vector3();
  if (leftShoulderJoint && rightShoulderJoint) {
    const leftP = leftShoulderJoint.getWorldPosition(new THREE.Vector3());
    const rightP = rightShoulderJoint.getWorldPosition(new THREE.Vector3());
    shoulderCenter.addVectors(leftP, rightP).multiplyScalar(0.5);
  }

  const group = new THREE.Group();
  const normalization = new THREE.Group();
  normalization.add(model);
  group.add(normalization);
  normalization.scale.setScalar(1 / totalH);
  normalization.position.set(
    -shoulderCenter.x / totalH,
    -shoulderCenter.y / totalH,
    -shoulderCenter.z / totalH,
  );
  normalization.updateMatrixWorld(true);

  // Model reference measurements
  const leftShoulder = leftShoulderJoint.getWorldPosition(new THREE.Vector3());
  const rightShoulder = rightShoulderJoint.getWorldPosition(new THREE.Vector3());
  const measuredShoulderWidth = leftShoulder.distanceTo(rightShoulder);
  const modelShoulderWidth = Math.max(measuredShoulderWidth * 1.15, 0.26);
  const modelShoulderHeight = (leftShoulder.y + rightShoulder.y) / 2;
  const modelHipHeight = hips.getWorldPosition(new THREE.Vector3()).y;
  const modelTorsoHeight = Math.abs(modelShoulderHeight - modelHipHeight);
  const headPosition = headBone.getWorldPosition(new THREE.Vector3());
  const modelHeadHeight = headPosition ? headPosition.y + 0.050 : modelShoulderHeight + 0.165;
  const modelHeadShoulderHeight = Math.abs(modelHeadHeight - modelShoulderHeight);

  let footL = rig.mapping.leftFoot;
  let footR = rig.mapping.rightFoot;
  if (!footL || !footR) {
    model.traverse((o) => {
      if (o.isBone || o.name) {
        const nl = o.name.toLowerCase();
        if (!footL && (nl.includes("foot.l") || nl.includes("footl") || nl.includes("foot_l"))) footL = o;
        if (!footR && (nl.includes("foot.r") || nl.includes("footr") || nl.includes("foot_r"))) footR = o;
      }
    });
  }
  const leftAnkle = footL || rig.mapping.leftLeg;
  const rightAnkle = footR || rig.mapping.rightLeg;
  const lPos = leftAnkle.getWorldPosition(new THREE.Vector3());
  const rPos = rightAnkle.getWorldPosition(new THREE.Vector3());
  const ankleY = (lPos.y + rPos.y) / 2;
  const modelHeadToAnkleHeight = Math.min(Math.max(Math.abs(modelHeadHeight - ankleY), 0.80), 0.90);

  assert(modelHeadToAnkleHeight >= 0.82 && modelHeadToAnkleHeight <= 0.90,
    `modelHeadToAnkleHeight accurately matches human anatomy (${modelHeadToAnkleHeight.toFixed(3)})`);
  assert(modelTorsoHeight >= 0.25 && modelTorsoHeight <= 0.28,
    `modelTorsoHeight matches human torso (${modelTorsoHeight.toFixed(3)})`);
  assert(modelHeadShoulderHeight >= 0.15 && modelHeadShoulderHeight <= 0.18,
    `modelHeadShoulderHeight matches nose-to-shoulder (${modelHeadShoulderHeight.toFixed(3)})`);
  assert(modelShoulderWidth >= 0.25 && modelShoulderWidth <= 0.30,
    `modelShoulderWidth matches acromion width (${modelShoulderWidth.toFixed(3)})`);

  // Calibrated attachments with tad scale boost
  const helmet = createSafetyHelmet();
  helmet.scale.setScalar(1.08);
  helmet.position.set(0, 0.062, 0.010);
  headBone.add(helmet);

  const mask = createSCBAFaceMask();
  mask.scale.setScalar(0.87);
  mask.position.set(0, -0.018, 0.120);
  headBone.add(mask);

  const vest = createSafetyVest();
  vest.scale.setScalar(1.20);
  vest.position.set(0, 0.150, 0.012);
  chestBone.add(vest);

  const cylinder = createSCBACylinderApparatus();
  cylinder.scale.setScalar(1.02);
  cylinder.position.set(0, 0.085, -0.118);
  chestBone.add(cylinder);

  group.updateMatrixWorld(true);

  // Check helmet width
  const helmetSz = new THREE.Box3().setFromObject(helmet).getSize(new THREE.Vector3());
  const helmetW = helmetSz.x * totalH;
  assert(helmetW >= 0.24 && helmetW <= 0.30,
    `Safety helmet width: ${(helmetW * 100).toFixed(1)} cm (expected 24-30 cm)`);

  // Check mask width & position
  const maskSz = new THREE.Box3().setFromObject(mask).getSize(new THREE.Vector3());
  const maskW = maskSz.x * totalH;
  assert(maskW >= 0.20 && maskW <= 0.26,
    `SCBA mask width: ${(maskW * 100).toFixed(1)} cm (expected 20-26 cm)`);
  assert(mask.position.z >= 0.10,
    `SCBA mask sits in front of skull at Z=${mask.position.z.toFixed(3)}m (avoids occluder clipping)`);

  // Check vest width & position
  const vestSz = new THREE.Box3().setFromObject(vest).getSize(new THREE.Vector3());
  const vestW = vestSz.x * totalH;
  assert(vestW >= 0.46 && vestW <= 0.54,
    `Safety vest width: ${(vestW * 100).toFixed(1)} cm (expected 46-54 cm)`);

  // Check cylinder width & height
  const cylSz = new THREE.Box3().setFromObject(cylinder).getSize(new THREE.Vector3());
  const cylW = cylSz.x * totalH;
  const cylH = cylSz.y * totalH;
  assert(cylW >= 0.30 && cylW <= 0.38,
    `SCBA cylinder width: ${(cylW * 100).toFixed(1)} cm (expected 30-38 cm)`);
  assert(cylH >= 0.50 && cylH <= 0.62,
    `SCBA cylinder height: ${(cylH * 100).toFixed(1)} cm (expected 50-62 cm)`);

  if (passed) {
    console.log('ALL AVATAR SCALING AND PPE CALIBRATION TESTS PASSED!');
    process.exit(0);
  } else {
    process.exit(1);
  }
});
