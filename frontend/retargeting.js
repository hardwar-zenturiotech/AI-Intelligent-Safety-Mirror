import * as THREE from "three";
import { isLandmarkUsable } from "./poseTracker.js";

const BONE_ALIASES = {
  hips: ["hips", "pelvis", "root"],
  spine: ["spine", "spine05", "spine04"],
  chest: ["spine1", "chest", "spine03", "spine02", "spine01", "upperchest"],
  neck: ["neck", "neck01"],
  head: ["head"],
  leftUpperArm: ["leftarm", "leftupperarm", "arm_l", "upperarm_l", "upperarm01.l"],
  leftForearm: ["leftforearm", "leftlowerarm", "leftforearm_twist", "lowerarm_l", "lowerarm01.l"],
  leftHand: ["lefthand", "hand_l", "wrist.l"],
  rightUpperArm: ["rightarm", "rightupperarm", "arm_r", "upperarm_r", "upperarm01.r"],
  rightForearm: ["rightforearm", "rightlowerarm", "rightforearm_twist", "lowerarm_r", "lowerarm01.r"],
  rightHand: ["righthand", "hand_r", "wrist.r"],
  leftUpperLeg: ["leftupleg", "leftupperleg", "thigh_l", "upperleg_l", "upperleg01.l"],
  leftLeg: ["leftleg", "leftlowerleg", "calf_l", "lowerleg_l", "lowerleg01.l"],
  rightUpperLeg: ["rightupleg", "rightupperleg", "thigh_r", "upperleg_r", "upperleg01.r"],
  rightLeg: ["rightleg", "rightlowerleg", "calf_r", "lowerleg_r", "lowerleg01.r"],
};

const SEGMENTS = [
  ["leftUpperArm", 11, 13],
  ["leftForearm", 13, 15],
  ["rightUpperArm", 12, 14],
  ["rightForearm", 14, 16],
  ["leftUpperLeg", 23, 25],
  ["leftLeg", 25, 27],
  ["rightUpperLeg", 24, 26],
  ["rightLeg", 26, 28],
];

const CONFIDENCE_THRESHOLD = 0.45;

function normalizedName(name) {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function findBone(bones, aliases) {
  const indexed = new Map(bones.map((bone) => [normalizedName(bone.name), bone]));
  for (const alias of aliases) {
    const exact = indexed.get(normalizedName(alias));
    if (exact) {
      return exact;
    }
  }

  for (const bone of bones) {
    const candidate = normalizedName(bone.name);
    const alias = aliases.find((name) => candidate.endsWith(normalizedName(name)));
    if (alias) {
      return bone;
    }
  }
  return null;
}

export function inspectHumanoidRig(root) {
  const skinnedMeshes = [];
  const bones = [];
  const seen = new Set();
  root.traverse((object) => {
    if (object.isSkinnedMesh && object.skeleton) {
      skinnedMeshes.push(object);
      for (const bone of object.skeleton.bones) {
        if (!seen.has(bone)) {
          bones.push(bone);
          seen.add(bone);
        }
      }
    }
    if (object.isBone && !seen.has(object)) {
      bones.push(object);
      seen.add(object);
    }
  });

  const mapping = Object.fromEntries(
    Object.entries(BONE_ALIASES).map(([role, aliases]) => [role, findBone(bones, aliases)]),
  );
  const anchors = {
    leftShoulder: findBone(bones, ["shoulder01l", "leftshoulder", "claviclel"]),
    rightShoulder: findBone(bones, ["shoulder01r", "rightshoulder", "clavicler"]),
  };
  return {
    skinnedMeshes,
    bones,
    mapping,
    anchors,
    missing: Object.entries(mapping)
      .filter(([, bone]) => !bone)
      .map(([role]) => role),
  };
}

function usablePair(landmarks, from, to) {
  return isLandmarkUsable(landmarks[from], CONFIDENCE_THRESHOLD) &&
    isLandmarkUsable(landmarks[to], CONFIDENCE_THRESHOLD);
}

// Use one camera-to-avatar convention: +X image-right, +Y up, +Z away to Three.js +Y up and +Z toward camera.
export function convertMediaPipeWorldVector(point, target) {
  return target.set(point.x, point.y, -point.z);
}

const TORSO_ALIGNMENT_WEIGHTS = [
  ["hips", 0.25],
  ["spine", 0.6],
  ["chest", 1],
];

export class MotionRetargeter {
  constructor(mapping, root, anchors = {}) {
    this.mapping = mapping;
    this.anchors = anchors;
    this.root = root;
    this.restLocal = new Map();
    this.restWorld = new Map();
    this.restRootWorld = new Map();
    this.directionOffsets = new Map();
    this.desiredLocal = new THREE.Quaternion();
    this.desiredWorld = new THREE.Quaternion();
    this.parentWorld = new THREE.Quaternion();
    this.rootWorld = new THREE.Quaternion();
    this.rootWorldInverse = new THREE.Quaternion();
    this.rootInverseMatrix = new THREE.Matrix4();
    this.parentChain = [];
    this.restTorsoFrame = new THREE.Quaternion();
    this.restTorsoFrameInverse = new THREE.Quaternion();
    this.measuredTorsoFrame = new THREE.Quaternion();
    this.filteredTorsoFrame = new THREE.Quaternion();
    this.torsoDelta = new THREE.Quaternion();
    this.weightedTorsoDelta = new THREE.Quaternion();
    this.measuredHeadFrame = new THREE.Quaternion();
    this.headRelativeTarget = new THREE.Quaternion();
    this.headRelativeFiltered = new THREE.Quaternion();
    this.identityQuaternion = new THREE.Quaternion();
    this.torsoFrameMatrix = new THREE.Matrix4();
    this.headFrameMatrix = new THREE.Matrix4();
    this.direction = new THREE.Vector3();
    this.restDirection = new THREE.Vector3();
    this.hipCenter = new THREE.Vector3();
    this.shoulderCenter = new THREE.Vector3();
    this.leftShoulder = new THREE.Vector3();
    this.rightShoulder = new THREE.Vector3();
    this.leftHip = new THREE.Vector3();
    this.rightHip = new THREE.Vector3();
    this.torsoUp = new THREE.Vector3();
    this.torsoLateral = new THREE.Vector3();
    this.shoulderLateral = new THREE.Vector3();
    this.hipLateral = new THREE.Vector3();
    this.torsoForward = new THREE.Vector3();
    this.frameUp = new THREE.Vector3();
    this.leftEar = new THREE.Vector3();
    this.rightEar = new THREE.Vector3();
    this.nose = new THREE.Vector3();
    this.faceLateral = new THREE.Vector3();
    this.faceForward = new THREE.Vector3();
    this.faceUp = new THREE.Vector3();
    this.headEuler = new THREE.Euler(0, 0, 0, "YXZ");
    this.torsoEuler = new THREE.Euler(0, 0, 0, "YXZ");
    this.delta = new THREE.Quaternion();
    this.tracked = false;
    this.diagnostics = {
      modelForward: new THREE.Vector3(0, 0, 1),
      trackedLeftShoulder: new THREE.Vector3(),
      trackedRightShoulder: new THREE.Vector3(),
      torsoForward: new THREE.Vector3(0, 0, 1),
      torsoUp: new THREE.Vector3(0, 1, 0),
      headForward: new THREE.Vector3(0, 0, 1),
      torsoYaw: 0,
      torsoPitch: 0,
      torsoRoll: 0,
      headYaw: 0,
      headPitch: 0,
      headRoll: 0,
    };

    root.updateMatrixWorld(true);
    root.getWorldQuaternion(this.rootWorld);
    this.rootWorldInverse.copy(this.rootWorld).invert();
    for (const bone of Object.values(mapping)) {
      if (!bone || this.restLocal.has(bone)) {
        continue;
      }
      this.restLocal.set(bone, bone.quaternion.clone());
      bone.getWorldQuaternion(this.desiredLocal);
      this.restWorld.set(bone, this.desiredLocal.clone());
      this.restRootWorld.set(
        bone,
        this.rootWorldInverse.clone().multiply(this.desiredLocal),
      );
    }
    this.captureRestTorsoFrame();
    this.filteredTorsoFrame.copy(this.restTorsoFrame);
    this.restTorsoFrameInverse.copy(this.restTorsoFrame).invert();
    this.captureRestDirections();
  }

  captureRestTorsoFrame() {
    const hips = this.mapping.hips;
    const leftShoulder = this.anchors.leftShoulder || this.mapping.leftUpperArm;
    const rightShoulder = this.anchors.rightShoulder || this.mapping.rightUpperArm;
    if (!hips || !leftShoulder || !rightShoulder) {
      this.restTorsoFrame.identity();
      return;
    }

    leftShoulder.getWorldPosition(this.leftShoulder);
    rightShoulder.getWorldPosition(this.rightShoulder);
    this.torsoUp.set(0, 1, 0);
    this.torsoLateral.subVectors(this.leftShoulder, this.rightShoulder);
    if (this.torsoUp.lengthSq() < 1e-8 || this.torsoLateral.lengthSq() < 1e-8) {
      this.restTorsoFrame.identity();
      return;
    }

    this.torsoUp.normalize();
    this.torsoLateral.projectOnPlane(this.torsoUp);
    if (this.torsoLateral.lengthSq() < 1e-8) {
      this.restTorsoFrame.identity();
      return;
    }
    this.torsoLateral.normalize();
    this.torsoForward.crossVectors(this.torsoLateral, this.torsoUp).normalize();
    this.frameUp.crossVectors(this.torsoForward, this.torsoLateral).normalize();
    this.torsoFrameMatrix.makeBasis(this.torsoLateral, this.frameUp, this.torsoForward);
    this.restTorsoFrame.setFromRotationMatrix(this.torsoFrameMatrix);
    this.diagnostics.modelForward.copy(this.torsoForward);
  }

  captureRestDirections() {
    const byRole = this.mapping;
    const pairs = [
      ["leftUpperArm", byRole.leftUpperArm, byRole.leftForearm],
      ["leftForearm", byRole.leftForearm, byRole.leftHand],
      ["rightUpperArm", byRole.rightUpperArm, byRole.rightForearm],
      ["rightForearm", byRole.rightForearm, byRole.rightHand],
      ["leftUpperLeg", byRole.leftUpperLeg, byRole.leftLeg],
      ["leftLeg", byRole.leftLeg, null],
      ["rightUpperLeg", byRole.rightUpperLeg, byRole.rightLeg],
      ["rightLeg", byRole.rightLeg, null],
    ];
    const from = new THREE.Vector3();
    const to = new THREE.Vector3();
    for (const [role, bone, child] of pairs) {
      if (!bone) {
        continue;
      }
      bone.getWorldPosition(from);
      const joint = child || bone.children.find((object) => object.isBone);
      if (!joint) {
        continue;
      }
      joint.getWorldPosition(to);
      this.root.worldToLocal(from);
      this.root.worldToLocal(to);
      this.restDirection.subVectors(to, from);
      if (this.restDirection.lengthSq() > 1e-8) {
        this.directionOffsets.set(role, { local: this.restDirection.clone().normalize() });
      }
    }
  }

  reset() {
    for (const [bone, quaternion] of this.restLocal) {
      bone.quaternion.copy(quaternion);
    }
    this.root.updateMatrixWorld(true);
    this.filteredTorsoFrame.copy(this.restTorsoFrame);
    this.headRelativeFiltered.identity();
    this.updateTorsoDiagnostics();
    this.diagnostics.headYaw = 0;
    this.diagnostics.headPitch = 0;
    this.diagnostics.headRoll = 0;
    this.tracked = false;
  }

  applyWorldRotation(bone, worldRotation, smoothing) {
    this.desiredLocal.copy(worldRotation);
    if (bone.parent) {
      this.parentWorld.identity();
      this.parentChain.length = 0;
      let ancestor = bone.parent;
      while (ancestor && ancestor !== this.root) {
        this.parentChain.push(ancestor);
        ancestor = ancestor.parent;
      }
      for (let index = this.parentChain.length - 1; index >= 0; index -= 1) {
        this.parentWorld.multiply(this.parentChain[index].quaternion);
      }
      this.desiredLocal.premultiply(this.parentWorld.invert());
    }
    bone.quaternion.slerp(this.desiredLocal, smoothing);
    this.root.updateMatrixWorld(true);
  }

  restoreBone(bone, smoothing) {
    const rest = this.restLocal.get(bone);
    if (rest) {
      bone.quaternion.slerp(rest, smoothing * 0.55);
      this.root.updateMatrixWorld(true);
    }
  }

  update(landmarks, worldLandmarks, deltaSeconds) {
    const delta = Math.min(Math.max(deltaSeconds ?? 0, 0), 0.1);
    const smoothing = 1 - Math.exp(-12 * delta);
    let updated = false;

    if (landmarks && worldLandmarks) {
      const torsoUpdated = this.updateTorso(worldLandmarks, smoothing);
      if (!torsoUpdated) this.restoreTorso(smoothing);
      updated = torsoUpdated || updated;

      this.root.updateMatrixWorld(true);
      this.rootInverseMatrix.copy(this.root.matrixWorld).invert();
      this.root.getWorldQuaternion(this.rootWorld);
      for (const [role, start, end] of SEGMENTS) {
        const bone = this.mapping[role];
        if (!bone) {
          continue;
        }
        if (
          !usablePair(landmarks, start, end) ||
          !isLandmarkUsable(worldLandmarks[start], CONFIDENCE_THRESHOLD) ||
          !isLandmarkUsable(worldLandmarks[end], CONFIDENCE_THRESHOLD)
        ) {
          this.restoreBone(bone, smoothing);
          continue;
        }
        this.direction.set(
          worldLandmarks[end].x - worldLandmarks[start].x,
          worldLandmarks[end].y - worldLandmarks[start].y,
          worldLandmarks[end].z - worldLandmarks[start].z,
        );
        if (this.direction.lengthSq() < 1e-8) {
          continue;
        }
        convertMediaPipeWorldVector(this.direction, this.direction);
        // Invert the fitted root transform so camera-mode anisotropic scaling does not skew the target direction.
        this.direction.transformDirection(this.rootInverseMatrix);
        const restDirections = this.directionOffsets.get(role);
        if (!restDirections) {
          continue;
        }

        this.delta.setFromUnitVectors(restDirections.local, this.direction);
        this.desiredWorld
          .copy(this.rootWorld)
          .multiply(this.delta)
          .multiply(this.restRootWorld.get(bone));
        this.applyWorldRotation(bone, this.desiredWorld, smoothing);
        updated = true;
      }

      const headUpdated = this.updateHead(worldLandmarks, smoothing);
      if (!headUpdated) this.restoreHead(smoothing);
      updated = headUpdated || updated;
    }

    if (!updated) {
      for (const [bone, quaternion] of this.restLocal) {
        bone.quaternion.slerp(quaternion, smoothing * 0.55);
      }
      this.filteredTorsoFrame.slerp(this.restTorsoFrame, smoothing * 0.55);
      this.headRelativeFiltered.slerp(this.identityQuaternion, smoothing * 0.55);
      this.updateTorsoDiagnostics();
    }
    this.tracked = updated;
    this.root.updateMatrixWorld(true);
    return updated;
  }

  updateTorso(worldLandmarks, smoothing) {
    if (
      !isLandmarkUsable(worldLandmarks[11], CONFIDENCE_THRESHOLD) ||
      !isLandmarkUsable(worldLandmarks[12], CONFIDENCE_THRESHOLD) ||
      !isLandmarkUsable(worldLandmarks[23], CONFIDENCE_THRESHOLD) ||
      !isLandmarkUsable(worldLandmarks[24], CONFIDENCE_THRESHOLD)
    ) {
      return false;
    }

    convertMediaPipeWorldVector(worldLandmarks[11], this.leftShoulder);
    convertMediaPipeWorldVector(worldLandmarks[12], this.rightShoulder);
    this.diagnostics.trackedLeftShoulder.copy(this.leftShoulder);
    this.diagnostics.trackedRightShoulder.copy(this.rightShoulder);
    convertMediaPipeWorldVector(worldLandmarks[23], this.leftHip);
    convertMediaPipeWorldVector(worldLandmarks[24], this.rightHip);
    this.hipCenter.addVectors(this.leftHip, this.rightHip).multiplyScalar(0.5);
    this.shoulderCenter
      .addVectors(this.leftShoulder, this.rightShoulder)
      .multiplyScalar(0.5);
    this.torsoUp.subVectors(this.shoulderCenter, this.hipCenter);
    if (this.torsoUp.lengthSq() < 1e-8) {
      return false;
    }
    this.torsoUp.normalize();

    this.shoulderLateral.subVectors(this.leftShoulder, this.rightShoulder);
    this.hipLateral.subVectors(this.leftHip, this.rightHip);
    if (this.shoulderLateral.lengthSq() < 1e-8 || this.hipLateral.lengthSq() < 1e-8) {
      return false;
    }
    this.shoulderLateral.normalize();
    this.hipLateral.normalize();
    if (this.shoulderLateral.dot(this.hipLateral) < -0.2) {
      return false;
    }
    this.torsoLateral
      .addVectors(this.shoulderLateral, this.hipLateral)
      .projectOnPlane(this.torsoUp);
    if (this.torsoLateral.lengthSq() < 0.05) {
      this.torsoLateral.copy(this.shoulderLateral).projectOnPlane(this.torsoUp);
    }
    if (this.torsoLateral.lengthSq() < 1e-8) {
      return false;
    }
    this.torsoLateral.normalize();
    this.torsoForward.crossVectors(this.torsoLateral, this.torsoUp);
    if (this.torsoForward.lengthSq() < 1e-8) {
      return false;
    }
    this.torsoForward.normalize();
    this.frameUp.crossVectors(this.torsoForward, this.torsoLateral).normalize();
    this.torsoFrameMatrix.makeBasis(this.torsoLateral, this.frameUp, this.torsoForward);
    this.measuredTorsoFrame.setFromRotationMatrix(this.torsoFrameMatrix);
    if (this.filteredTorsoFrame.dot(this.measuredTorsoFrame) < 0) {
      this.measuredTorsoFrame.set(
        -this.measuredTorsoFrame.x,
        -this.measuredTorsoFrame.y,
        -this.measuredTorsoFrame.z,
        -this.measuredTorsoFrame.w,
      );
    }
    this.filteredTorsoFrame.slerp(this.measuredTorsoFrame, smoothing);
    this.torsoDelta.copy(this.filteredTorsoFrame).multiply(this.restTorsoFrameInverse);
    this.updateTorsoDiagnostics();

    for (const [role, weight] of TORSO_ALIGNMENT_WEIGHTS) {
      const bone = this.mapping[role];
      const restWorld = this.restWorld.get(bone);
      if (!bone || !restWorld) continue;
      this.weightedTorsoDelta.identity().slerp(this.torsoDelta, weight);
      this.desiredWorld.copy(this.weightedTorsoDelta).multiply(restWorld);
      this.applyWorldRotation(bone, this.desiredWorld, smoothing);
    }
    return true;
  }

  updateTorsoDiagnostics() {
    this.torsoDelta.copy(this.filteredTorsoFrame).multiply(this.restTorsoFrameInverse);
    this.diagnostics.torsoForward
      .set(0, 0, 1)
      .applyQuaternion(this.filteredTorsoFrame)
      .normalize();
    this.diagnostics.torsoUp
      .set(0, 1, 0)
      .applyQuaternion(this.filteredTorsoFrame)
      .normalize();
    this.torsoEuler.setFromQuaternion(this.torsoDelta, "YXZ");
    this.diagnostics.torsoYaw = this.torsoEuler.y;
    this.diagnostics.torsoPitch = this.torsoEuler.x;
    this.diagnostics.torsoRoll = this.torsoEuler.z;
  }

  restoreTorso(smoothing) {
    this.filteredTorsoFrame.slerp(this.restTorsoFrame, smoothing * 0.55);
    this.updateTorsoDiagnostics();
    for (const [role] of TORSO_ALIGNMENT_WEIGHTS) {
      this.restoreBone(this.mapping[role], smoothing);
    }
  }

  updateHead(worldLandmarks, smoothing) {
    const head = this.mapping.head;
    if (
      !head ||
      !this.restWorld.has(head) ||
      !isLandmarkUsable(worldLandmarks[0], CONFIDENCE_THRESHOLD) ||
      !isLandmarkUsable(worldLandmarks[7], CONFIDENCE_THRESHOLD) ||
      !isLandmarkUsable(worldLandmarks[8], CONFIDENCE_THRESHOLD)
    ) {
      return false;
    }
    convertMediaPipeWorldVector(worldLandmarks[7], this.leftEar);
    convertMediaPipeWorldVector(worldLandmarks[8], this.rightEar);
    convertMediaPipeWorldVector(worldLandmarks[0], this.nose);
    this.faceLateral.subVectors(this.leftEar, this.rightEar);
    this.faceForward
      .subVectors(
        this.nose,
        this.leftEar.add(this.rightEar).multiplyScalar(0.5),
      )
      .projectOnPlane(this.faceLateral);
    if (
      this.faceLateral.lengthSq() < 1e-8 ||
      this.faceForward.lengthSq() < 1e-8
    ) {
      return false;
    }
    this.faceLateral.normalize();
    this.faceForward.normalize();
    this.faceUp.crossVectors(this.faceForward, this.faceLateral).normalize();
    this.faceLateral.crossVectors(this.faceUp, this.faceForward).normalize();
    this.headFrameMatrix.makeBasis(this.faceLateral, this.faceUp, this.faceForward);
    this.measuredHeadFrame.setFromRotationMatrix(this.headFrameMatrix);
    this.headRelativeTarget
      .copy(this.filteredTorsoFrame)
      .invert()
      .multiply(this.measuredHeadFrame);
    if (this.headRelativeFiltered.dot(this.headRelativeTarget) < 0) {
      this.headRelativeTarget.set(
        -this.headRelativeTarget.x,
        -this.headRelativeTarget.y,
        -this.headRelativeTarget.z,
        -this.headRelativeTarget.w,
      );
    }
    this.headRelativeFiltered.slerp(this.headRelativeTarget, smoothing);
    this.headEuler.setFromQuaternion(this.headRelativeFiltered, "YXZ");
    this.diagnostics.headYaw = this.headEuler.y;
    this.diagnostics.headPitch = this.headEuler.x;
    this.diagnostics.headRoll = this.headEuler.z;
    this.diagnostics.headForward
      .set(0, 0, 1)
      .applyQuaternion(this.measuredHeadFrame)
      .normalize();

    this.desiredWorld
      .copy(this.torsoDelta)
      .multiply(this.restWorld.get(head))
      .multiply(this.headRelativeFiltered);
    this.applyWorldRotation(head, this.desiredWorld, smoothing);
    return true;
  }

  restoreHead(smoothing) {
    this.headRelativeFiltered.slerp(this.identityQuaternion, smoothing * 0.55);
    this.restoreBone(this.mapping.head, smoothing);
    this.headEuler.setFromQuaternion(this.headRelativeFiltered, "YXZ");
    this.diagnostics.headYaw = this.headEuler.y;
    this.diagnostics.headPitch = this.headEuler.x;
    this.diagnostics.headRoll = this.headEuler.z;
  }
}

export function getRigReport(rig) {
  return {
    detected: rig.bones.map((bone) => bone.name),
    mapped: Object.entries(rig.mapping)
      .filter(([, bone]) => bone)
      .map(([role, bone]) => `${role}: ${bone.name}`),
    missing: rig.missing,
    skinnedMeshCount: rig.skinnedMeshes.length,
  };
}
