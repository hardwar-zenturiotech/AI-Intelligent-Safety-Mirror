import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mapLandmarkToStage } from "./camera.js";
import { isLandmarkUsable } from "./poseTracker.js";
import { MotionRetargeter, getRigReport, inspectHumanoidRig } from "./retargeting.js";
import {
  createSCBACylinderApparatus,
  createSCBAFaceMask,
  createSafetyHelmet,
  createSafetyVest,
  createProtectiveGloves,
} from "./scbaApparatus.js";

const LANDMARKS = { leftHip: 23, rightHip: 24, leftShoulder: 11, rightShoulder: 12 };

function mapVisibleLandmarkToStage(landmark, video, stage) {
  if (
    !isLandmarkUsable(landmark) ||
    landmark.x < 0 ||
    landmark.x > 1 ||
    landmark.y < 0 ||
    landmark.y > 1
  ) {
    return null;
  }
  const point = mapLandmarkToStage(landmark, video, stage, false);
  return point &&
    point.x >= 0 &&
    point.x <= stage.width &&
    point.y >= 0 &&
    point.y <= stage.height
    ? point
    : null;
}

class AdaptiveScalarFilter {
  constructor(minCutoff = 0.6, beta = 4.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.current = 0;
    this.hasValue = false;
  }

  reset() {
    this.hasValue = false;
  }

  filter(target, deltaSeconds) {
    if (!this.hasValue || deltaSeconds <= 0) {
      this.current = target;
      this.hasValue = true;
      return this.current;
    }

    const diff = Math.abs(this.current - target);
    // Subtle deadband on micro-scale fluctuations to stop size breathing
    const effectiveDiff = diff > 0.015 * Math.abs(this.current) ? diff : diff * 0.1;
    const speed = deltaSeconds > 0 ? effectiveDiff / deltaSeconds : 0;
    const cutoff = this.minCutoff + this.beta * speed;
    const alpha = THREE.MathUtils.clamp(
      1 - Math.exp(-2 * Math.PI * cutoff * deltaSeconds),
      0.04,
      1.0,
    );

    this.current += (target - this.current) * alpha;
    return this.current;
  }
}

class AdaptivePointFilter {
  constructor(minCutoff = 0.8, beta = 5.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.currentX = 0;
    this.currentY = 0;
    this.hasValue = false;
  }

  reset() {
    this.hasValue = false;
  }

  filter(targetX, targetY, deltaSeconds) {
    if (!this.hasValue || deltaSeconds <= 0) {
      this.currentX = targetX;
      this.currentY = targetY;
      this.hasValue = true;
      return { x: this.currentX, y: this.currentY };
    }

    const dist = Math.hypot(targetX - this.currentX, targetY - this.currentY);
    // Deadband on sub-pixel micro-jitter to prevent shivering while standing still
    const effectiveDist = dist > 2.2 ? dist - 2.2 : dist * 0.1;
    const speed = deltaSeconds > 0 ? effectiveDist / deltaSeconds : 0;
    const cutoff = this.minCutoff + this.beta * speed;
    const alpha = THREE.MathUtils.clamp(
      1 - Math.exp(-2 * Math.PI * cutoff * deltaSeconds),
      0.04,
      1.0,
    );

    this.currentX += (targetX - this.currentX) * alpha;
    this.currentY += (targetY - this.currentY) * alpha;
    return { x: this.currentX, y: this.currentY };
  }
}

export class Avatar {
  constructor(container, onStatus) {
    this.container = container;
    this.onStatus = onStatus;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
    this.camera.position.z = 1000;
    try {
      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
      });
    } catch (error) {
      this.unavailable = true;
      this.onStatus({
        state: "error",
        message: `WebGL avatar rendering is unavailable: ${error.message}`,
      });
      return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.className = "avatar-canvas";
    container.append(this.renderer.domElement);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x303638, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 2.5);
    light.position.set(1, 2, 4);
    this.scene.add(light);
    this.environmentGrid = new THREE.GridHelper(8, 24, 0x60706e, 0x35403f);
    this.environmentGrid.position.y = -0.75;
    this.environmentGrid.visible = false;
    this.scene.add(this.environmentGrid);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.scaleTarget = new THREE.Vector3(1, 1, 1);
    this.positionTarget = new THREE.Vector3();
    this.debugPosition = new THREE.Vector3();
    this.neutralPosition = new THREE.Vector3();
    this.tracked = null;
    this.renderMode = "camera";
    this.modelShoulderWidth = 0.28;
    this.modelHipWidth = 0.22;
    this.modelShoulderHeight = 0;
    this.modelShoulderCenterX = 0;
    this.modelHeight = 1;
    this.modelHipHeight = -0.264;
    this.modelTorsoHeight = 0.264;
    this.modelHeadHeight = 0.165;
    this.modelHeadShoulderHeight = 0.165;
    this.modelHeadToAnkleHeight = 0.86;
    this.modelShoulderAnchors = "upper-arm bones";
    this.lastScaleReference = "waiting for pose";
    this.lastHorizontalScaleReference = "waiting for pose";
    this.lastDetectedShoulderWidth = 0;
    this.lastDetectedBodyHeight = 0;
    this.lastDetectedTorsoHeight = 0;
    this.lastDetectedHeadShoulderHeight = 0;
    this.lastDetectedDepth = 0;
    this.calibratedPixelsPerMeter = null;
    this.calibratedTorsoMeters = 0.50;
    this.calibratedShoulderMeters = 0.40;
    this.userScaleMultiplier = 1.50;
    this.userDepthOffset = 0;
    this.lastJointOffsets = {};
    this.calibrationHelpers = [];
    this.calibrationDebugVisible = false;
    this.originFilter = new AdaptivePointFilter(1.2, 8.0);
    this.scaleFilter = new AdaptiveScalarFilter(0.6, 4.0);
    this.widthFactorFilter = new AdaptiveScalarFilter(0.6, 4.0);
    this.silhouetteBounds = null;
    this.scbaVisible = false;
    this.scbaCylinder = null;
    this.scbaMask = null;
    this.ppeHelmet = null;
    this.ppeVest = null;
    this.ppeGloveL = null;
    this.ppeGloveR = null;
    this.ppeVisibility = {
      helmet: false,
      vest: false,
      gloves: false,
      respirator: false,
    };
    this.avatarBodyMode = "occluder";
    this.lastTime = performance.now();
    this.frameCount = 0;
    this.fpsTime = this.lastTime;
    this.fps = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.animate = this.animate.bind(this);
    this.animationId = requestAnimationFrame(this.animate);
    this.load();
  }

  async load() {
    if (this.unavailable) {
      return;
    }
    try {
      const gltf = await new GLTFLoader().loadAsync("/models/avatar.glb");
      this.model = gltf.scene;
      this.model.updateMatrixWorld(true);
      const rig = inspectHumanoidRig(this.model);
      const report = getRigReport(rig);
      this.report = report;
      this.onStatus({ state: "inspecting", ...report });
      if (!rig.skinnedMeshes.length) {
        throw new Error(
          `avatar.glb contains ${report.detected.length} skeleton bones and ${rig.skinnedMeshes.length} SkinnedMesh objects. A rigged humanoid GLB is required.`,
        );
      }

      const bounds = new THREE.Box3().setFromObject(this.model);
      const size = bounds.getSize(new THREE.Vector3());
      if (!Number.isFinite(size.y) || size.y <= 0) {
        throw new Error("avatar.glb has no measurable humanoid height.");
      }
      this.modelHeightMeters = size.y;
      const hips = rig.mapping.hips;
      if (!hips) {
        throw new Error("No Hips/Pelvis bone was detected in the humanoid skeleton.");
      }

      // Glenohumeral shoulder joints (upper arm pivots) correspond directly to MediaPipe landmarks 11 & 12
      const leftUpperArm = rig.mapping.leftUpperArm;
      const rightUpperArm = rig.mapping.rightUpperArm;
      const leftShoulderJoint = leftUpperArm || rig.anchors.leftShoulder;
      const rightShoulderJoint = rightUpperArm || rig.anchors.rightShoulder;

      // Find shoulder center in unscaled model coordinates
      this.model.updateMatrixWorld(true);
      const shoulderCenter = new THREE.Vector3();
      if (leftShoulderJoint && rightShoulderJoint) {
        const leftP = leftShoulderJoint.getWorldPosition(new THREE.Vector3());
        const rightP = rightShoulderJoint.getWorldPosition(new THREE.Vector3());
        shoulderCenter.addVectors(leftP, rightP).multiplyScalar(0.5);
      } else {
        shoulderCenter.copy(hips.getWorldPosition(new THREE.Vector3()));
      }

      const normalization = new THREE.Group();
      normalization.add(this.model);
      this.group.add(normalization);
      normalization.scale.setScalar(1 / size.y);
      // Pivot normalization at the shoulder center: rotating or turning profile pivots around the neck/shoulders,
      // completely eliminating the 50cm arc swing that previously threw the head/helmet off to the side!
      normalization.position.set(
        -shoulderCenter.x / size.y,
        -shoulderCenter.y / size.y,
        -shoulderCenter.z / size.y,
      );
      normalization.updateMatrixWorld(true);
      this.modelHipHeight = hips.getWorldPosition(new THREE.Vector3()).y;
      if (leftShoulderJoint && rightShoulderJoint) {
        const leftShoulder = leftShoulderJoint.getWorldPosition(new THREE.Vector3());
        const rightShoulder = rightShoulderJoint.getWorldPosition(new THREE.Vector3());
        const measuredShoulderWidth = leftShoulder.distanceTo(rightShoulder);
        // Calibrate model shoulder width: MediaPipe measures outer acromion width (~0.28 normalized),
        // whereas glenohumeral joint centers are ~0.228. Scale appropriately to prevent oversized avatar.
        if (Number.isFinite(measuredShoulderWidth) && measuredShoulderWidth > 0.01) {
          this.modelShoulderWidth = Math.max(measuredShoulderWidth * 1.15, 0.26);
        }
        this.modelShoulderHeight = (leftShoulder.y + rightShoulder.y) / 2;
        this.modelShoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2;
        this.modelTorsoHeight = Math.abs(this.modelShoulderHeight - this.modelHipHeight);
        // True facial nose tip is approximately 0.165 above shoulder center in normalized humanoid coords
        const headPosition = rig.mapping.head?.getWorldPosition(new THREE.Vector3());
        this.modelHeadHeight = headPosition ? headPosition.y + 0.050 : this.modelShoulderHeight + 0.165;
        this.modelHeadShoulderHeight = Math.abs(this.modelHeadHeight - this.modelShoulderHeight);
        this.modelShoulderAnchors = `${leftShoulderJoint.name} / ${rightShoulderJoint.name}`;
      }

      // Find foot / ankle bones for accurate ground and head-to-ankle calibration
      let footL = rig.mapping.leftFoot;
      let footR = rig.mapping.rightFoot;
      if (!footL || !footR) {
        this.model.traverse((o) => {
          if (o.isBone || o.name) {
            const nl = o.name.toLowerCase();
            if (!footL && (nl.includes("foot.l") || nl.includes("footl") || nl.includes("foot_l"))) footL = o;
            if (!footR && (nl.includes("foot.r") || nl.includes("footr") || nl.includes("foot_r"))) footR = o;
          }
        });
      }
      const leftAnkle = footL || rig.mapping.leftLeg;
      const rightAnkle = footR || rig.mapping.rightLeg;
      if (leftAnkle && rightAnkle) {
        const lPos = leftAnkle.getWorldPosition(new THREE.Vector3());
        const rPos = rightAnkle.getWorldPosition(new THREE.Vector3());
        const ankleY = (lPos.y + rPos.y) / 2;
        this.modelAnkleHeight = ankleY;
        this.modelHeadToAnkleHeight = Math.min(Math.max(Math.abs((this.modelHeadHeight || 0.165) - ankleY), 0.80), 0.90);
      } else {
        this.modelAnkleHeight = -0.75;
        this.modelHeadToAnkleHeight = 0.86;
      }
      if (rig.mapping.leftUpperLeg && rig.mapping.rightUpperLeg) {
        const leftHipPosition = rig.mapping.leftUpperLeg.getWorldPosition(new THREE.Vector3());
        const rightHipPosition = rig.mapping.rightUpperLeg.getWorldPosition(new THREE.Vector3());
        const measuredHipWidth = leftHipPosition.distanceTo(rightHipPosition);
        if (Number.isFinite(measuredHipWidth) && measuredHipWidth > 0.01) {
          this.modelHipWidth = measuredHipWidth;
        }
      }
      this.retargeter = new MotionRetargeter(rig.mapping, this.group, rig.anchors);
      this.skeletonHelper = new THREE.SkeletonHelper(this.model);
      this.skeletonHelper.visible = false;
      this.scene.add(this.skeletonHelper);
      this.createCalibrationHelpers(rig.mapping);

      // Attach 3D PPE items to the humanoid skeleton
      // The avatar model is normalized by (1 / size.y). Geometries in scbaApparatus.js are
      // modeled in real-world metric dimensions. We scale each attachment to match standard PPE dimensions:
      // - Helmet: ~23.5 cm width
      // - Mask: ~19.0 cm width
      // - Vest: ~43.0 cm width
      // - Cylinder: ~30.0 cm width, ~50 cm height
      const chestBone = rig.mapping.chest || rig.mapping.spine;
      const headBone = rig.mapping.head;
      const leftHandBone = rig.mapping.leftHand;
      const rightHandBone = rig.mapping.rightHand;

      if (chestBone) {
        // SCBA back air cylinder apparatus
        this.scbaCylinder = createSCBACylinderApparatus();
        this.scbaCylinder.scale.setScalar(1.02);
        this.scbaCylinder.position.set(0, 0.085, -0.118);
        this.scbaCylinder.renderOrder = 2;
        this.scbaCylinder.visible = Boolean(this.ppeVisibility.respirator);
        chestBone.add(this.scbaCylinder);

        // High-visibility safety vest
        this.ppeVest = createSafetyVest();
        this.ppeVest.scale.setScalar(1.20);
        this.ppeVest.position.set(0, 0.150, 0.012);
        this.ppeVest.renderOrder = 2;
        this.ppeVest.visible = Boolean(this.ppeVisibility.vest);
        chestBone.add(this.ppeVest);
      }

      if (headBone) {
        // SCBA full-face gas mask: calibrated to human facial proportions
        this.scbaMask = createSCBAFaceMask();
        this.scbaMask.scale.setScalar(0.87);
        this.scbaMask.position.set(0, -0.018, 0.120);
        this.scbaMask.renderOrder = 2;
        this.scbaMask.visible = Boolean(this.ppeVisibility.respirator);
        headBone.add(this.scbaMask);

        // Apply any user calibration already queued
        if (this.maskCalibration) {
          this.setMaskCalibration(this.maskCalibration);
        }

        // Safety hard hat helmet
        this.ppeHelmet = createSafetyHelmet();
        this.ppeHelmet.scale.setScalar(1.08);
        this.ppeHelmet.position.set(0, 0.062, 0.010);
        this.ppeHelmet.renderOrder = 2;
        this.ppeHelmet.visible = Boolean(this.ppeVisibility.helmet);
        headBone.add(this.ppeHelmet);
      }

      if (leftHandBone) {
        // Left protective gauntlet glove
        this.ppeGloveL = createProtectiveGloves("left");
        this.ppeGloveL.scale.setScalar(1.22);
        this.ppeGloveL.position.set(0, 0, 0);
        this.ppeGloveL.renderOrder = 2;
        this.ppeGloveL.visible = Boolean(this.ppeVisibility.gloves);
        leftHandBone.add(this.ppeGloveL);
      }

      if (rightHandBone) {
        // Right protective gauntlet glove
        this.ppeGloveR = createProtectiveGloves("right");
        this.ppeGloveR.scale.setScalar(1.22);
        this.ppeGloveR.position.set(0, 0, 0);
        this.ppeGloveR.renderOrder = 2;
        this.ppeGloveR.visible = Boolean(this.ppeVisibility.gloves);
        rightHandBone.add(this.ppeGloveR);
      }

      // Re-apply any PPE item visibility states set before or during load
      for (const [eqType, isVisible] of Object.entries(this.ppeVisibility)) {
        this.setPPEItemVisible(eqType, isVisible);
      }

      this.setAvatarBodyMode(this.avatarBodyMode);
      this.report = report;
      this.onStatus({ state: "ready", ...report });
    } catch (error) {
      this.onStatus({ state: "error", message: error.message, ...(this.report || {}) });
      console.error("Could not load a usable humanoid avatar.", error);
    }
  }

  setAvatarBodyMode(mode) {
    this.avatarBodyMode = mode; // "visible" | "occluder" | "hidden"
    if (!this.model) return;
    this.model.traverse((obj) => {
      if (obj.isSkinnedMesh) {
        const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const mat of materials) {
          if (!mat) continue;
          if (mode === "occluder") {
            obj.visible = true;
            mat.colorWrite = false;
            mat.depthWrite = true;
            mat.polygonOffset = true;
            mat.polygonOffsetFactor = 2.0;
            mat.polygonOffsetUnits = 2.0;
            obj.renderOrder = 0;
          } else if (mode === "visible") {
            obj.visible = true;
            mat.colorWrite = true;
            mat.depthWrite = true;
            mat.polygonOffset = false;
            obj.renderOrder = 1;
          } else if (mode === "hidden") {
            obj.visible = false;
          }
          mat.needsUpdate = true;
        }
      }
    });
    this.renderer?.render(this.scene, this.camera);
  }

  setRenderMode(mode) {
    this.renderMode = mode;
    if (!this.renderer || !this.scene || !this.camera) return;

    if (mode === "environment") {
      this.scene.background = new THREE.Color("#141b1c");
      this.renderer.setClearColor(0x141b1c, 1);
      if (this.environmentGrid) {
        this.environmentGrid.visible = true;
      }
      this.camera.position.z = 3.2;
      this.camera.near = 0.01;
      this.camera.far = 100;
      this.camera.fov = 45;
      this.setAvatarBodyMode("visible");
    } else if (mode === "camera-avatar") {
      this.scene.background = null;
      this.renderer.setClearColor(0x000000, 0);
      if (this.environmentGrid) {
        this.environmentGrid.visible = false;
      }
      this.camera.position.z = 1000;
      this.camera.near = 0.1;
      this.camera.far = 5000;
      this.setAvatarBodyMode("visible");
    } else {
      // Default "camera" mode: AR PPE equipment overlay, 3D body acts as invisible depth occluder
      this.scene.background = null;
      this.renderer.setClearColor(0x000000, 0);
      if (this.environmentGrid) {
        this.environmentGrid.visible = false;
      }
      this.camera.position.z = 1000;
      this.camera.near = 0.1;
      this.camera.far = 5000;
      this.setAvatarBodyMode("occluder");
    }
    this.camera.updateProjectionMatrix();
    this.resize();
    this.renderer?.render(this.scene, this.camera);
  }

  setDebug({ avatarSkeleton = false, jointPositions = false } = {}) {
    if (this.skeletonHelper) {
      this.skeletonHelper.visible = avatarSkeleton;
    }
    this.showJointPositions = jointPositions;
  }

  setPPEItemVisible(type, visible) {
    const isVis = Boolean(visible);
    this.ppeVisibility[type] = isVis;

    const applyVisibility = (group) => {
      if (!group) return;
      group.visible = isVis;
      group.traverse((child) => {
        child.visible = isVis;
        if (child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          for (const m of mats) {
            if (m) m.needsUpdate = true;
          }
        }
      });
    };

    if (type === "helmet" && this.ppeHelmet) {
      applyVisibility(this.ppeHelmet);
    } else if (type === "vest" && this.ppeVest) {
      applyVisibility(this.ppeVest);
    } else if (type === "gloves") {
      applyVisibility(this.ppeGloveL);
      applyVisibility(this.ppeGloveR);
    } else if (type === "respirator") {
      this.scbaVisible = isVis;
      applyVisibility(this.scbaCylinder);
      applyVisibility(this.scbaMask);
    }

    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  setSCBAVisible(visible) {
    this.setPPEItemVisible("respirator", visible);
  }

  createCalibrationHelpers(mapping) {
    const mappedBones = new Set(Object.values(mapping).filter(Boolean));
    for (const bone of mappedBones) {
      const axes = new THREE.AxesHelper(0.12);
      axes.visible = this.calibrationDebugVisible;
      bone.add(axes);
      this.calibrationHelpers.push(axes);
    }

    const head = mapping.head;
    if (head) {
      const headWorldRotation = head.getWorldQuaternion(new THREE.Quaternion());
      const localForward = this.retargeter.diagnostics.modelForward
        .clone()
        .applyQuaternion(headWorldRotation.invert())
        .normalize();
      const arrow = new THREE.ArrowHelper(
        localForward,
        new THREE.Vector3(),
        0.22,
        0xffe36d,
        0.06,
        0.035,
      );
      arrow.visible = this.calibrationDebugVisible;
      head.add(arrow);
      this.calibrationHelpers.push(arrow);
    }
  }

  setCalibrationDebug(visible) {
    this.calibrationDebugVisible = visible;
    for (const helper of this.calibrationHelpers) {
      helper.visible = visible;
    }
  }

  getDebugBonePoints(stage) {
    if (!this.retargeter) return {};
    this.group.updateMatrixWorld(true);
    this.camera.updateMatrixWorld(true);
    const points = {};
    const projectBone = (role, bone) => {
      if (!bone) return;
      bone.getWorldPosition(this.debugPosition);
      this.debugPosition.project(this.camera);
      if (this.debugPosition.z < -1 || this.debugPosition.z > 1) return;
      let x = ((this.debugPosition.x + 1) / 2) * stage.width;
      if (this.renderMode === "camera" || this.renderMode === "camera-avatar") x = stage.width - x;
      points[role] = {
        x,
        y: ((1 - this.debugPosition.y) / 2) * stage.height,
      };
    };
    for (const role of [
      "leftForearm",
      "rightForearm",
      "leftHand",
      "rightHand",
      "hips",
      "leftUpperLeg",
      "leftLeg",
      "rightUpperLeg",
      "rightLeg",
    ]) {
      projectBone(role, this.retargeter.mapping[role]);
    }
    projectBone("leftShoulder", this.retargeter.anchors.leftShoulder);
    projectBone("rightShoulder", this.retargeter.anchors.rightShoulder);
    return points;
  }

  update(landmarks, worldLandmarks, video = this.video) {
    if (!landmarks || !worldLandmarks) {
      this.tracked = null;
      return;
    }
    this.tracked = { landmarks, worldLandmarks, video };
    // Immediately calculate placement and retargeting on tracking update
    this.placeAvatar();
  }

  setUserCalibration({ size, depth } = {}) {
    if (typeof size === "number" && Number.isFinite(size) && size > 0) {
      this.userScaleMultiplier = size;
    }
    if (typeof depth === "number" && Number.isFinite(depth)) {
      this.userDepthOffset = depth;
    }
    if (this.tracked) {
      this.placeAvatar();
    }
  }

  setSegmentationBounds(bounds) {
    this.silhouetteBounds = bounds;
  }

  setMaskCalibration({ size = 1.0, offsetX = 0, offsetY = 0, depth = 0 } = {}) {
    this.maskCalibration = { size, offsetX, offsetY, depth };
    if (this.scbaMask) {
      const baseScale = 0.87;
      const effectiveScale = baseScale * size;
      this.scbaMask.scale.setScalar(effectiveScale);
      // In headBone local space: convert pixel sliders to meters
      const pxToMeters = 0.0008;
      this.scbaMask.position.set(
        offsetX * pxToMeters,
        -0.018 - offsetY * pxToMeters,
        0.120 + depth * pxToMeters,
      );
    }
  }

  resetCalibration() {
    this.calibratedPixelsPerMeter = null;
    this.lastDetectedDepth = 0;
    this.userScaleMultiplier = 1.50;
    this.userDepthOffset = 0;
    this.originFilter?.reset();
    this.scaleFilter?.reset();
    this.widthFactorFilter?.reset();
    this.retargeter?.reset();
    this.group?.position.set(0, 0, 0);
    this.group?.scale.setScalar(1);
  }

  placeAvatar() {
    if (!this.retargeter || !this.tracked) {
      return;
    }
    const delta = Math.min(Math.max(this.frameDelta || 0.016, 0.001), 0.1);
    const { landmarks, worldLandmarks, video } = this.tracked;
    const stage = this.container.getBoundingClientRect();
    if (!stage.width || !stage.height) {
      return;
    }
    const leftHip = mapVisibleLandmarkToStage(
      landmarks[LANDMARKS.leftHip],
      video,
      stage,
    );
    const rightHip = mapVisibleLandmarkToStage(
      landmarks[LANDMARKS.rightHip],
      video,
      stage,
    );
    let leftShoulder = mapVisibleLandmarkToStage(
      landmarks[LANDMARKS.leftShoulder],
      video,
      stage,
    );
    let rightShoulder = mapVisibleLandmarkToStage(
      landmarks[LANDMARKS.rightShoulder],
      video,
      stage,
    );
    // Boundary outlier rejection: reject joints clamped against screen borders
    if (leftShoulder && (leftShoulder.y <= 12 || leftShoulder.x <= 12 || leftShoulder.x >= stage.width - 12)) {
      leftShoulder = null;
    }
    if (rightShoulder && (rightShoulder.y <= 12 || rightShoulder.x <= 12 || rightShoulder.x >= stage.width - 12)) {
      rightShoulder = null;
    }
    const hasHipCenter = Boolean(leftHip && rightHip);
    const hasShoulderCenter = Boolean(leftShoulder && rightShoulder);
    let shoulderWidth =
      hasShoulderCenter
        ? Math.hypot(
            leftShoulder.x - rightShoulder.x,
            leftShoulder.y - rightShoulder.y,
          )
        : 0;
    if (shoulderWidth > stage.width * 0.55) {
      shoulderWidth = this.lastDetectedShoulderWidth || stage.width * 0.28;
    }
    const hipWidth = hasHipCenter
      ? Math.hypot(leftHip.x - rightHip.x, leftHip.y - rightHip.y)
      : 0;
    const projectLandmark = (index) => {
      const landmark = landmarks[index];
      if (
        !isLandmarkUsable(landmark) ||
        landmark.x < 0 ||
        landmark.x > 1 ||
        landmark.y < 0 ||
        landmark.y > 1
      ) {
        return null;
      }
      return mapLandmarkToStage(landmark, video, stage, false);
    };
    let imageNose = projectLandmark(0);
    const imageLeftAnkle = projectLandmark(27);
    const imageRightAnkle = projectLandmark(28);
    const imageLeftHip = projectLandmark(LANDMARKS.leftHip);
    const imageRightHip = projectLandmark(LANDMARKS.rightHip);
    const imageLeftShoulder = projectLandmark(LANDMARKS.leftShoulder);
    const imageRightShoulder = projectLandmark(LANDMARKS.rightShoulder);
    const imageHipCenter =
      imageLeftHip && imageRightHip
        ? {
            x: (imageLeftHip.x + imageRightHip.x) / 2,
            y: (imageLeftHip.y + imageRightHip.y) / 2,
          }
        : null;
    const imageShoulderCenter =
      imageLeftShoulder && imageRightShoulder
        ? {
            x: (imageLeftShoulder.x + imageRightShoulder.x) / 2,
            y: (imageLeftShoulder.y + imageRightShoulder.y) / 2,
          }
        : null;

    // Anatomical Outlier Rejection:
    // In human anatomy, the nose cannot be below the shoulders or hips in an upright stance (screen y increases downward).
    if (imageNose && imageShoulderCenter && imageNose.y > imageShoulderCenter.y + 0.02 * stage.height) {
      imageNose = null;
    }
    const ankleY =
      imageLeftAnkle && imageRightAnkle
        ? (imageLeftAnkle.y + imageRightAnkle.y) / 2
        : null;
    const bodyHeight =
      imageNose && ankleY !== null && ankleY > imageNose.y
        ? ankleY - imageNose.y + shoulderWidth * 0.18
        : 0;
    const torsoHeight =
      imageHipCenter && imageShoulderCenter
        ? Math.hypot(
            imageShoulderCenter.x - imageHipCenter.x,
            imageShoulderCenter.y - imageHipCenter.y,
          )
        : 0;
    const headShoulderHeight =
      imageNose && imageShoulderCenter
        ? Math.hypot(
            imageNose.x - imageShoulderCenter.x,
            imageNose.y - imageShoulderCenter.y,
          )
        : 0;
    this.lastDetectedShoulderWidth = shoulderWidth;
    this.lastDetectedBodyHeight = bodyHeight;
    this.lastDetectedTorsoHeight = torsoHeight;
    this.lastDetectedHeadShoulderHeight = headShoulderHeight;

    // 1. Measure true 3D metric dimensions of the detected human from world landmarks
    let humanTorsoMeters = 0.50;
    let humanShoulderMeters = 0.40;
    const hasWorldShoulders =
      isLandmarkUsable(worldLandmarks[11]) && isLandmarkUsable(worldLandmarks[12]);
    const hasWorldHips =
      isLandmarkUsable(worldLandmarks[23]) && isLandmarkUsable(worldLandmarks[24]);

    if (hasWorldShoulders) {
      const dx = worldLandmarks[11].x - worldLandmarks[12].x;
      const dy = worldLandmarks[11].y - worldLandmarks[12].y;
      const dz = worldLandmarks[11].z - worldLandmarks[12].z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist > 0.15 && dist < 0.85) {
        humanShoulderMeters = dist;
      }
    }
    if (hasWorldShoulders && hasWorldHips) {
      const smX = (worldLandmarks[11].x + worldLandmarks[12].x) * 0.5;
      const smY = (worldLandmarks[11].y + worldLandmarks[12].y) * 0.5;
      const smZ = (worldLandmarks[11].z + worldLandmarks[12].z) * 0.5;
      const hmX = (worldLandmarks[23].x + worldLandmarks[24].x) * 0.5;
      const hmY = (worldLandmarks[23].y + worldLandmarks[24].y) * 0.5;
      const hmZ = (worldLandmarks[23].z + worldLandmarks[24].z) * 0.5;
      const dist = Math.hypot(smX - hmX, smY - hmY, smZ - hmZ);
      if (dist > 0.2 && dist < 0.9) {
        humanTorsoMeters = dist;
      }
    }
    this.calibratedTorsoMeters = humanTorsoMeters;
    this.calibratedShoulderMeters = humanShoulderMeters;

    // 2. Unproject 2D foreshortening using detected torso yaw/pitch so the avatar doesn't collapse when turning
    const yaw = this.retargeter?.diagnostics.torsoYaw || 0;
    const pitch = this.retargeter?.diagnostics.torsoPitch || 0;
    const cosYaw = Math.max(Math.abs(Math.cos(yaw)), 0.65);
    const cosPitch = Math.max(Math.abs(Math.cos(pitch)), 0.65);

    const unprojectedShoulderWidth = shoulderWidth / cosYaw;
    const unprojectedTorsoHeight = torsoHeight / cosPitch;

    // 3. Compute accurate scale matching the human on screen
    let scale = 0;
    const detectedHeadToAnkle =
      imageNose && ankleY !== null && ankleY > imageNose.y
        ? ankleY - imageNose.y
        : 0;

    const ankleScale =
      detectedHeadToAnkle > 0 && this.modelHeadToAnkleHeight > 0.05
        ? detectedHeadToAnkle / this.modelHeadToAnkleHeight
        : 0;
    const shoulderScale =
      unprojectedShoulderWidth > 0 && this.modelShoulderWidth > 0.02
        ? unprojectedShoulderWidth / this.modelShoulderWidth
        : 0;
    const torsoScale =
      unprojectedTorsoHeight > 0 && this.modelTorsoHeight > 0.02
        ? unprojectedTorsoHeight / this.modelTorsoHeight
        : 0;
    const headScale =
      headShoulderHeight > 0 && this.modelHeadShoulderHeight > 0.02
        ? headShoulderHeight / this.modelHeadShoulderHeight
        : 0;

    if (ankleScale > 0) {
      scale = ankleScale;
      this.lastScaleReference = "head-to-ankles";
    } else if (torsoScale > 0) {
      scale = torsoScale;
      this.lastScaleReference = "torso height";
    } else if (headScale > 0) {
      scale = headScale;
      this.lastScaleReference = "head-to-shoulders";
    } else if (shoulderScale > 0) {
      scale = shoulderScale;
      this.lastScaleReference = "shoulder width";
    } else if (hipWidth > 0 && this.modelHipWidth > 0.02) {
      scale = hipWidth / this.modelHipWidth;
      this.lastScaleReference = "hip width";
    } else {
      scale = this.group.scale.y || 100;
      this.lastScaleReference = "fallback scale";
    }

    scale = Math.max(scale, 20);

    // Anchor scale: prevent sudden size breathing or contracting in profile turns
    if (this.lastStableScale && Number.isFinite(this.lastStableScale) && this.lastStableScale > 20) {
      scale = THREE.MathUtils.clamp(scale, this.lastStableScale * 0.94, this.lastStableScale * 1.06);
    }
    this.lastStableScale = scale;

    // Anatomical width scaling: fit the human's shoulder width naturally without distortion
    let widthFactor = 1.0;
    if (shoulderWidth >= 0.12 && Math.abs(yaw) < 0.7 && unprojectedShoulderWidth > 0 && this.modelShoulderWidth > 0) {
      const rawShoulderScale = unprojectedShoulderWidth / this.modelShoulderWidth;
      widthFactor = THREE.MathUtils.clamp(rawShoulderScale / scale, 0.90, 1.15);
    }

    // Adaptive filtering on scale and widthFactor to eliminate size flicker & breathing
    const filteredScale = this.scaleFilter.filter(scale, delta);
    const filteredWidthFactor = this.widthFactorFilter.filter(widthFactor, delta);
    const scaleX = filteredScale * filteredWidthFactor;
    const scaleY = filteredScale;
    this.lastHorizontalScaleReference = "shoulder width";

    // 4. Calculate depth (Z)
    let targetDepth = 0;
    if (this.renderMode === "environment") {
      targetDepth = THREE.MathUtils.clamp(this.userDepthOffset / 250, -2.5, 2.5);
    } else {
      targetDepth = THREE.MathUtils.clamp(this.userDepthOffset, -450, 450);
    }
    this.lastDetectedDepth = targetDepth;

    // 5. Calculate screen anchor position (originX, originY) for the avatar shoulder root
    let originX = stage.width / 2;
    let originY = stage.height / 2;

    if (hasShoulderCenter) {
      // Anchoring directly on shoulder center stabilizes upper body PPE (vest, helmet, mask)
      originX = imageShoulderCenter.x;
      originY = imageShoulderCenter.y;
    } else if (hasHipCenter) {
      // Shoulders out of frame: anchor relative to hips
      originX = imageHipCenter.x;
      originY = imageHipCenter.y - scaleY * this.modelTorsoHeight;
    } else if (imageNose) {
      // Only head/face visible: anchor relative to nose
      originX = imageNose.x;
      originY = imageNose.y + scaleY * this.modelHeadShoulderHeight;
    }

    // Adaptive filtering on screen anchor to eliminate jitter while standing still
    const filteredOrigin = this.originFilter.filter(originX, originY, delta);
    originX = filteredOrigin.x;
    originY = filteredOrigin.y;

    // 6. Project to Three.js coordinates with perspective compensation
    const visibleHeight =
      this.renderMode === "environment"
        ? 2 * this.camera.position.z * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))
        : 0;

    if (this.renderMode === "environment") {
      const targetScale = (scaleY / stage.height) * visibleHeight * this.userScaleMultiplier * 1.0;
      this.scaleTarget.setScalar(targetScale);
      this.positionTarget.set(
        ((originX / stage.width) * 2 - 1) * this.camera.aspect * (visibleHeight / 2) * 0.85,
        (0.5 - originY / stage.height) * visibleHeight * 0.85,
        targetDepth,
      );
    } else {
      // In camera mode: cancel out perspective projection factor so the model projects
      // to EXACTLY (originX, originY) on screen at ANY depth!
      const perspectiveFactor = Math.max((1000 - targetDepth) / 1000, 0.2);
      const modelScaleBoost = 1.0;
      const finalScaleX = scaleX * perspectiveFactor * this.userScaleMultiplier * modelScaleBoost;
      const finalScaleY = scaleY * perspectiveFactor * this.userScaleMultiplier * modelScaleBoost;
      this.scaleTarget.set(finalScaleX, finalScaleY, finalScaleX);

      this.positionTarget.set(
        (originX - stage.width / 2) * perspectiveFactor,
        (stage.height / 2 - originY) * perspectiveFactor,
        targetDepth,
      );
    }

    // 7. Responsive continuous time-based smoothing with rapid fast-movement tracking
    let posAlpha = 1 - Math.exp(-24 * delta);
    let scaleAlpha = 1 - Math.exp(-20 * delta);

    const posDist = this.group.position.distanceTo(this.positionTarget);
    // When moving quickly (>20px shift), increase responsiveness so mask stays locked to the face
    const boost = THREE.MathUtils.clamp((posDist - 20) / 40, 0, 0.65);
    posAlpha = THREE.MathUtils.clamp(posAlpha + boost, 0.20, 0.95);

    this.group.scale.lerp(this.scaleTarget, scaleAlpha);
    this.group.position.lerp(this.positionTarget, posAlpha);

    this.retargeter.update(landmarks, worldLandmarks, delta);
  }

  resize() {
    if (!this.renderer) {
      return;
    }
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) {
      return;
    }
    this.camera.aspect = width / height;
    if (this.renderMode === "camera" || this.renderMode === "camera-avatar") {
      this.camera.fov = (2 * Math.atan(height / (2 * this.camera.position.z)) * 180) / Math.PI;
    }
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  animate(now) {
    this.animationId = requestAnimationFrame(this.animate);
    const delta = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;
    this.frameDelta = delta;
    if (this.tracked) {
      this.placeAvatar();
    } else if (this.retargeter) {
      this.retargeter.update(null, null, delta);
      this.group.position.lerp(this.neutralPosition, 1 - Math.exp(-2 * delta));
      this.scaleTarget.setScalar(this.renderMode === "environment" ? 1 : 0);
      this.group.scale.lerp(this.scaleTarget, 1 - Math.exp(-2 * delta));
    }
    this.frameCount += 1;
    if (now - this.fpsTime >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / (now - this.fpsTime));
      this.frameCount = 0;
      this.fpsTime = now;
    }
    this.renderer?.render(this.scene, this.camera);
    if (this.onPostRender) {
      this.onPostRender();
    }
  }

  close() {
    cancelAnimationFrame(this.animationId);
    this.resizeObserver?.disconnect();
    this.renderer?.dispose();
    this.retargeter?.reset();
  }
}
