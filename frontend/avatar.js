import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mapLandmarkToStage } from "./camera.js";
import { isLandmarkUsable } from "./poseTracker.js";
import { MotionRetargeter, getRigReport, inspectHumanoidRig } from "./retargeting.js";
import { createSCBACylinderApparatus, createSCBAFaceMask } from "./scbaApparatus.js";

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
  constructor(minCutoff = 1.0, beta = 4.0) {
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
    const speed = deltaSeconds > 0 ? diff / deltaSeconds : 0;
    const cutoff = this.minCutoff + this.beta * speed;
    const alpha = THREE.MathUtils.clamp(
      1 - Math.exp(-2 * Math.PI * cutoff * deltaSeconds),
      0.06,
      1.0,
    );

    this.current += (target - this.current) * alpha;
    return this.current;
  }
}

class AdaptivePointFilter {
  constructor(minCutoff = 1.2, beta = 5.0) {
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
    const speed = deltaSeconds > 0 ? dist / deltaSeconds : 0;
    const cutoff = this.minCutoff + this.beta * speed;
    const alpha = THREE.MathUtils.clamp(
      1 - Math.exp(-2 * Math.PI * cutoff * deltaSeconds),
      0.06,
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
    this.modelShoulderWidth = 0.38;
    this.modelHipWidth = 0.24;
    this.modelShoulderHeight = 0.68;
    this.modelShoulderCenterX = 0;
    this.modelHeight = 1;
    this.modelHipHeight = 0;
    this.modelTorsoHeight = 0.32;
    this.modelHeadHeight = 0.38;
    this.modelHeadShoulderHeight = 0.18;
    this.modelHeadToAnkleHeight = 0.62;
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
    this.userScaleMultiplier = 1.0;
    this.userDepthOffset = 0;
    this.lastJointOffsets = {};
    this.calibrationHelpers = [];
    this.calibrationDebugVisible = false;
    this.originFilter = new AdaptivePointFilter(1.5, 6.0);
    this.scaleFilter = new AdaptiveScalarFilter(1.2, 5.0);
    this.widthFactorFilter = new AdaptiveScalarFilter(1.2, 5.0);
    this.scbaVisible = false;
    this.scbaCylinder = null;
    this.scbaMask = null;
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
      const hips = rig.mapping.hips;
      if (!hips) {
        throw new Error("No Hips/Pelvis bone was detected in the humanoid skeleton.");
      }

      const normalization = new THREE.Group();
      normalization.add(this.model);
      this.group.add(normalization);
      normalization.updateMatrixWorld(true);
      const hipPosition = hips.getWorldPosition(new THREE.Vector3());
      normalization.scale.setScalar(1 / size.y);
      normalization.position.set(
        -hipPosition.x / size.y,
        -hipPosition.y / size.y,
        -hipPosition.z / size.y,
      );
      normalization.updateMatrixWorld(true);
      this.modelHipHeight = hips.getWorldPosition(new THREE.Vector3()).y;
      // Glenohumeral shoulder joints (upper arm pivots) correspond directly to MediaPipe landmarks 11 & 12
      const leftUpperArm = rig.mapping.leftUpperArm;
      const rightUpperArm = rig.mapping.rightUpperArm;
      const leftShoulderJoint = leftUpperArm || rig.anchors.leftShoulder;
      const rightShoulderJoint = rightUpperArm || rig.anchors.rightShoulder;
      if (leftShoulderJoint && rightShoulderJoint) {
        const leftShoulder = leftShoulderJoint.getWorldPosition(new THREE.Vector3());
        const rightShoulder = rightShoulderJoint.getWorldPosition(new THREE.Vector3());
        const measuredShoulderWidth = leftShoulder.distanceTo(rightShoulder);
        if (Number.isFinite(measuredShoulderWidth) && measuredShoulderWidth > 0.01) {
          this.modelShoulderWidth = measuredShoulderWidth;
        }
        this.modelShoulderHeight = (leftShoulder.y + rightShoulder.y) / 2;
        this.modelShoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2;
        this.modelTorsoHeight = Math.abs(this.modelShoulderHeight - this.modelHipHeight);
        const headPosition = rig.mapping.head?.getWorldPosition(new THREE.Vector3());
        this.modelHeadHeight = headPosition ? headPosition.y : this.modelShoulderHeight + 0.12;
        this.modelHeadShoulderHeight = Math.abs(this.modelHeadHeight - this.modelShoulderHeight);
        this.modelShoulderAnchors = `${leftShoulderJoint.name} / ${rightShoulderJoint.name}`;
      }
      const leftAnkle = rig.mapping.leftLeg;
      const rightAnkle = rig.mapping.rightLeg;
      if (leftAnkle && rightAnkle) {
        const lPos = leftAnkle.getWorldPosition(new THREE.Vector3());
        const rPos = rightAnkle.getWorldPosition(new THREE.Vector3());
        const ankleY = (lPos.y + rPos.y) / 2;
        this.modelAnkleHeight = ankleY;
        this.modelHeadToAnkleHeight = Math.abs((this.modelHeadHeight || 0.38) - ankleY);
      } else {
        this.modelAnkleHeight = -0.50;
        this.modelHeadToAnkleHeight = 0.62;
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

      // Attach SCBA Face Mask and SCBA Back Cylinder Apparatus to the rig
      const chestBone = rig.mapping.chest || rig.mapping.spine;
      const headBone = rig.mapping.head;
      if (chestBone) {
        this.scbaCylinder = createSCBACylinderApparatus();
        // Snugly on the upper back/spine surface
        this.scbaCylinder.position.set(0, -0.06, -0.075);
        this.scbaCylinder.renderOrder = 2;
        this.scbaCylinder.visible = this.scbaVisible;
        chestBone.add(this.scbaCylinder);
      }
      if (headBone) {
        this.scbaMask = createSCBAFaceMask();
        // Positioned over the front face
        this.scbaMask.position.set(0, 0.015, 0.115);
        this.scbaMask.renderOrder = 2;
        this.scbaMask.visible = this.scbaVisible;
        headBone.add(this.scbaMask);
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
        if (mode === "occluder") {
          obj.visible = true;
          obj.material.colorWrite = false;
          obj.material.depthWrite = true;
          obj.renderOrder = 0;
        } else if (mode === "visible") {
          obj.visible = true;
          obj.material.colorWrite = true;
          obj.material.depthWrite = true;
          obj.renderOrder = 1;
        } else if (mode === "hidden") {
          obj.visible = false;
        }
      }
    });
  }

  setRenderMode(mode) {
    this.renderMode = mode;
    if (mode === "environment") {
      this.scene.background = new THREE.Color("#141b1c");
      if (this.environmentGrid) {
        this.environmentGrid.visible = true;
      }
      this.camera.position.z = 3.2;
      this.camera.near = 0.01;
      this.camera.far = 100;
      this.setAvatarBodyMode("visible");
    } else if (mode === "camera-avatar") {
      this.scene.background = null;
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
  }

  setDebug({ avatarSkeleton = false, jointPositions = false } = {}) {
    if (this.skeletonHelper) {
      this.skeletonHelper.visible = avatarSkeleton;
    }
    this.showJointPositions = jointPositions;
  }

  setSCBAVisible(visible) {
    this.scbaVisible = Boolean(visible);
    if (this.scbaCylinder) {
      this.scbaCylinder.visible = this.scbaVisible;
    }
    if (this.scbaMask) {
      this.scbaMask.visible = this.scbaVisible;
    }
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
      if (this.renderMode === "camera") x = stage.width - x;
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

  resetCalibration() {
    this.calibratedPixelsPerMeter = null;
    this.lastDetectedDepth = 0;
    this.userScaleMultiplier = 1.0;
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
    const leftShoulder = mapVisibleLandmarkToStage(
      landmarks[LANDMARKS.leftShoulder],
      video,
      stage,
    );
    const rightShoulder = mapVisibleLandmarkToStage(
      landmarks[LANDMARKS.rightShoulder],
      video,
      stage,
    );
    const hasHipCenter = leftHip && rightHip;
    const hasShoulderCenter = leftShoulder && rightShoulder;
    const shoulderWidth =
      hasShoulderCenter
        ? Math.hypot(
            leftShoulder.x - rightShoulder.x,
            leftShoulder.y - rightShoulder.y,
          )
        : 0;
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
    const imageNose = projectLandmark(0);
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
    const cosYaw = Math.max(Math.abs(Math.cos(yaw)), 0.35);
    const cosPitch = Math.max(Math.abs(Math.cos(pitch)), 0.35);

    const unprojectedShoulderWidth = shoulderWidth / cosYaw;
    const unprojectedTorsoHeight = torsoHeight / cosPitch;

    // 3. Compute accurate scale matching the human on screen
    let scale = 0;
    const detectedHeadToAnkle =
      imageNose && ankleY !== null && ankleY > imageNose.y
        ? ankleY - imageNose.y
        : 0;

    if (detectedHeadToAnkle > 0 && this.modelHeadToAnkleHeight > 0.05) {
      scale = detectedHeadToAnkle / this.modelHeadToAnkleHeight;
      this.lastScaleReference = "head-to-ankles";
    } else if (unprojectedTorsoHeight > 0 && this.modelTorsoHeight > 0.02) {
      scale = unprojectedTorsoHeight / this.modelTorsoHeight;
      this.lastScaleReference = "torso height";
    } else if (headShoulderHeight > 0 && this.modelHeadShoulderHeight > 0.02) {
      scale = headShoulderHeight / this.modelHeadShoulderHeight;
      this.lastScaleReference = "head-to-shoulders";
    } else if (unprojectedShoulderWidth > 0 && this.modelShoulderWidth > 0.02) {
      scale = unprojectedShoulderWidth / this.modelShoulderWidth;
      this.lastScaleReference = "shoulder width";
    } else if (hipWidth > 0 && this.modelHipWidth > 0.02) {
      scale = hipWidth / this.modelHipWidth;
      this.lastScaleReference = "hip width";
    } else {
      scale = this.group.scale.y || 100;
      this.lastScaleReference = "fallback scale";
    }

    scale = Math.max(scale, 20);

    // Anatomical width scaling: fit the human's shoulder width naturally without distortion
    const shoulderScale =
      unprojectedShoulderWidth > 0 && this.modelShoulderWidth > 0
        ? unprojectedShoulderWidth / this.modelShoulderWidth
        : scale;
    const widthFactor = THREE.MathUtils.clamp(shoulderScale / scale, 0.85, 1.25);

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

    // 5. Calculate screen anchor position (originX, originY) for the avatar Hips root
    let originX = stage.width / 2;
    let originY = stage.height / 2;

    if (hasShoulderCenter && hasHipCenter) {
      // Both shoulders and hips visible: center torso horizontally and align vertical torso span
      originX = (imageShoulderCenter.x + imageHipCenter.x) / 2;
      originY = (imageShoulderCenter.y + imageHipCenter.y) / 2 + (scaleY * this.modelTorsoHeight) / 2;
    } else if (hasShoulderCenter) {
      // Hips out of frame (sitting or waist-up): anchor directly below shoulders
      originX = imageShoulderCenter.x;
      originY = imageShoulderCenter.y + scaleY * this.modelShoulderHeight;
    } else if (hasHipCenter) {
      // Shoulders out of frame: anchor directly at hips
      originX = imageHipCenter.x;
      originY = imageHipCenter.y;
    } else if (imageNose) {
      // Only head/face visible: anchor directly below head
      originX = imageNose.x;
      originY = imageNose.y + scaleY * (this.modelHeadHeight || 0.38);
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
      const targetScale = (scaleY / stage.height) * visibleHeight * this.userScaleMultiplier;
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
      const finalScaleX = scaleX * perspectiveFactor * this.userScaleMultiplier;
      const finalScaleY = scaleY * perspectiveFactor * this.userScaleMultiplier;
      this.scaleTarget.set(finalScaleX, finalScaleY, finalScaleX);

      this.positionTarget.set(
        (originX - stage.width / 2) * perspectiveFactor,
        (stage.height / 2 - originY) * perspectiveFactor,
        targetDepth,
      );
    }

    // 7. Responsive continuous time-based smoothing
    let posAlpha = 1 - Math.exp(-26 * delta);
    let scaleAlpha = 1 - Math.exp(-22 * delta);

    const posDist = this.group.position.distanceTo(this.positionTarget);
    const boost = THREE.MathUtils.clamp((posDist - 10) / 40, 0, 0.4);
    posAlpha = THREE.MathUtils.clamp(posAlpha + boost, 0.15, 0.95);

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
    if (this.renderMode === "camera") {
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
  }

  close() {
    cancelAnimationFrame(this.animationId);
    this.resizeObserver?.disconnect();
    this.renderer?.dispose();
    this.retargeter?.reset();
  }
}
