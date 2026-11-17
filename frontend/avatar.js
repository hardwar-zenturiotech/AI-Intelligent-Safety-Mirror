import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mapLandmarkToStage } from "./camera.js";
import { isLandmarkUsable } from "./poseTracker.js";
import { MotionRetargeter, getRigReport, inspectHumanoidRig } from "./retargeting.js";

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

export class Avatar {
  constructor(container, onStatus) {
    this.container = container;
    this.onStatus = onStatus;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
    this.camera.position.z = 1000;
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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
    this.modelHeadShoulderHeight = 0.18;
    this.modelShoulderAnchors = "upper-arm bones";
    this.lastScaleReference = "waiting for pose";
    this.lastHorizontalScaleReference = "waiting for pose";
    this.lastDetectedShoulderWidth = 0;
    this.lastDetectedBodyHeight = 0;
    this.lastDetectedTorsoHeight = 0;
    this.lastDetectedHeadShoulderHeight = 0;
    this.lastJointOffsets = {};
    this.calibrationHelpers = [];
    this.calibrationDebugVisible = false;
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
      const leftShoulderBone = rig.anchors.leftShoulder || rig.mapping.leftUpperArm;
      const rightShoulderBone = rig.anchors.rightShoulder || rig.mapping.rightUpperArm;
      if (leftShoulderBone && rightShoulderBone) {
        const leftShoulder = leftShoulderBone.getWorldPosition(new THREE.Vector3());
        const rightShoulder = rightShoulderBone.getWorldPosition(new THREE.Vector3());
        const measuredShoulderWidth = leftShoulder.distanceTo(rightShoulder);
        if (Number.isFinite(measuredShoulderWidth) && measuredShoulderWidth > 0.01) {
          this.modelShoulderWidth = measuredShoulderWidth;
        }
        this.modelShoulderHeight = (leftShoulder.y + rightShoulder.y) / 2;
        this.modelShoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2;
        this.modelTorsoHeight = Math.abs(this.modelShoulderHeight - this.modelHipHeight);
        const headPosition = rig.mapping.head?.getWorldPosition(new THREE.Vector3());
        const headDistance = headPosition
          ? Math.abs(headPosition.y - this.modelShoulderHeight)
          : 0;
        this.modelHeadShoulderHeight =
          headDistance > 0.02 ? headDistance : Math.abs(1 - this.modelShoulderHeight);
        this.modelShoulderAnchors =
          rig.anchors.leftShoulder && rig.anchors.rightShoulder
            ? `${leftShoulderBone.name} / ${rightShoulderBone.name}`
            : "upper-arm bones (fallback)";
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
      this.report = report;
      this.onStatus({ state: "ready", ...report });
    } catch (error) {
      this.onStatus({ state: "error", message: error.message, ...(this.report || {}) });
      console.error("Could not load a usable humanoid avatar.", error);
    }
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
    } else {
      this.scene.background = null;
      if (this.environmentGrid) {
        this.environmentGrid.visible = false;
      }
      this.camera.position.z = 1000;
      this.camera.near = 0.1;
      this.camera.far = 5000;
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
  }

  resetCalibration() {
    this.retargeter?.reset();
    this.group?.position.set(0, 0, 0);
    this.group?.scale.setScalar(1);
  }

  placeAvatar() {
    if (!this.retargeter || !this.tracked) {
      return;
    }
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
    const scaleFromShoulders =
      shoulderWidth > 0 && this.modelShoulderWidth > 0
        ? shoulderWidth / this.modelShoulderWidth
        : 0;
    const scaleFromHips =
      hipWidth > 0 && this.modelHipWidth > 0 ? hipWidth / this.modelHipWidth : 0;
    const widthScale = scaleFromShoulders || scaleFromHips;
    this.lastHorizontalScaleReference =
      scaleFromShoulders > 0
        ? "shoulder width"
        : scaleFromHips > 0
          ? "hip width"
          : "no visible width landmarks";
    let scaleY = 0;
    if (bodyHeight > 0 && this.modelHeight > 0) {
      scaleY = bodyHeight / this.modelHeight;
      this.lastScaleReference = "nose-to-ankles";
    } else if (torsoHeight > 0 && this.modelTorsoHeight > 0.02) {
      scaleY = torsoHeight / this.modelTorsoHeight;
      this.lastScaleReference = "shoulder-to-hip height";
    } else if (headShoulderHeight > 0 && this.modelHeadShoulderHeight > 0.02) {
      scaleY = headShoulderHeight / this.modelHeadShoulderHeight;
      this.lastScaleReference = "nose-to-shoulder height";
    } else {
      scaleY = widthScale;
      this.lastScaleReference = shoulderWidth > 0 || hipWidth > 0
        ? "shoulder/hip width"
        : "no visible scale landmarks";
    }
    const hasScaleMeasurement =
      bodyHeight > 0 ||
      torsoHeight > 0 ||
      headShoulderHeight > 0 ||
      widthScale > 0;
    const scaleX = hasScaleMeasurement
      ? Math.max(widthScale || scaleY, 20)
      : this.group.scale.x;
    scaleY = hasScaleMeasurement ? Math.max(scaleY, 20) : this.group.scale.y;
    const visibleHeight =
      this.renderMode === "environment"
        ? 2 * this.camera.position.z * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))
        : 0;
    if (hasScaleMeasurement) {
      if (this.renderMode === "environment") {
        const targetScale = THREE.MathUtils.clamp(
          (scaleY / stage.height) * visibleHeight,
          0.7,
          visibleHeight * 1.4,
        );
        this.scaleTarget.setScalar(targetScale);
      } else {
        this.scaleTarget.set(scaleX, Math.max(scaleY, 20), scaleX);
      }
      this.group.scale.lerp(this.scaleTarget, 0.18);
    }
    if (hasHipCenter || hasShoulderCenter) {
      let originX;
      let originY;
      if (hasHipCenter) {
        originX = (leftHip.x + rightHip.x) / 2;
        originY = (leftHip.y + rightHip.y) / 2;
      } else {
        const shoulderPixelsPerUnit =
          this.renderMode === "environment"
            ? (this.group.scale.x * stage.height) / visibleHeight
            : this.group.scale.x;
        const shoulderPixelsPerUnitY =
          this.renderMode === "environment"
            ? shoulderPixelsPerUnit
            : this.group.scale.y;
        originX =
          (leftShoulder.x + rightShoulder.x) / 2 -
          this.modelShoulderCenterX * shoulderPixelsPerUnit;
        originY =
          (leftShoulder.y + rightShoulder.y) / 2 +
          this.modelShoulderHeight * shoulderPixelsPerUnitY;
      }
      if (this.renderMode === "environment") {
        this.positionTarget.set(
          ((originX / stage.width) * 2 - 1) * this.camera.aspect * 1.2,
          (0.5 - originY / stage.height) * 2.4,
          0,
        );
      } else {
        this.positionTarget.set(originX - stage.width / 2, stage.height / 2 - originY, 0);
      }
      this.group.position.lerp(this.positionTarget, 0.18);
    }
    this.retargeter.update(landmarks, worldLandmarks, this.frameDelta);
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
