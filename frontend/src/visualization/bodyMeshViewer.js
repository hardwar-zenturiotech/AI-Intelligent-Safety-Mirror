/**
 * EMOSH Body Mesh Reconstruction Viewer
 * Renders animated 3D human body surface mesh reconstructed by EMOSH-Tracker.
 * Updates dynamic vertex positions, renders surface topology, and attaches virtual SCBA PPE gear.
 */

import * as THREE from "three";
import {
  createSCBACylinderApparatus,
  createSCBAFaceMask,
  createSafetyHelmet,
  createSafetyVest,
  createProtectiveGloves,
} from "../../scbaApparatus.js";

export class BodyMeshViewer {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.group.name = "EMOSH_Reconstruction_Group";
    this.scene.add(this.group);

    this.meshSequence = null;
    this.currentFrame = 0;
    this.isPlaying = false;
    this.isLooping = true;
    this.playbackFps = 30;
    this.lastFrameTime = 0;
    this.visible = false;
    this.group.visible = false;

    // Body surface mesh
    this.geometry = new THREE.BufferGeometry();
    this.material = new THREE.MeshStandardMaterial({
      color: 0x4a7c91,
      roughness: 0.45,
      metalness: 0.15,
      side: THREE.DoubleSide,
      wireframe: false,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.group.add(this.mesh);

    // Optional wireframe overlay
    this.wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.25,
    });
    this.wireframeMesh = new THREE.Mesh(this.geometry, this.wireframeMaterial);
    this.wireframeMesh.visible = false;
    this.group.add(this.wireframeMesh);

    // Virtual SCBA PPE equipment attachments for EMOSH mesh
    this.ppeGroup = new THREE.Group();
    this.group.add(this.ppeGroup);

    this.initPPEEquipment();

    // Scale and position targets
    this.group.position.set(0, -0.45, 0);
    this.group.scale.setScalar(1.0);
  }

  initPPEEquipment() {
    // 1. Helmet
    this.helmet = createSafetyHelmet();
    this.helmet.scale.setScalar(1.02);
    this.ppeGroup.add(this.helmet);

    // 2. Full-face gas mask
    this.mask = createSCBAFaceMask();
    this.mask.scale.setScalar(0.88);
    this.ppeGroup.add(this.mask);

    // 3. Safety vest / harness
    this.vest = createSafetyVest();
    this.vest.scale.setScalar(1.15);
    this.ppeGroup.add(this.vest);

    // 4. Back-mounted cylinder
    this.cylinder = createSCBACylinderApparatus();
    this.cylinder.scale.setScalar(1.0);
    this.ppeGroup.add(this.cylinder);

    // 5. Gauntlet gloves
    this.gloveL = createProtectiveGloves("left");
    this.gloveL.scale.setScalar(1.1);
    this.ppeGroup.add(this.gloveL);

    this.gloveR = createProtectiveGloves("right");
    this.gloveR.scale.setScalar(1.1);
    this.ppeGroup.add(this.gloveR);

    // 6. Front apparatus
    this.front = null;
  }

  setCustomModelManager(manager) {
    this.customModelManager = manager;
    manager.onChange((evt) => {
      this.refreshSlotMesh(evt.slotId);
    });
  }

  refreshSlotMesh(slotId) {
    if (!this.customModelManager || !this.ppeGroup) return;

    const safeReplace = (oldObj, newObj) => {
      if (oldObj && oldObj.parent) {
        oldObj.parent.remove(oldObj);
      }
      if (newObj) {
        newObj.renderOrder = 2;
        this.ppeGroup.add(newObj);
      }
      return newObj;
    };

    if (slotId === "helmet") {
      const vis = this.helmet ? this.helmet.visible : true;
      this.helmet = safeReplace(this.helmet, this.customModelManager.instantiateSlotMesh("helmet"));
      if (this.helmet) this.helmet.visible = vis;
    } else if (slotId === "respirator") {
      const vis = this.mask ? this.mask.visible : true;
      this.mask = safeReplace(this.mask, this.customModelManager.instantiateSlotMesh("respirator"));
      if (this.mask) this.mask.visible = vis;
    } else if (slotId === "torso") {
      const vis = this.vest ? this.vest.visible : true;
      this.vest = safeReplace(this.vest, this.customModelManager.instantiateSlotMesh("torso"));
      if (this.vest) this.vest.visible = vis;
    } else if (slotId === "back") {
      const vis = this.cylinder ? this.cylinder.visible : true;
      this.cylinder = safeReplace(this.cylinder, this.customModelManager.instantiateSlotMesh("back"));
      if (this.cylinder) this.cylinder.visible = vis;
    } else if (slotId === "front") {
      const vis = this.front ? this.front.visible : true;
      this.front = safeReplace(this.front, this.customModelManager.instantiateSlotMesh("front"));
      if (this.front) this.front.visible = vis;
    } else if (slotId === "gloves") {
      const visL = this.gloveL ? this.gloveL.visible : true;
      const visR = this.gloveR ? this.gloveR.visible : true;
      this.gloveL = safeReplace(this.gloveL, this.customModelManager.instantiateSlotMesh("gloves", { hand: "left" }));
      this.gloveR = safeReplace(this.gloveR, this.customModelManager.instantiateSlotMesh("gloves", { hand: "right" }));
      if (this.gloveL) this.gloveL.visible = visL;
      if (this.gloveR) this.gloveR.visible = visR;
    }
  }

  setPPEVisible(item, visible) {
    if (item === "helmet" && this.helmet) this.helmet.visible = visible;
    if (item === "respirator") {
      if (this.mask) this.mask.visible = visible;
      if (this.cylinder) this.cylinder.visible = visible;
    }
    if (item === "vest") {
      if (this.vest) this.vest.visible = visible;
      if (this.front) this.front.visible = visible;
    }
    if (item === "front" && this.front) this.front.visible = visible;
    if (item === "gloves") {
      if (this.gloveL) this.gloveL.visible = visible;
      if (this.gloveR) this.gloveR.visible = visible;
    }
  }

  /**
   * Loads a reconstructed mesh sequence data object.
   */
  loadSequence(sequenceData) {
    if (!sequenceData || !sequenceData.frames || !sequenceData.frames.length) {
      throw new Error("Invalid EMOSH mesh sequence data");
    }

    this.meshSequence = sequenceData;
    this.playbackFps = sequenceData.fps || 30;
    this.currentFrame = 0;

    // Set topology indices
    const indices = new Uint32Array(sequenceData.indices);
    this.geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    // Allocate initial position and normal attributes
    const frame0 = sequenceData.frames[0];
    const positions = new Float32Array(frame0.vertices);
    const normals = new Float32Array(frame0.normals);

    this.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.normal.needsUpdate = true;
    this.geometry.computeBoundingBox();
    this.geometry.computeBoundingSphere();

    this.applyFrame(0);
    this.setVisible(true);
    this.play();
  }

  /**
   * Applies a specific frame's vertices and updates PPE anchor positioning.
   */
  applyFrame(frameIndex) {
    if (!this.meshSequence || !this.meshSequence.frames[frameIndex]) return;

    const frame = this.meshSequence.frames[frameIndex];
    const posAttr = this.geometry.attributes.position;
    const normAttr = this.geometry.attributes.normal;

    if (posAttr && frame.vertices) {
      const posArray = posAttr.array;
      for (let i = 0; i < frame.vertices.length; i++) {
        posArray[i] = frame.vertices[i];
      }
      posAttr.needsUpdate = true;
    }

    if (normAttr && frame.normals) {
      const normArray = normAttr.array;
      for (let i = 0; i < frame.normals.length; i++) {
        normArray[i] = frame.normals[i];
      }
      normAttr.needsUpdate = true;
    }

    // Attach virtual SCBA safety equipment to EMOSH anatomical landmarks
    const joints = frame.joints;
    if (joints) {
      // 1. Helmet: anchored to head center
      if (joints.head && this.helmet) {
        this.helmet.position.set(joints.head[0], joints.head[1] + 0.05, joints.head[2] + 0.01);
        if (typeof frame.yaw === "number") this.helmet.rotation.y = frame.yaw;
      }

      // 2. Face mask: anchored to front of head
      if (joints.head && this.mask) {
        this.mask.position.set(joints.head[0], joints.head[1] - 0.02, joints.head[2] + 0.11);
        if (typeof frame.yaw === "number") this.mask.rotation.y = frame.yaw;
      }

      // 3. High-vis vest: anchored to chest center
      if (joints.chest && this.vest) {
        this.vest.position.set(joints.chest[0], joints.chest[1] + 0.04, joints.chest[2] + 0.01);
        if (typeof frame.yaw === "number") this.vest.rotation.y = frame.yaw;
      }

      // 3b. Front equipment: anchored to front of chest
      if (joints.chest && this.front) {
        const yaw = typeof frame.yaw === "number" ? frame.yaw : 0;
        const frontZOffset = 0.125;
        this.front.position.set(
          joints.chest[0] + Math.sin(yaw) * frontZOffset,
          joints.chest[1] + 0.04,
          joints.chest[2] + Math.cos(yaw) * frontZOffset,
        );
        this.front.rotation.y = yaw;
      }

      // 4. SCBA cylinder: anchored to back of chest (-Z local offset rotated with body)
      if (joints.chest && this.cylinder) {
        const yaw = typeof frame.yaw === "number" ? frame.yaw : 0;
        const backZOffset = -0.125;
        this.cylinder.position.set(
          joints.chest[0] - Math.sin(yaw) * backZOffset,
          joints.chest[1] + 0.02,
          joints.chest[2] + Math.cos(yaw) * backZOffset,
        );
        this.cylinder.rotation.y = yaw;
      }

      // 5. Gauntlet gloves: anchored to hand joints
      if (joints.leftHand && this.gloveL) {
        this.gloveL.position.set(joints.leftHand[0], joints.leftHand[1], joints.leftHand[2]);
      }
      if (joints.rightHand && this.gloveR) {
        this.gloveR.position.set(joints.rightHand[0], joints.rightHand[1], joints.rightHand[2]);
      }
    }

    this.currentFrame = frameIndex;
  }

  update(deltaSeconds) {
    if (!this.isPlaying || !this.meshSequence || !this.visible) return;

    this.lastFrameTime += deltaSeconds;
    const frameInterval = 1.0 / this.playbackFps;

    if (this.lastFrameTime >= frameInterval) {
      const framesToAdvance = Math.floor(this.lastFrameTime / frameInterval);
      this.lastFrameTime %= frameInterval;

      let nextFrame = this.currentFrame + framesToAdvance;
      if (nextFrame >= this.meshSequence.frames.length) {
        if (this.isLooping) {
          nextFrame %= this.meshSequence.frames.length;
        } else {
          nextFrame = this.meshSequence.frames.length - 1;
          this.pause();
        }
      }
      this.applyFrame(nextFrame);
    }
  }

  play() {
    this.isPlaying = true;
  }

  pause() {
    this.isPlaying = false;
  }

  seek(frameIndex) {
    if (!this.meshSequence) return;
    const clamped = Math.max(0, Math.min(frameIndex, this.meshSequence.frames.length - 1));
    this.applyFrame(clamped);
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  setWireframe(wireframe) {
    this.wireframeMesh.visible = wireframe;
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
    this.wireframeMaterial.dispose();
    this.scene.remove(this.group);
  }
}
