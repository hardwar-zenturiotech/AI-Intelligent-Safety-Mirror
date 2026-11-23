import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import {
  createSafetyHelmet,
  createSCBAFaceMask,
  createSafetyVest,
  createProtectiveGloves,
  createSCBACylinderApparatus,
} from "../../scbaApparatus.js";

/**
 * Creates a lightweight default front accessory (chest-mounted monitor / PASS badge)
 */
export function createDefaultFrontAccessory(options = {}) {
  const group = new THREE.Group();
  group.name = "Front_Chest_Apparatus";

  const caseMat = new THREE.MeshStandardMaterial({
    color: 0x1f2428,
    roughness: 0.6,
    metalness: 0.2,
  });
  const screenColor = options.screenColor !== undefined ? options.screenColor : 0x00e5ff;
  const screenMat = new THREE.MeshStandardMaterial({
    color: screenColor,
    roughness: 0.2,
    metalness: 0.8,
    emissive: screenColor,
    emissiveIntensity: 0.35,
  });
  const yellowMat = new THREE.MeshStandardMaterial({
    color: 0xf5b800,
    roughness: 0.3,
  });

  // Main badge body
  const bodyGeo = new THREE.BoxGeometry(0.10, 0.08, 0.024);
  const body = new THREE.Mesh(bodyGeo, caseMat);
  group.add(body);

  // Digital status readout / HUD
  const screenGeo = new THREE.PlaneGeometry(0.075, 0.045);
  const screen = new THREE.Mesh(screenGeo, screenMat);
  screen.position.set(0, 0.005, 0.013);
  group.add(screen);

  // Safety clip / top mount
  const clipGeo = new THREE.CylinderGeometry(0.006, 0.006, 0.09, 12);
  clipGeo.rotateZ(Math.PI / 2);
  const clip = new THREE.Mesh(clipGeo, yellowMat);
  clip.position.set(0, 0.042, 0.008);
  group.add(clip);

  return group;
}

/**
 * Standard configuration for all supported PPE equipment slots with built-in 3D model presets.
 */
export const DEFAULT_SLOT_CONFIGS = {
  helmet: {
    id: "helmet",
    name: "Safety Helmet",
    category: "Head Protection",
    bone: "head",
    equipmentKey: "helmet",
    defaultFactory: () => createSafetyHelmet(),
    defaultTransform: {
      scale: 1.08,
      posX: 0,
      posY: 0.062,
      posZ: 0.010,
      rotX: 0,
      rotY: 0,
      rotZ: 0,
    },
    presets: [
      { id: "default", name: "Standard Industrial Hard Hat (Yellow)", factory: () => createSafetyHelmet({ color: 0xfac000, ridgeColor: 0xeb9800 }) },
      { id: "helmet-white", name: "Engineer Safety Hard Hat (White)", factory: () => createSafetyHelmet({ color: 0xf8f9fa, ridgeColor: 0xd8dde4 }) },
      { id: "helmet-red", name: "Fire Rescue Emergency Helmet (Safety Red)", factory: () => createSafetyHelmet({ color: 0xd32f2f, ridgeColor: 0xb71c1c }) },
      { id: "helmet-blue", name: "Site Supervisor Hard Hat (Electric Blue)", factory: () => createSafetyHelmet({ color: 0x1976d2, ridgeColor: 0x0d47a1 }) },
      { id: "helmet-lime", name: "High-Visibility Construction Helmet (Neon Lime)", factory: () => createSafetyHelmet({ color: 0x76ff03, ridgeColor: 0x64dd17 }) },
    ],
  },
  respirator: {
    id: "respirator",
    name: "Respirator / Mask",
    category: "Respiratory & Face",
    bone: "head",
    equipmentKey: "respirator",
    defaultFactory: () => createSCBAFaceMask(),
    defaultTransform: {
      scale: 0.87,
      posX: 0,
      posY: -0.018,
      posZ: 0.120,
      rotX: 0,
      rotY: 0,
      rotZ: 0,
    },
    presets: [
      { id: "default", name: "SCBA Full-Face Pressure Mask (Standard Blue Tint)", factory: () => createSCBAFaceMask({ visorColor: 0xa4dcff }) },
      { id: "mask-gold", name: "Flash & Heat Thermal Reflective Mask (Gold/Amber)", factory: () => createSCBAFaceMask({ visorColor: 0xffc107 }) },
      { id: "mask-stealth", name: "Tactical Response Mask (Smoked Blackout)", factory: () => createSCBAFaceMask({ visorColor: 0x212529, rimColor: 0x121416 }) },
    ],
  },
  torso: {
    id: "torso",
    name: "Torso / Vest",
    category: "Body Protection",
    bone: "chest",
    equipmentKey: "vest",
    defaultFactory: () => createSafetyVest(),
    defaultTransform: {
      scale: 1.20,
      posX: 0,
      posY: 0.150,
      posZ: 0.012,
      rotX: 0,
      rotY: 0,
      rotZ: 0,
    },
    presets: [
      { id: "default", name: "High-Vis Safety Harness Vest (Neon Lime)", factory: () => createSafetyVest({ color: 0xbcee00 }) },
      { id: "vest-orange", name: "Highway Construction Safety Vest (Safety Orange)", factory: () => createSafetyVest({ color: 0xff6d00 }) },
      { id: "vest-red", name: "Emergency Incident Commander Vest (Safety Red)", factory: () => createSafetyVest({ color: 0xd50000 }) },
      { id: "vest-tactical", name: "Industrial Heavy Fall-Arrest Harness (Black)", factory: () => createSafetyVest({ color: 0x212121 }) },
    ],
  },
  gloves: {
    id: "gloves",
    name: "Protective Gloves",
    category: "Hand Protection",
    bone: "hands",
    equipmentKey: "gloves",
    defaultFactory: (side = "left") => createProtectiveGloves(side),
    defaultTransform: {
      scale: 1.22,
      posX: 0,
      posY: 0,
      posZ: 0,
      rotX: 0,
      rotY: 0,
      rotZ: 0,
    },
    presets: [
      { id: "default", name: "Heavy Industrial Gauntlet Gloves (Safety Orange)", factory: (hand) => createProtectiveGloves(hand, { color: 0xf26522 }) },
      { id: "gloves-yellow", name: "Impact Mechanics Gloves (High-Vis Yellow)", factory: (hand) => createProtectiveGloves(hand, { color: 0xfac000 }) },
      { id: "gloves-red", name: "Thermal Fire-Resistant Gauntlets (Kevlar Red)", factory: (hand) => createProtectiveGloves(hand, { color: 0xd32f2f }) },
      { id: "gloves-blue", name: "Heavy Chemical Nitrile Gauntlets (Industrial Blue)", factory: (hand) => createProtectiveGloves(hand, { color: 0x1565c0 }) },
      { id: "gloves-tactical", name: "Reinforced Tactical Impact Gloves (Black)", factory: (hand) => createProtectiveGloves(hand, { color: 0x212121 }) },
    ],
  },
  back: {
    id: "back",
    name: "Back / Cylinder",
    category: "Back-Mounted Apparatus",
    bone: "chest",
    equipmentKey: "respirator",
    defaultFactory: () => createSCBACylinderApparatus(),
    defaultTransform: {
      scale: 1.02,
      posX: 0,
      posY: 0.085,
      posZ: -0.118,
      rotX: 0,
      rotY: 0,
      rotZ: 0,
    },
    presets: [
      { id: "default", name: "SCBA Twin Carbon Composite Cylinder (Safety Yellow)", factory: () => createSCBACylinderApparatus({ color: 0xf5b800 }) },
      { id: "back-red", name: "Emergency Oxygen Quick-Response Cylinder (Red)", factory: () => createSCBACylinderApparatus({ color: 0xd32f2f }) },
      { id: "back-grey", name: "300-Bar Extended Duration Composite Tanks (Carbon Grey)", factory: () => createSCBACylinderApparatus({ color: 0x607d8b }) },
    ],
  },
  front: {
    id: "front",
    name: "Front Equipment",
    category: "Chest & Front Attachment",
    bone: "chest",
    equipmentKey: "front",
    defaultFactory: () => createDefaultFrontAccessory(),
    defaultTransform: {
      scale: 1.00,
      posX: 0,
      posY: 0.110,
      posZ: 0.115,
      rotX: 0,
      rotY: 0,
      rotZ: 0,
    },
    presets: [
      { id: "default", name: "PASS Digital Safety Monitor Badge (Cyan HUD)", factory: () => createDefaultFrontAccessory({ screenColor: 0x00e5ff }) },
      { id: "front-amber", name: "4-Gas Atmospheric Multi-Sensor Detector (Amber Alert)", factory: () => createDefaultFrontAccessory({ screenColor: 0xffa000 }) },
      { id: "front-green", name: "Telemetry Life-Support System Monitor (Green Status)", factory: () => createDefaultFrontAccessory({ screenColor: 0x00e676 }) },
    ],
  },
};

/**
 * Manages custom user-uploaded 3D models and calibrations for each PPE slot.
 */
export class CustomModelManager {
  constructor() {
    this.slots = {};
    this.listeners = new Set();
    this.gltfLoader = new GLTFLoader();
    this.fbxLoader = new FBXLoader();
    this.objLoader = new OBJLoader();

    // Initialize all slots with defaults
    for (const [key, config] of Object.entries(DEFAULT_SLOT_CONFIGS)) {
      this.slots[key] = {
        ...config,
        selectedModelId: "default",
        customModels: [],
        isCustom: false,
        customModel: null,
        customFileName: null,
        transform: { ...config.defaultTransform },
      };
    }
  }

  onChange(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notify(event) {
    for (const cb of this.listeners) {
      try {
        cb(event);
      } catch (err) {
        console.error("CustomModelManager callback error:", err);
      }
    }
  }

  getSlot(slotId) {
    return this.slots[slotId] || null;
  }

  getAllSlots() {
    return Object.values(this.slots);
  }

  /**
   * Returns a list of all available 3D models for this slot (presets + custom uploads).
   */
  getAvailableModels(slotId) {
    const slot = this.slots[slotId];
    if (!slot) return [];

    const models = [];
    if (slot.presets) {
      slot.presets.forEach((p) => {
        models.push({
          id: p.id,
          name: p.name,
          isCustom: false,
          selected: slot.selectedModelId === p.id,
        });
      });
    }

    if (slot.customModels) {
      slot.customModels.forEach((cm) => {
        models.push({
          id: cm.id,
          name: cm.name,
          fileName: cm.fileName,
          isCustom: true,
          selected: slot.selectedModelId === cm.id,
        });
      });
    }

    return models;
  }

  /**
   * Selects an active 3D model for the given slot (either a preset id or a custom model id).
   */
  selectModel(slotId, modelId) {
    const slot = this.slots[slotId];
    if (!slot) return;

    slot.selectedModelId = modelId;
    const customMatch = slot.customModels ? slot.customModels.find((cm) => cm.id === modelId) : null;

    if (customMatch) {
      slot.isCustom = true;
      slot.customModel = customMatch.modelGroup;
      slot.customFileName = customMatch.fileName;
    } else {
      slot.isCustom = false;
      slot.customModel = null;
      slot.customFileName = null;
    }

    this.notify({
      type: "model-selected",
      slotId,
      modelId,
      isCustom: slot.isCustom,
      fileName: slot.customFileName,
      slot: { ...slot },
    });
  }

  /**
   * Loads a user-uploaded 3D model file (GLB, GLTF, FBX, OBJ) into a specific slot.
   */
  async loadModelFromFile(slotId, file) {
    const slot = this.slots[slotId];
    if (!slot) {
      throw new Error(`Invalid equipment slot: "${slotId}"`);
    }

    const filename = file.name.toLowerCase();
    const ext = filename.split(".").pop();

    let parsedGroup = null;

    if (ext === "glb" || ext === "gltf") {
      const buffer = await file.arrayBuffer();
      const gltf = await this.gltfLoader.parseAsync(buffer, "");
      parsedGroup = gltf.scene || gltf.scenes?.[0];
    } else if (ext === "fbx") {
      const buffer = await file.arrayBuffer();
      parsedGroup = this.fbxLoader.parse(buffer, "");
    } else if (ext === "obj") {
      const text = await file.text();
      parsedGroup = this.objLoader.parse(text);
    } else {
      throw new Error(`Unsupported 3D model format ".${ext}". Please upload GLB, GLTF, FBX, or OBJ.`);
    }

    if (!parsedGroup) {
      throw new Error("Unable to parse 3D geometry from uploaded file.");
    }

    // Prepare model: compute bounds, center around origin, normalize scale to humanoid range
    this.normalizeModel(parsedGroup, slotId);

    const customId = `custom-${Date.now()}`;
    const customEntry = {
      id: customId,
      name: `Custom: ${file.name}`,
      fileName: file.name,
      modelGroup: parsedGroup,
      date: new Date().toISOString(),
    };

    if (!slot.customModels) {
      slot.customModels = [];
    }

    // Replace if already uploaded with same filename, or add new
    const existingIdx = slot.customModels.findIndex((cm) => cm.fileName === file.name);
    if (existingIdx >= 0) {
      slot.customModels[existingIdx] = customEntry;
    } else {
      slot.customModels.unshift(customEntry);
    }

    // Select the newly uploaded model immediately
    this.selectModel(slotId, customId);

    return parsedGroup;
  }

  /**
   * Normalizes the size and centers an imported 3D model around its anchor.
   */
  normalizeModel(modelGroup, slotId = "helmet") {
    const SLOT_TARGET_SIZES = {
      helmet: 0.28,     // ~28 cm
      respirator: 0.24, // ~24 cm
      torso: 0.48,      // ~48 cm
      gloves: 0.26,     // ~26 cm
      back: 0.52,       // ~52 cm
      front: 0.14,      // ~14 cm
    };
    const targetSize = SLOT_TARGET_SIZES[slotId] || 0.28;

    // Traverse to ensure shadows, material visibility, and proper depth testing
    modelGroup.traverse((child) => {
      child.visible = true;
      if (child.isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
        child.renderOrder = 2;
        if (child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          mats.forEach((m) => {
            if (m) {
              m.side = THREE.DoubleSide;
              m.colorWrite = true;
              m.depthWrite = true;
              m.depthTest = true;
              m.needsUpdate = true;
            }
          });
        }
      }
    });

    // Compute initial bounding box
    const box = new THREE.Box3().setFromObject(modelGroup);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Convert standard units:
    if (maxDim > 50.0) {
      // Millimeters (e.g. 280 mm) -> meters
      modelGroup.scale.multiplyScalar(0.001);
    } else if (maxDim > 1.2) {
      // Centimeters (e.g. 28 cm) -> meters
      modelGroup.scale.multiplyScalar(0.01);
    }

    // Recompute box after unit scaling
    const box2 = new THREE.Box3().setFromObject(modelGroup);
    const size2 = box2.getSize(new THREE.Vector3());
    const maxDim2 = Math.max(size2.x, size2.y, size2.z);

    // If still significantly outside anatomical proportions, calibrate to target size
    if (maxDim2 > 1.1 || (maxDim2 < 0.08 && maxDim2 > 0)) {
      modelGroup.scale.multiplyScalar(targetSize / maxDim2);
    }

    // Recompute box after auto-scale and center geometry
    const updatedBox = new THREE.Box3().setFromObject(modelGroup);
    const center = updatedBox.getCenter(new THREE.Vector3());
    modelGroup.position.sub(center);
  }

  /**
   * Resets a slot back to the default built-in 3D model.
   */
  resetSlotToDefault(slotId) {
    const slot = this.slots[slotId];
    if (!slot) return;

    this.selectModel(slotId, "default");
    slot.transform = { ...slot.defaultTransform };

    this.notify({
      type: "slot-reset",
      slotId,
      slot: { ...slot },
    });
  }

  /**
   * Updates transformation parameters (scale, position, rotation) for a slot.
   */
  setSlotTransform(slotId, partialTransform = {}) {
    const slot = this.slots[slotId];
    if (!slot) return;

    slot.transform = {
      ...slot.transform,
      ...partialTransform,
    };

    this.notify({
      type: "transform-changed",
      slotId,
      transform: { ...slot.transform },
    });
  }

  /**
   * Instantiates and returns the 3D Object3D for a slot with transforms applied.
   */
  instantiateSlotMesh(slotId, options = {}) {
    const slot = this.slots[slotId];
    if (!slot) return null;

    const wrapper = new THREE.Group();
    wrapper.name = `Slot_${slotId}_Wrapper`;

    let contentMesh = null;

    if (slot.isCustom && slot.customModel) {
      contentMesh = slot.customModel.clone(true);
      // Mirror right hand glove if a single custom glove model is uploaded
      if (slotId === "gloves" && options.hand === "right") {
        contentMesh.scale.x *= -1;
      }
    } else {
      // Find preset by selectedModelId
      const preset = slot.presets?.find((p) => p.id === slot.selectedModelId) || slot.presets?.[0];
      if (preset && typeof preset.factory === "function") {
        contentMesh = preset.factory(options.hand);
      } else if (typeof slot.defaultFactory === "function") {
        contentMesh = slot.defaultFactory(options.hand);
      }
    }

    if (contentMesh) {
      wrapper.add(contentMesh);
    }

    this.applySlotTransformToGroup(wrapper, slot.transform, options);
    return wrapper;
  }

  /**
   * Applies slot transform values to a wrapper group.
   */
  applySlotTransformToGroup(group, transform, options = {}) {
    if (!group || !transform) return;

    const scale = typeof transform.scale === "number" ? transform.scale : 1.0;
    group.scale.set(scale, scale, scale);

    const px = transform.posX || 0;
    const py = transform.posY || 0;
    const pz = transform.posZ || 0;
    group.position.set(px, py, pz);

    const rx = transform.rotX || 0;
    const ry = transform.rotY || 0;
    const rz = transform.rotZ || 0;
    group.rotation.set(rx, ry, rz);
  }
}
