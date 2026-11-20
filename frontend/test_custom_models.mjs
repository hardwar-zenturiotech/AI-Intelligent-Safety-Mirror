import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  CustomModelManager,
  DEFAULT_SLOT_CONFIGS,
  createDefaultFrontAccessory,
} from "./src/ppe/customModelManager.js";

test("CustomModelManager - Slot Configurations & Defaults", async (t) => {
  const manager = new CustomModelManager();

  await t.test("All 6 standard equipment slots are initialized", () => {
    const slots = manager.getAllSlots();
    assert.equal(slots.length, 6, "Expected exactly 6 equipment slots");

    const expectedKeys = ["helmet", "respirator", "torso", "gloves", "back", "front"];
    expectedKeys.forEach((key) => {
      const slot = manager.getSlot(key);
      assert.ok(slot, `Slot "${key}" should exist`);
      assert.equal(slot.isCustom, false, `Slot "${key}" should start with isCustom=false`);
      assert.equal(slot.customModel, null, `Slot "${key}" should start with null customModel`);
      assert.ok(slot.defaultTransform, `Slot "${key}" should have defaultTransform`);
    });
  });

  await t.test("Default built-in 3D model factories produce valid Object3D instances", () => {
    // 1. Helmet
    const helmetMesh = manager.instantiateSlotMesh("helmet");
    assert.ok(helmetMesh instanceof THREE.Group, "Helmet mesh must be a THREE.Group");
    assert.ok(helmetMesh.children.length > 0, "Helmet must have geometry children");
    assert.equal(helmetMesh.scale.x, DEFAULT_SLOT_CONFIGS.helmet.defaultTransform.scale);

    // 2. Respirator (Mask)
    const respiratorMesh = manager.instantiateSlotMesh("respirator");
    assert.ok(respiratorMesh instanceof THREE.Group, "Respirator mesh must be a THREE.Group");
    assert.ok(respiratorMesh.children.length > 0, "Respirator must have geometry children");
    assert.equal(respiratorMesh.scale.x, DEFAULT_SLOT_CONFIGS.respirator.defaultTransform.scale);

    // 3. Torso (Vest)
    const torsoMesh = manager.instantiateSlotMesh("torso");
    assert.ok(torsoMesh instanceof THREE.Group, "Torso mesh must be a THREE.Group");
    assert.ok(torsoMesh.children.length > 0, "Torso vest must have geometry children");
    assert.equal(torsoMesh.scale.x, DEFAULT_SLOT_CONFIGS.torso.defaultTransform.scale);

    // 4. Gloves (Left & Right)
    const gloveL = manager.instantiateSlotMesh("gloves", { hand: "left" });
    const gloveR = manager.instantiateSlotMesh("gloves", { hand: "right" });
    assert.ok(gloveL instanceof THREE.Group, "Left glove must be a THREE.Group");
    assert.ok(gloveR instanceof THREE.Group, "Right glove must be a THREE.Group");
    assert.ok(gloveL.children.length > 0, "Left glove must have children");
    assert.ok(gloveR.children.length > 0, "Right glove must have children");

    // 5. Back (Cylinder)
    const backMesh = manager.instantiateSlotMesh("back");
    assert.ok(backMesh instanceof THREE.Group, "Back cylinder must be a THREE.Group");
    assert.ok(backMesh.children.length > 0, "Cylinder must have children");
    assert.equal(backMesh.scale.x, DEFAULT_SLOT_CONFIGS.back.defaultTransform.scale);

    // 6. Front (Chest Accessory)
    const frontMesh = manager.instantiateSlotMesh("front");
    assert.ok(frontMesh instanceof THREE.Group, "Front accessory must be a THREE.Group");
    assert.ok(frontMesh.children.length > 0, "Front accessory must have children");
  });

  await t.test("Slot transformations can be modified and trigger listeners", () => {
    let changeEvent = null;
    const unsubscribe = manager.onChange((evt) => {
      changeEvent = evt;
    });

    manager.setSlotTransform("helmet", {
      scale: 1.5,
      posX: 0.02,
      posY: 0.08,
      posZ: -0.01,
      rotY: 0.25,
    });

    assert.ok(changeEvent, "onChange listener should fire");
    assert.equal(changeEvent.slotId, "helmet");
    assert.equal(changeEvent.type, "transform-changed");

    const updatedMesh = manager.instantiateSlotMesh("helmet");
    assert.equal(updatedMesh.scale.x, 1.5);
    assert.equal(updatedMesh.position.x, 0.02);
    assert.equal(updatedMesh.position.y, 0.08);
    assert.equal(updatedMesh.position.z, -0.01);
    assert.equal(updatedMesh.rotation.y, 0.25);

    unsubscribe();
  });

  await t.test("Custom 3D model simulation and reset to default", () => {
    // Simulate an uploaded custom 3D model group (e.g. parsed GLB)
    const customGroup = new THREE.Group();
    customGroup.name = "Custom_Mining_Helmet";
    const boxGeo = new THREE.BoxGeometry(0.3, 0.25, 0.35);
    const boxMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    customGroup.add(new THREE.Mesh(boxGeo, boxMat));

    const slot = manager.getSlot("helmet");
    manager.normalizeModel(customGroup);
    slot.isCustom = true;
    slot.customModel = customGroup;
    slot.customFileName = "mining_helmet.glb";

    const customMesh = manager.instantiateSlotMesh("helmet");
    assert.ok(customMesh instanceof THREE.Group);
    assert.ok(customMesh.children[0].name.includes("Custom_Mining_Helmet"), "Must instantiate the custom model");

    // Reset back to default
    manager.resetSlotToDefault("helmet");
    assert.equal(slot.isCustom, false, "Slot must revert isCustom to false");
    assert.equal(slot.customModel, null, "Custom model must be cleared");

    const restoredMesh = manager.instantiateSlotMesh("helmet");
    assert.ok(restoredMesh.children[0].name.includes("Safety_Helmet"), "Must restore original safety helmet");
  });

  await t.test("Available 3D models and preset selection options", () => {
    const available = manager.getAvailableModels("helmet");
    assert.ok(available.length >= 4, "Helmet should have multiple built-in presets");
    assert.ok(available.some((m) => m.id === "default"), "Should contain default helmet");
    assert.ok(available.some((m) => m.id === "helmet-red"), "Should contain red firefighter helmet preset");

    // Select red helmet preset
    manager.selectModel("helmet", "helmet-red");
    const slot = manager.getSlot("helmet");
    assert.equal(slot.selectedModelId, "helmet-red");
    assert.equal(slot.isCustom, false);

    const redMesh = manager.instantiateSlotMesh("helmet");
    assert.ok(redMesh instanceof THREE.Group);
    assert.ok(redMesh.children.length > 0);

    // Switch back to default
    manager.selectModel("helmet", "default");
    assert.equal(slot.selectedModelId, "default");
  });

  await t.test("Default Front Accessory geometry creates authentic chest unit", () => {
    const frontUnit = createDefaultFrontAccessory();
    assert.ok(frontUnit instanceof THREE.Group);
    assert.equal(frontUnit.name, "Front_Chest_Apparatus");
    assert.equal(frontUnit.children.length, 3, "Expected case, screen, and clip meshes");
  });
});
