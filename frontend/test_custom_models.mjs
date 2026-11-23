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

  await t.test("All 6 slots model switching and hierarchy refresh", () => {
    const headBone = new THREE.Bone();
    headBone.name = "head";
    const chestBone = new THREE.Bone();
    chestBone.name = "spine03";
    const wristL = new THREE.Bone();
    wristL.name = "wristL";
    const wristR = new THREE.Bone();
    wristR.name = "wristR";

    const allSlots = ["helmet", "respirator", "torso", "gloves", "back", "front"];
    allSlots.forEach((slotId) => {
      const models = manager.getAvailableModels(slotId);
      assert.ok(models.length >= 2, `Slot "${slotId}" must have multiple presets available`);

      // Switch to secondary model preset
      const secondModel = models[1];
      manager.selectModel(slotId, secondModel.id);
      const slot = manager.getSlot(slotId);
      assert.equal(slot.selectedModelId, secondModel.id);

      const instantiated = manager.instantiateSlotMesh(slotId);
      assert.ok(instantiated instanceof THREE.Group, `Slot "${slotId}" must instantiate valid Group`);
      assert.ok(instantiated.name.startsWith(`Slot_${slotId}`), `Slot "${slotId}" name must follow Slot_${slotId} pattern`);
      assert.ok(instantiated.children.length > 0, `Slot "${slotId}" must have content children`);

      // Verify materials have colorWrite=true and depthWrite=true
      instantiated.traverse((c) => {
        if (c.isMesh && c.material) {
          const mats = Array.isArray(c.material) ? c.material : [c.material];
          mats.forEach((m) => {
            if (m) {
              assert.equal(m.colorWrite, true, `Mesh material in slot ${slotId} must have colorWrite=true`);
              assert.equal(m.depthWrite, true, `Mesh material in slot ${slotId} must have depthWrite=true`);
            }
          });
        }
      });
    });
  });

  await t.test("Replaced models preserve exact scale, depth, and position across all slots", () => {
    const slots = ["helmet", "respirator", "torso", "gloves", "back", "front"];

    slots.forEach((slotId) => {
      const slot = manager.getSlot(slotId);
      const defaultMesh = manager.instantiateSlotMesh(slotId);

      // Create an arbitrarily-sized custom model (e.g. 1500mm / 1.5m CAD model with arbitrary center)
      const customGroup = new THREE.Group();
      const geom = new THREE.BoxGeometry(1.5, 1.2, 0.9);
      geom.translate(0.5, -0.3, 0.2); // skewed pivot
      const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x00ff00 }));
      customGroup.add(mesh);

      // Normalize it for this slot
      manager.normalizeModel(customGroup, slotId);

      // Attach custom model to slot
      const customId = `custom-test-${slotId}`;
      slot.customModels = [
        {
          id: customId,
          name: `Custom ${slotId}`,
          fileName: `model_${slotId}.glb`,
          modelGroup: customGroup,
        },
      ];
      manager.selectModel(slotId, customId);

      const replacedMesh = manager.instantiateSlotMesh(slotId);

      // 1. Verify scale is preserved
      assert.equal(
        replacedMesh.scale.x,
        defaultMesh.scale.x,
        `Slot "${slotId}" replaced model must have identical scale.x (${defaultMesh.scale.x})`
      );
      assert.equal(
        replacedMesh.scale.y,
        defaultMesh.scale.y,
        `Slot "${slotId}" replaced model must have identical scale.y (${defaultMesh.scale.y})`
      );
      assert.equal(
        replacedMesh.scale.z,
        defaultMesh.scale.z,
        `Slot "${slotId}" replaced model must have identical scale.z (${defaultMesh.scale.z})`
      );

      // 2. Verify depth (posZ) and position (posX, posY) are preserved
      assert.equal(
        replacedMesh.position.x,
        defaultMesh.position.x,
        `Slot "${slotId}" replaced model must have identical posX (${defaultMesh.position.x})`
      );
      assert.equal(
        replacedMesh.position.y,
        defaultMesh.position.y,
        `Slot "${slotId}" replaced model must have identical posY (${defaultMesh.position.y})`
      );
      assert.equal(
        replacedMesh.position.z,
        defaultMesh.position.z,
        `Slot "${slotId}" replaced model must have identical posZ/depth (${defaultMesh.position.z})`
      );

      // 3. Verify movement synchronicity when attached to an animated bone
      const bone = new THREE.Bone();
      bone.name = `test_bone_${slotId}`;
      bone.add(replacedMesh);

      // Simulate bone translation & rotation during live tracking
      bone.position.set(0.12, 1.45, -0.35);
      bone.rotation.set(0.1, 0.45, -0.2);
      bone.updateMatrixWorld(true);

      const worldPos = new THREE.Vector3();
      replacedMesh.getWorldPosition(worldPos);
      assert.ok(
        Number.isFinite(worldPos.x) && Number.isFinite(worldPos.y) && Number.isFinite(worldPos.z),
        `Slot "${slotId}" replaced model must compute valid world transform when bone moves`
      );

      // Reset back to default for clean state
      manager.resetSlotToDefault(slotId);
    });
  });
});
