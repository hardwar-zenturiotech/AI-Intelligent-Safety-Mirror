import {
  BODYPIX_PARTS,
  PART_NAMES,
  LIMB_GROUPS,
  DEFAULT_PART_COLORS,
  BodyPartSegmentationEngine,
} from "./src/bodyPartSegmentation.js";

let passed = true;
function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    passed = false;
  } else {
    console.log(`PASS: ${msg}`);
  }
}

console.log("=== BodyPix Body-Part Segmentation Test Suite ===");

// 1. Verify Constants
assert(Object.keys(BODYPIX_PARTS).length === 25, "All 24 body parts + background defined.");
assert(PART_NAMES.length === 24, "24 official part names mapped in order.");
assert(PART_NAMES[0] === "left_face" && PART_NAMES[1] === "right_face", "Head parts mapped at 0 and 1.");
assert(PART_NAMES[10] === "left_hand" && PART_NAMES[11] === "right_hand", "Hand parts mapped at 10 and 11.");
assert(PART_NAMES[12] === "torso_front" && PART_NAMES[13] === "torso_back", "Torso parts mapped at 12 and 13.");
assert(PART_NAMES[22] === "left_foot" && PART_NAMES[23] === "right_foot", "Foot parts mapped at 22 and 23.");

// 2. Verify Limb Groups
assert(Array.isArray(LIMB_GROUPS.head) && LIMB_GROUPS.head.length === 2, "Head limb group mapped.");
assert(Array.isArray(LIMB_GROUPS.torso) && LIMB_GROUPS.torso.length === 2, "Torso limb group mapped.");
assert(Array.isArray(LIMB_GROUPS.leftArm) && LIMB_GROUPS.leftArm.length === 5, "Left arm limb group mapped (upper front/back, lower front/back, hand).");
assert(Array.isArray(LIMB_GROUPS.rightArm) && LIMB_GROUPS.rightArm.length === 5, "Right arm limb group mapped (upper front/back, lower front/back, hand).");

// 3. Verify Default Colors
assert(Object.keys(DEFAULT_PART_COLORS).length === 24, "All 24 body parts have distinct assigned colors.");
for (let i = 0; i < 24; i++) {
  const c = DEFAULT_PART_COLORS[i];
  assert(
    c &&
      typeof c.r === "number" &&
      typeof c.g === "number" &&
      typeof c.b === "number" &&
      c.r >= 0 && c.r <= 255,
    `Part ${i} (${PART_NAMES[i]}) has valid RGB color`,
  );
}

// 4. Test Engine Instantiation and Defaults
const engine = new BodyPartSegmentationEngine({
  architecture: "MobileNetV1",
  outputStride: 16,
  multiplier: 0.75,
  quantBytes: 2,
  targetFps: 12,
  debugMode: "overlay",
});
assert(engine.options.architecture === "MobileNetV1", "Architecture option preserved.");
assert(engine.options.outputStride === 16, "OutputStride preserved.");
assert(engine.options.targetFps === 12, "TargetFps preserved.");
assert(engine.options.debugMode === "overlay", "Default debugMode is overlay.");

// 5. Test Mask Processing & Querying with Synthetic Data
const width = 10;
const height = 10;
const synthData = new Uint8ClampedArray(width * height * 4);

// Simulate background everywhere first
for (let i = 0; i < width * height; i++) {
  synthData[i * 4 + 0] = 0;
  synthData[i * 4 + 1] = 0;
  synthData[i * 4 + 2] = 0;
  synthData[i * 4 + 3] = 0; // alpha = 0 (background)
}

// Place Head (part 0) at pixel (2, 2)
const headIdx = (2 * width + 2) * 4;
synthData[headIdx + 0] = BODYPIX_PARTS.LEFT_FACE;
synthData[headIdx + 3] = 255;

// Place Left Hand (part 10) at pixel (5, 5)
const handIdx = (5 * width + 5) * 4;
synthData[handIdx + 0] = BODYPIX_PARTS.LEFT_HAND;
synthData[handIdx + 3] = 255;

// Place Torso Front (part 12) at pixel (7, 7)
const torsoIdx = (7 * width + 7) * 4;
synthData[torsoIdx + 0] = BODYPIX_PARTS.TORSO_FRONT;
synthData[torsoIdx + 3] = 255;

engine.latestMaskData = { width, height, data: synthData };

// Test getPartMask
const headMask = engine.getPartMask("left_face");
assert(headMask && headMask[2 * width + 2] === 1, "getPartMask('left_face') identifies head pixel.");
assert(headMask[0] === 0 && headMask[5 * width + 5] === 0, "getPartMask('left_face') rejects non-head pixels.");

const handMask = engine.getPartMask(BODYPIX_PARTS.LEFT_HAND);
assert(handMask && handMask[5 * width + 5] === 1, "getPartMask(10) identifies left hand pixel.");

// Test getLimbMask
const leftArmMask = engine.getLimbMask("leftArm");
assert(leftArmMask && leftArmMask[5 * width + 5] === 1, "getLimbMask('leftArm') includes left hand.");
assert(leftArmMask[2 * width + 2] === 0, "getLimbMask('leftArm') excludes head.");

const torsoLimbMask = engine.getLimbMask("torso");
assert(torsoLimbMask && torsoLimbMask[7 * width + 7] === 1, "getLimbMask('torso') includes torso front.");

// Test getPartAtNormalizedCoord
const headQuery = engine.getPartAtNormalizedCoord(0.25, 0.25);
assert(headQuery.isPerson && headQuery.id === 0 && headQuery.name === "left_face", "getPartAtNormalizedCoord identifies head at (0.25, 0.25).");

const bgQuery = engine.getPartAtNormalizedCoord(0.0, 0.0);
assert(!bgQuery.isPerson && bgQuery.id === -1 && bgQuery.name === "background", "getPartAtNormalizedCoord identifies background at (0, 0).");

// 6. Test Settings & Disposal
engine.setDebugMode("parts");
assert(engine.options.debugMode === "parts", "setDebugMode works.");

engine.setSelectedPartGroup("leftArm");
assert(engine.options.selectedPartGroup === "leftArm", "setSelectedPartGroup works.");

engine.setTargetFps(20);
assert(engine.options.targetFps === 20, "setTargetFps works.");

engine.setInternalResolution("low");
assert(engine.options.internalResolution === "low", "setInternalResolution('low') updates resolution for GPU acceleration.");

// 7. Test 32-bit Fast Color LUT Generation
const lut32 = engine.getOrCreateLUT32("overlay", "all", 0.65);
assert(lut32 instanceof Uint32Array && lut32.length === 24, "Fast 32-bit LUT generated with 24 precomputed colors.");
assert(lut32[0] !== 0, "Head color in 32-bit LUT is non-zero.");

// 8. Test Silhouette Bounds Computation from Segmentation
engine.latestRawPartData = synthData;
engine.outputWidth = width;
engine.outputHeight = height;
const bounds = engine.computeBounds();
assert(bounds !== null, "computeBounds returned valid bounding box.");
assert(bounds.minX === 2 && bounds.maxX === 7, "Bounds minX/maxX accurately bounded detected pixels.");
assert(bounds.minY === 2 && bounds.maxY === 7, "Bounds minY/maxY accurately bounded detected pixels.");

// 9. Test Body Orientation Metrics & Front vs. Rear View Detection
// Front View Simulation (Torso Front dominant)
const frontSynth = new Int32Array(width * height);
frontSynth.fill(-1);
// Populate 40 pixels of torso_front (12)
for (let y = 4; y <= 7; y++) {
  for (let x = 3; x <= 6; x++) {
    frontSynth[y * width + x] = BODYPIX_PARTS.TORSO_FRONT;
  }
}
// Add 4 head pixels
frontSynth[1 * width + 4] = BODYPIX_PARTS.LEFT_FACE;
frontSynth[1 * width + 5] = BODYPIX_PARTS.RIGHT_FACE;

engine.latestRawPartData = frontSynth;
engine.smoothedFacingBias = 1.0;
const frontMetrics = engine.computeBodyPartMetrics();
assert(frontMetrics !== null, "computeBodyPartMetrics returned valid metrics for front view.");
assert(frontMetrics.torsoFrontPixels === 16, "Torso front pixel count accurately tallied (16).");
assert(frontMetrics.torsoBackPixels === 0, "Torso back pixels zero in front view.");
assert(frontMetrics.facingBias > 0.5, `Front facing bias detected (${frontMetrics.facingBias.toFixed(2)} > 0.5).`);
assert(frontMetrics.viewMode === "front", "View mode classified as 'front'.");
assert(frontMetrics.isFrontView === true, "isFrontView is true.");
assert(frontMetrics.isRearView === false, "isRearView is false.");

// Rear View Simulation (Torso Back dominant for back-mounted SCBA cylinder inspection)
const rearSynth = new Int32Array(width * height);
rearSynth.fill(-1);
// Populate 40 pixels of torso_back (13)
for (let y = 4; y <= 7; y++) {
  for (let x = 3; x <= 6; x++) {
    rearSynth[y * width + x] = BODYPIX_PARTS.TORSO_BACK;
  }
}
engine.latestRawPartData = rearSynth;
engine.smoothedFacingBias = -1.0;
const rearMetrics = engine.computeBodyPartMetrics();
assert(rearMetrics !== null, "computeBodyPartMetrics returned valid metrics for rear view.");
assert(rearMetrics.torsoBackPixels === 16, "Torso back pixel count accurately tallied (16).");
assert(rearMetrics.facingBias < -0.5, `Rear facing bias detected (${rearMetrics.facingBias.toFixed(2)} < -0.5).`);
assert(rearMetrics.viewMode === "back", "View mode classified as 'back'.");
assert(rearMetrics.isRearView === true, "isRearView is true for SCBA cylinder awareness.");
assert(rearMetrics.isFrontView === false, "isFrontView is false.");

// 10. Test UnifiedPoseTracker with BodyPix Engine Selection
const { UnifiedPoseTracker } = await import("./rtmposeTracker.js");
const unified = new UnifiedPoseTracker({ preferredEngine: "bodypix", bodyPartEngine: engine });
assert(unified.preferredEngine === "bodypix", "UnifiedPoseTracker accepts 'bodypix' as preferred engine.");
await unified.setEngine("bodypix");
assert(unified.activeEngine === "bodypix", "UnifiedPoseTracker successfully switches to 'bodypix'.");

engine.close();
assert(!engine.enabled && !engine.ready, "Engine closes and marks ready/enabled false.");

if (passed) {
  console.log("ALL BODYPIX BODY-PART SEGMENTATION TESTS PASSED!");
  process.exit(0);
} else {
  process.exit(1);
}
