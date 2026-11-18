import { BodyPixSegmenter } from "./bodyPixSegmenter.js";

let allPassed = true;
function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    allPassed = false;
  } else {
    console.log(`PASS: ${msg}`);
  }
}

console.log("=== BodyPix Segmentation & Edge Detection Suite ===");

// 1. Initialisation test
const segmenter = new BodyPixSegmenter({
  architecture: "MobileNetV1",
  outputStride: 16,
  multiplier: 0.75,
});

assert(segmenter !== null, "BodyPixSegmenter instantiated successfully.");
assert(segmenter.enabled === true, "BodyPixSegmenter defaults to enabled.");
assert(segmenter.showEdges === true, "Edge detection defaults to enabled.");

// 2. Synthetic mask boundary & edge detection test
// Create a 64x64 synthetic person segmentation mask
const maskW = 64;
const maskH = 64;
const maskData = new Uint8Array(maskW * maskH);

// Draw a filled rectangular silhouette representing a torso from (x: 20..44, y: 15..50)
for (let y = 15; y <= 50; y++) {
  for (let x = 20; x <= 44; x++) {
    maskData[y * maskW + x] = 1;
  }
}

const syntheticSegmentation = {
  width: maskW,
  height: maskH,
  data: maskData,
};

// Polyfill canvas for headless Node environment if document is not defined
if (typeof document === "undefined") {
  globalThis.document = {
    createElement: (tag) => ({
      width: 0,
      height: 0,
      getContext: () => ({
        createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
        putImageData: () => {},
        drawImage: () => {},
        clearRect: () => {},
        save: () => {},
        restore: () => {},
      }),
    }),
  };
}

// Mock canvases on segmenter
segmenter.maskCanvas = document.createElement("canvas");
segmenter.maskCtx = segmenter.maskCanvas.getContext("2d");
segmenter.edgeCanvas = document.createElement("canvas");
segmenter.edgeCtx = segmenter.edgeCanvas.getContext("2d");

segmenter.computeSilhouetteAndEdges(syntheticSegmentation, 640, 480);
const bounds = segmenter.getSilhouetteBounds();

assert(bounds !== null, "Silhouette bounds computed from segmentation mask.");
assert(Math.round(bounds.minX) === 200, `Bounds minX matches silhouette (expected 200, got ${Math.round(bounds.minX)})`);
assert(Math.round(bounds.maxX) === 440, `Bounds maxX matches silhouette (expected 440, got ${Math.round(bounds.maxX)})`);
assert(Math.round(bounds.width) === 240, `Bounds width matches silhouette (expected 240, got ${Math.round(bounds.width)})`);
assert(bounds.height > 200, `Bounds height is positive (${bounds.height.toFixed(1)})`);

// 3. Test empty mask rejection
const emptySegmentation = {
  width: maskW,
  height: maskH,
  data: new Uint8Array(maskW * maskH),
};
segmenter.computeSilhouetteAndEdges(emptySegmentation, 640, 480);
assert(segmenter.getSilhouetteBounds() === null, "Empty mask correctly yields null silhouette bounds.");

if (allPassed) {
  console.log("ALL BODYPIX SEGMENTATION & EDGE DETECTION TESTS PASSED!");
  process.exit(0);
} else {
  process.exit(1);
}
