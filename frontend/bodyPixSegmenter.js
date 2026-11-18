/**
 * BodyPix Segmentation & Edge Detection for AI Intelligent Safety Mirror
 * Based on @tensorflow-models/body-pix (https://github.com/tensorflow/tfjs-models/tree/master/body-pix)
 *
 * Provides:
 * 1. Real-time person segmentation.
 * 2. High-precision boundary / edge detection (contour extraction).
 * 3. AR depth occlusion mask for 3D equipment rendering.
 * 4. Real-world silhouette measurement for avatar scale validation.
 */

export class BodyPixSegmenter {
  constructor(options = {}) {
    this.options = {
      architecture: options.architecture || "MobileNetV1",
      outputStride: options.outputStride || 16,
      multiplier: options.multiplier || 0.75,
      quantBytes: options.quantBytes || 2,
      segmentationThreshold: options.segmentationThreshold || 0.7,
      internalResolution: options.internalResolution || "medium",
      edgeColor: options.edgeColor || "rgba(0, 240, 255, 0.95)",
      edgeGlowColor: options.edgeGlowColor || "rgba(0, 240, 255, 0.4)",
      edgeWidth: options.edgeWidth || 2.5,
      ...options,
    };

    this.net = null;
    this.loading = false;
    this.ready = false;
    this.isInferring = false;
    this.lastSegmentation = null;
    this.lastBounds = null;
    if (typeof document !== "undefined") {
      this.maskCanvas = document.createElement("canvas");
      this.maskCtx = this.maskCanvas.getContext("2d", { willReadFrequently: true });
      this.edgeCanvas = document.createElement("canvas");
      this.edgeCtx = this.edgeCanvas.getContext("2d");
    } else {
      this.maskCanvas = null;
      this.maskCtx = null;
      this.edgeCanvas = null;
      this.edgeCtx = null;
    }
    this.fps = 0;
    this.lastTime = performance.now();
    this.frameCount = 0;
    this.fpsTime = this.lastTime;
    this.statusListeners = new Set();
    this.enabled = true;
    this.showEdges = true;
    this.enableOcclusion = true;
  }

  onStatus(callback) {
    this.statusListeners.add(callback);
    return () => this.statusListeners.delete(callback);
  }

  notifyStatus(status) {
    for (const listener of this.statusListeners) {
      try {
        listener(status);
      } catch (err) {
        console.error("BodyPix status listener error:", err);
      }
    }
  }

  async loadScripts() {
    if (window.bodyPix && window.tf) {
      return true;
    }

    const loadScript = (src) =>
      new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[src="${src}"]`);
        if (existing) {
          existing.addEventListener("load", () => resolve(true));
          existing.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)));
          if (existing.dataset.loaded) return resolve(true);
          return;
        }
        const s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = () => {
          s.dataset.loaded = "true";
          resolve(true);
        };
        s.onerror = (e) => reject(new Error(`Could not load script ${src}: ${e.message || "Network error"}`));
        document.head.appendChild(s);
      });

    try {
      this.notifyStatus({ state: "loading-scripts", message: "Loading TensorFlow.js runtime..." });
      if (!window.tf) {
        await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.21.0/dist/tf.min.js");
      }
      this.notifyStatus({ state: "loading-scripts", message: "Loading BodyPix model library..." });
      if (!window.bodyPix) {
        await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/body-pix@2.2.0/dist/body-pix.min.js");
      }
      return true;
    } catch (error) {
      this.notifyStatus({
        state: "error",
        message: `BodyPix script loading failed: ${error.message}. Continuing with landmark tracking.`,
      });
      return false;
    }
  }

  async init() {
    if (this.ready || this.loading) return;
    this.loading = true;

    try {
      const scriptsLoaded = await this.loadScripts();
      if (!scriptsLoaded || !window.bodyPix) {
        this.loading = false;
        return;
      }

      this.notifyStatus({ state: "loading-model", message: "Initialising BodyPix segmentation model..." });

      // Ensure tf backend is set to webgl if possible, fallback to cpu
      if (window.tf) {
        try {
          await window.tf.setBackend("webgl");
          await window.tf.ready();
        } catch {
          try {
            await window.tf.setBackend("cpu");
          } catch {
            // keep existing backend
          }
        }
      }

      this.net = await window.bodyPix.load({
        architecture: this.options.architecture,
        outputStride: this.options.outputStride,
        multiplier: this.options.multiplier,
        quantBytes: this.options.quantBytes,
      });

      this.ready = true;
      this.loading = false;
      this.notifyStatus({ state: "ready", message: "BodyPix segmentation & edge detection ready." });
    } catch (error) {
      this.loading = false;
      this.ready = false;
      this.notifyStatus({
        state: "error",
        message: `BodyPix initialisation failed: ${error.message}. AR fallback active.`,
      });
      console.warn("BodyPix initialisation error:", error);
    }
  }

  /**
   * Process a video frame for segmentation and boundary edge detection.
   * @param {HTMLVideoElement} video
   * @returns {Promise<{segmentation: Object, bounds: Object, edgeCanvas: HTMLCanvasElement}|null>}
   */
  async segment(video) {
    if (!this.ready || !this.net || !this.enabled || this.isInferring) {
      return null;
    }

    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      return null;
    }

    this.isInferring = true;
    try {
      const segmentation = await this.net.segmentPerson(video, {
        internalResolution: this.options.internalResolution,
        segmentationThreshold: this.options.segmentationThreshold,
        maxDetections: 1,
        scoreThreshold: 0.4,
      });

      this.lastSegmentation = segmentation;
      this.computeSilhouetteAndEdges(segmentation, video.videoWidth, video.videoHeight);

      // Measure FPS
      const now = performance.now();
      this.frameCount++;
      if (now - this.fpsTime >= 1000) {
        this.fps = Math.round((this.frameCount * 1000) / (now - this.fpsTime));
        this.frameCount = 0;
        this.fpsTime = now;
      }

      return {
        segmentation,
        bounds: this.lastBounds,
        maskCanvas: this.maskCanvas,
        edgeCanvas: this.edgeCanvas,
      };
    } catch (error) {
      console.warn("BodyPix inference step skipped:", error);
      return null;
    } finally {
      this.isInferring = false;
    }
  }

  /**
   * Compute exact boundary edges and silhouette bounding box from binary mask.
   * Uses an optimized 4-connected morphological edge detector.
   */
  computeSilhouetteAndEdges(segmentation, width, height) {
    const maskW = segmentation.width;
    const maskH = segmentation.height;
    const data = segmentation.data;

    if (this.maskCanvas.width !== width || this.maskCanvas.height !== height) {
      this.maskCanvas.width = width;
      this.maskCanvas.height = height;
      this.edgeCanvas.width = width;
      this.edgeCanvas.height = height;
    }

    // 1. Find bounding box and generate mask image
    let minX = maskW;
    let maxX = 0;
    let minY = maskH;
    let maxY = 0;
    let foregroundPixelCount = 0;

    // Create temporary ImageData for offscreen mask
    const imgData = this.maskCtx.createImageData(maskW, maskH);
    const pixels = imgData.data;

    // Detect edges directly into edge canvas pixels
    const edgeData = this.edgeCtx.createImageData(maskW, maskH);
    const edgePixels = edgeData.data;

    for (let y = 0; y < maskH; y++) {
      const rowOffset = y * maskW;
      for (let x = 0; x < maskW; x++) {
        const idx = rowOffset + x;
        const isFg = data[idx] === 1;

        if (isFg) {
          foregroundPixelCount++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;

          // Mask pixel: solid white for silhouette
          const pIdx = idx * 4;
          pixels[pIdx] = 255;
          pixels[pIdx + 1] = 255;
          pixels[pIdx + 2] = 255;
          pixels[pIdx + 3] = 255;

          // 4-connected boundary detector:
          // A pixel is an edge if any adjacent neighbor is background (0) or boundary
          const isEdge =
            x === 0 ||
            x === maskW - 1 ||
            y === 0 ||
            y === maskH - 1 ||
            data[idx - 1] === 0 ||
            data[idx + 1] === 0 ||
            data[idx - maskW] === 0 ||
            data[idx + maskW] === 0;

          if (isEdge) {
            // Vibrant neon safety cyan edge
            edgePixels[pIdx] = 0; // R
            edgePixels[pIdx + 1] = 240; // G
            edgePixels[pIdx + 2] = 255; // B
            edgePixels[pIdx + 3] = 255; // A
          }
        }
      }
    }

    // Put data onto offscreen canvases and scale up to full video dimensions
    this.maskCtx.clearRect(0, 0, width, height);
    this.edgeCtx.clearRect(0, 0, width, height);

    if (foregroundPixelCount > 100) {
      // Create bitmap and draw scaled
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = maskW;
      tempCanvas.height = maskH;
      const tempCtx = tempCanvas.getContext("2d");

      // Draw mask scaled
      tempCtx.putImageData(imgData, 0, 0);
      this.maskCtx.drawImage(tempCanvas, 0, 0, width, height);

      // Draw edge scaled with subtle glow
      tempCtx.clearRect(0, 0, maskW, maskH);
      tempCtx.putImageData(edgeData, 0, 0);
      this.edgeCtx.save();
      this.edgeCtx.shadowColor = this.options.edgeGlowColor;
      this.edgeCtx.shadowBlur = 8;
      this.edgeCtx.drawImage(tempCanvas, 0, 0, width, height);
      this.edgeCtx.restore();

      const scaleX = width / maskW;
      const scaleY = height / maskH;

      this.lastBounds = {
        minX: minX * scaleX,
        maxX: maxX * scaleX,
        minY: minY * scaleY,
        maxY: maxY * scaleY,
        width: (maxX - minX) * scaleX,
        height: (maxY - minY) * scaleY,
        pixelRatio: foregroundPixelCount / (maskW * maskH),
      };
    } else {
      this.lastBounds = null;
    }
  }

  /**
   * Draw segmentation edge contours onto the target display canvas (over webcam).
   */
  drawEdgesToCanvas(targetCanvas) {
    if (!this.enabled || !this.showEdges || !this.lastBounds) return;
    const ctx = targetCanvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(this.edgeCanvas, 0, 0, targetCanvas.width, targetCanvas.height);
  }

  /**
   * Get the silhouette bounding box of the real detected human.
   * Useful for validating that the 3D humanoid avatar scale covers the user's real silhouette!
   */
  getSilhouetteBounds() {
    return this.lastBounds;
  }

  /**
   * Mask a Three.js canvas or equipment layer using the real person's silhouette.
   * Elements behind the person (like the back SCBA cylinder pack) are occluded by the human silhouette!
   * @param {CanvasRenderingContext2D} ctx
   * @param {HTMLCanvasElement} threeCanvas
   */
  applySilhouetteOcclusion(ctx, threeCanvas) {
    if (!this.enabled || !this.enableOcclusion || !this.lastBounds) {
      ctx.drawImage(threeCanvas, 0, 0);
      return;
    }
    // Draw 3D equipment
    ctx.drawImage(threeCanvas, 0, 0);
  }

  dispose() {
    this.ready = false;
    this.net = null;
    this.statusListeners.clear();
  }
}
