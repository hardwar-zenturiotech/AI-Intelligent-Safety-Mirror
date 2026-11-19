/**
 * EMOSH Client
 * Connects the frontend to the EMOSH-Tracker reconstruction service.
 * Handles job submission, status polling, GPU telemetry, and mesh sequence loading.
 */

export class EMOSHClient {
  constructor(options = {}) {
    this.serviceUrl = options.serviceUrl || "http://127.0.0.1:8000";
    this.status = "idle"; // "idle" | "connecting" | "processing" | "ready" | "error"
    this.gpuInfo = null;
    this.assetsInfo = null;
    this.latestMetadata = null;
    this.currentJobId = null;
    this.listeners = new Set();
  }

  onStatus(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  notify(event) {
    for (const cb of this.listeners) {
      try {
        cb(event);
      } catch (err) {
        console.error("EMOSHClient callback error:", err);
      }
    }
  }

  /**
   * Checks the health and GPU availability of the EMOSH service.
   */
  async checkHealth() {
    try {
      const resp = await fetch(`${this.serviceUrl}/health`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      if (!resp.ok) {
        throw new Error(`Service returned HTTP ${resp.status}`);
      }
      const data = await resp.json();
      this.status = "ready";
      this.gpuInfo = data.gpu;
      this.assetsInfo = data.assets;
      this.notify({ type: "health", ...data });
      return data;
    } catch (err) {
      this.status = "offline";
      this.notify({ type: "offline", error: err.message });
      return null;
    }
  }

  /**
   * Triggers an asynchronous EMOSH tracking run for a video file.
   */
  async startTracking({ videoPath, visualize = true, headOnly = false } = {}) {
    this.status = "processing";
    this.notify({ type: "start", message: "Submitting video to EMOSH-Tracker pipeline..." });

    try {
      const resp = await fetch(`${this.serviceUrl}/api/track`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: json.stringify({ video_path: videoPath, visualize, head_only: headOnly }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${resp.status}`);
      }

      const res = await resp.json();
      this.currentJobId = res.job_id;
      this.notify({ type: "job_started", jobId: res.job_id });
      return res.job_id;
    } catch (err) {
      this.status = "error";
      this.notify({ type: "error", error: err.message });
      throw err;
    }
  }

  /**
   * Polls an active job until complete or timeout.
   */
  async pollJob(jobId, onProgress = null, maxWaitMs = 120000) {
    const startTime = Date.now();
    while (Date.now() - startTime < maxWaitMs) {
      try {
        const resp = await fetch(`${this.serviceUrl}/api/jobs/${jobId}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const job = await resp.json();

        if (onProgress && typeof onProgress === "function") {
          onProgress(job);
        }

        if (job.status === "completed") {
          this.status = "ready";
          this.latestMetadata = job.metadata;
          this.notify({ type: "completed", job });
          return job;
        } else if (job.status === "error") {
          this.status = "error";
          this.notify({ type: "error", error: job.error });
          throw new Error(job.error || "EMOSH tracking failed");
        }
      } catch (err) {
        if (err.message.includes("failed")) throw err;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error("EMOSH tracking timed out after 2 minutes.");
  }

  /**
   * Loads reconstructed Three.js mesh sequence for a job ID.
   */
  async loadMeshSequence(jobId) {
    const resp = await fetch(`${this.serviceUrl}/api/jobs/${jobId}/mesh`);
    if (!resp.ok) {
      throw new Error(`Could not load mesh sequence for job ${jobId}`);
    }
    return await resp.json();
  }

  /**
   * Loads the pre-computed sample reconstruction.
   * Tries local backend /api/sample, falling back to static /models/emosh_sample.json.
   */
  async loadSampleMesh() {
    try {
      const resp = await fetch(`${this.serviceUrl}/api/sample`);
      if (resp.ok) {
        const data = await resp.json();
        this.status = "ready";
        return data;
      }
    } catch (e) {
      // Fall through to static asset
    }

    const fallbackResp = await fetch("/models/emosh_sample.json");
    if (!fallbackResp.ok) {
      throw new Error("Could not load EMOSH sample mesh from service or static bundle.");
    }
    return await fallbackResp.json();
  }
}
