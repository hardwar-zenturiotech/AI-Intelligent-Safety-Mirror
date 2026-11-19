import { createReadStream, createWriteStream, readFileSync, existsSync, copyFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { mkdir, rename, unlink } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const frontendDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = resolve(frontendDirectory, "..");
const recordingsDirectory = resolve(projectDirectory, "recordings");
const motionCaptureModels = [
  {
    fileName: "models/avatar.glb",
    source: resolve(projectDirectory, "models", "avatar.glb"),
    contentType: "model/gltf-binary",
  },
  {
    fileName: "models/pose_landmarker_lite.task",
    source: resolve(projectDirectory, "models", "pose_landmarker_lite.task"),
    contentType: "application/octet-stream",
  },
  {
    fileName: "models/rtmpose-m.onnx",
    source: resolve(projectDirectory, "models", "rtmpose-m.onnx"),
    contentType: "application/octet-stream",
  },
  {
    fileName: "models/rtmpose-s.onnx",
    source: resolve(projectDirectory, "models", "rtmpose-s.onnx"),
    contentType: "application/octet-stream",
  },
];
const maxRecordingBytes = 250 * 1024 * 1024;
const visionWasmDirectory = resolve(
  frontendDirectory,
  "node_modules",
  "@mediapipe",
  "tasks-vision",
  "wasm",
);
const visionWasmFiles = ["vision_wasm_internal.js", "vision_wasm_internal.wasm"];

export default defineConfig({
  root: projectDirectory,
  publicDir: resolve(frontendDirectory, "public"),
  plugins: [
    {
      name: "scba-mask-model",
      configureServer(server) {
        server.middlewares.use((request, response, next) => {
          if (request.url === "/") {
            response.writeHead(302, { Location: "/frontend/" });
            response.end();
            return;
          }

          const requestPath = request.url?.split("?")[0] ?? "";
          const motionCaptureModel = motionCaptureModels.find(
            (model) => requestPath === `/${model.fileName}`,
          );
          if (motionCaptureModel && request.method === "GET") {
            response.setHeader("Content-Type", motionCaptureModel.contentType);
            createReadStream(motionCaptureModel.source)
              .on("error", (error) => {
                if (!response.headersSent) {
                  response.writeHead(error.code === "ENOENT" ? 404 : 500);
                  response.end("Motion-capture model is unavailable.");
                }
              })
              .pipe(response);
            return;
          }

          if (requestPath === "/api/recordings") {
            if (request.method !== "POST") {
              response.writeHead(405, { "Content-Type": "application/json", Allow: "POST" });
              response.end(JSON.stringify({ error: "Only POST is supported." }));
              return;
            }

            const origin = request.headers.origin;
            if (origin) {
              try {
                if (new URL(origin).host !== request.headers.host) {
                  response.writeHead(403, { "Content-Type": "application/json" });
                  response.end(JSON.stringify({ error: "Cross-origin recording uploads are not allowed." }));
                  return;
                }
              } catch {
                response.writeHead(403, { "Content-Type": "application/json" });
                response.end(JSON.stringify({ error: "Invalid request origin." }));
                return;
              }
            }

            const contentType = request.headers["content-type"] || "";
            const isWebm = contentType.toLowerCase().includes("webm");
            const isMp4 = contentType.toLowerCase().includes("mp4");
            if (!isWebm && !isMp4) {
              response.writeHead(415, { "Content-Type": "application/json" });
              response.end(JSON.stringify({ error: "Only WebM and MP4 video recordings are supported." }));
              return;
            }

            const contentLength = Number(request.headers["content-length"]);
            if (Number.isFinite(contentLength) && contentLength > maxRecordingBytes) {
              response.writeHead(413, { "Content-Type": "application/json" });
              response.end(JSON.stringify({ error: "Recording exceeds the 250 MB limit." }));
              return;
            }

            const ext = isMp4 ? "mp4" : "webm";
            const filename =
              `safety_mirror_${new Date().toISOString().replace(/[:.]/g, "-")}_${randomUUID()}.${ext}`;
            const temporaryPath = resolve(recordingsDirectory, `.${filename}.part`);
            const recordingPath = resolve(recordingsDirectory, filename);
            let bytesWritten = 0;
            const sizeLimit = new Transform({
              transform(chunk, encoding, callback) {
                bytesWritten += chunk.length;
                if (bytesWritten > maxRecordingBytes) {
                  const error = new Error("Recording exceeds the 250 MB limit.");
                  error.code = "LIMIT_EXCEEDED";
                  callback(error);
                  return;
                }
                callback(null, chunk);
              },
            });

            void (async () => {
              try {
                await mkdir(recordingsDirectory, { recursive: true });
                await pipeline(
                  request,
                  sizeLimit,
                  createWriteStream(temporaryPath, { flags: "wx" }),
                );
                await rename(temporaryPath, recordingPath);
                response.writeHead(201, { "Content-Type": "application/json" });
                response.end(JSON.stringify({ filename, bytesWritten }));
              } catch (error) {
                try {
                  await unlink(temporaryPath);
                } catch (cleanupError) {
                  if (cleanupError.code !== "ENOENT") {
                    console.error("Could not remove incomplete recording.", cleanupError);
                  }
                }

                if (error.code === "LIMIT_EXCEEDED" || bytesWritten > maxRecordingBytes) {
                  response.writeHead(413, { "Content-Type": "application/json" });
                  response.end(JSON.stringify({ error: "Recording exceeds the 250 MB limit." }));
                  return;
                }
                console.error("Could not save browser recording.", error);
                response.writeHead(500, { "Content-Type": "application/json" });
                response.end(JSON.stringify({ error: `Could not save the recording: ${error.message}` }));
              }
            })();
            return;
          }

          const wasmPrefix = "/mediapipe/wasm/";
          const wasmFile = requestPath.startsWith(wasmPrefix)
            ? requestPath.slice(wasmPrefix.length)
            : "";
          if (visionWasmFiles.includes(wasmFile)) {
            response.setHeader(
              "Content-Type",
              wasmFile.endsWith(".wasm") ? "application/wasm" : "text/javascript",
            );
            createReadStream(resolve(visionWasmDirectory, wasmFile)).pipe(response);
            return;
          }
          next();
        });
      },
      generateBundle() {
        for (const model of motionCaptureModels) {
          if (existsSync(model.source)) {
            this.emitFile({
              type: "asset",
              fileName: model.fileName,
              source: readFileSync(model.source),
            });
          }
        }
        for (const file of visionWasmFiles) {
          this.emitFile({
            type: "asset",
            fileName: `mediapipe/wasm/${file}`,
            source: readFileSync(resolve(visionWasmDirectory, file)),
          });
        }
      },
      closeBundle() {
        const nestedIndex = resolve(frontendDirectory, "dist", "frontend", "index.html");
        const rootIndex = resolve(frontendDirectory, "dist", "index.html");
        if (existsSync(nestedIndex)) {
          copyFileSync(nestedIndex, rootIndex);
        }
      },
    },
  ],
  build: {
    outDir: resolve(frontendDirectory, "dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(frontendDirectory, "index.html"),
    },
  },
});
