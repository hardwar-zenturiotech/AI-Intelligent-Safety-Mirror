import { createServer } from "node:http";
import { createReadStream, createWriteStream, existsSync, statSync } from "node:fs";
import { mkdir, rename, unlink } from "node:fs/promises";
import { join, resolve, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 8088;
const HOST = process.env.HOST || "127.0.0.1";

const distDirCandidates = [
  resolve(__dirname, "resources"),
  resolve(__dirname, "app_data"),
  resolve(__dirname, "dist"),
  resolve(__dirname, "frontend", "dist"),
];
const distDir = distDirCandidates.find(d => existsSync(d)) || distDirCandidates[0];
const recordingsDir = resolve(__dirname, "recordings");
const maxRecordingBytes = 250 * 1024 * 1024;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".task": "application/octet-stream",
  ".onnx": "application/octet-stream",
  ".fbx": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".webm": "video/webm",
  ".mp4": "video/mp4",
  ".map": "application/json",
};

const server = createServer(async (req, res) => {
  // CORS & Security Headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Range");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const parsedUrl = new URL(req.url, `http://${req.headers.host || HOST}`);
  const pathname = decodeURIComponent(parsedUrl.pathname);

  // Health check
  if (pathname === "/api/server-health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", app: "AI Intelligent Safety Mirror", port: PORT }));
    return;
  }

  // Handle Recording Upload
  if (pathname === "/api/recordings" && req.method === "POST") {
    const contentType = req.headers["content-type"] || "";
    const isWebm = contentType.toLowerCase().includes("webm");
    const isMp4 = contentType.toLowerCase().includes("mp4");
    const ext = isMp4 ? "mp4" : "webm";

    const filename = `safety_mirror_${new Date().toISOString().replace(/[:.]/g, "-")}_${randomUUID()}.${ext}`;
    const tempPath = join(recordingsDir, `.${filename}.part`);
    const finalPath = join(recordingsDir, filename);

    let bytesWritten = 0;
    const sizeLimit = new Transform({
      transform(chunk, encoding, callback) {
        bytesWritten += chunk.length;
        if (bytesWritten > maxRecordingBytes) {
          const err = new Error("Recording exceeds 250 MB limit");
          err.code = "LIMIT_EXCEEDED";
          callback(err);
          return;
        }
        callback(null, chunk);
      },
    });

    try {
      await mkdir(recordingsDir, { recursive: true });
      await pipeline(req, sizeLimit, createWriteStream(tempPath, { flags: "wx" }));
      await rename(tempPath, finalPath);
      res.writeHead(201, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ filename, bytesWritten }));
    } catch (err) {
      try { await unlink(tempPath); } catch {}
      const status = err.code === "LIMIT_EXCEEDED" ? 413 : 500;
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message || "Failed to save recording" }));
    }
    return;
  }

  // Static File Serving from frontend/dist
  let relativePath = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  
  // Handle /frontend/ or root requests
  if (relativePath === "frontend" || relativePath === "frontend/") {
    relativePath = "index.html";
  } else if (relativePath.startsWith("frontend/")) {
    relativePath = relativePath.slice("frontend/".length);
    if (!relativePath) relativePath = "index.html";
  }

  let filePath = resolve(distDir, relativePath);

  // Security check: ensure filePath is inside distDir
  if (!filePath.startsWith(distDir)) {
    res.writeHead(403, { "Content-Type": "text/plain" });
    res.end("Forbidden");
    return;
  }

  // Fallback to index.html for SPA if not found, or direct file
  if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
    const alternative = resolve(distDir, "index.html");
    if (existsSync(alternative)) {
      filePath = alternative;
    } else {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("File not found. Please build the frontend first (npm run build).");
      return;
    }
  }

  try {
    const stat = statSync(filePath);
    const ext = extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";

    res.setHeader("Content-Type", contentType);
    res.setHeader("Content-Length", stat.size);

    if (ext === ".wasm" || ext === ".task" || ext === ".onnx" || ext === ".glb") {
      res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
    }

    createReadStream(filePath).pipe(res);
  } catch (err) {
    if (!res.headersSent) {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("Server error reading file: " + err.message);
    }
  }
});

server.listen(PORT, HOST, () => {
  console.log(`=======================================================`);
  console.log(`AI Intelligent Safety Mirror Production Server Running`);
  console.log(`URL: http://${HOST}:${PORT}`);
  console.log(`Serving: ${distDir}`);
  console.log(`=======================================================`);
});
