"""
EMOSH-Tracker HTTP Microservice & Reconstruction API
Provides REST endpoints for asynchronous video tracking, GPU telemetry,
and Three.js mesh sequence streaming.
"""

import os
import sys
import json
import uuid
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from gpu_telemetry import get_gpu_info
from model_validator import validate_assets
from run_tracking import run_pipeline
from emosh_adapter import EMOSHOutputAdapter

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory jobs registry
JOBS = {}
JOBS_LOCK = threading.Lock()

def process_tracking_job(job_id, video_path, visualize, head_only, device):
    job_output_dir = OUTPUTS_DIR / job_id
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["start_time"] = time.time()

    try:
        meta = run_pipeline(
            input_path=video_path,
            output_dir=job_output_dir,
            visualize=visualize,
            head_only=head_only,
            device=device,
        )
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["metadata"] = meta
            JOBS[job_id]["output_dir"] = str(job_output_dir)
            JOBS[job_id]["end_time"] = time.time()
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["end_time"] = time.time()

class EMOSHRequestHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # 1. Health / System Status
        if path in ["/", "/health", "/api/status"]:
            gpu = get_gpu_info()
            assets = validate_assets()
            with JOBS_LOCK:
                active_count = sum(1 for j in JOBS.values() if j.get("status") == "processing")
                completed_count = sum(1 for j in JOBS.values() if j.get("status") == "completed")
            self.send_json(200, {
                "status": "online",
                "engine": "EMOSH-Tracker",
                "version": "1.0",
                "gpu": gpu,
                "assets": assets,
                "jobs": {
                    "active": active_count,
                    "completed": completed_count,
                    "total": len(JOBS),
                },
            })
            return

        # 2. List Jobs
        if path == "/api/jobs":
            with JOBS_LOCK:
                job_list = [
                    {
                        "job_id": jid,
                        "status": info.get("status"),
                        "input": info.get("input_file"),
                        "created_at": info.get("created_at"),
                        "duration": round(info.get("end_time", time.time()) - info.get("start_time", time.time()), 2)
                        if "start_time" in info else 0,
                    }
                    for jid, info in JOBS.items()
                ]
            self.send_json(200, {"jobs": job_list})
            return

        # 3. Get Specific Job Status
        if path.startswith("/api/jobs/") and not path.endswith("/mesh") and not path.endswith("/metadata"):
            job_id = path.split("/")[3]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self.send_json(404, {"error": f"Job {job_id} not found"})
                return
            self.send_json(200, job)
            return

        # 4. Get Job Metadata
        if path.startswith("/api/jobs/") and path.endswith("/metadata"):
            job_id = path.split("/")[3]
            meta_file = OUTPUTS_DIR / job_id / "metadata.json"
            if not meta_file.exists():
                self.send_json(404, {"error": "Metadata not found"})
                return
            with open(meta_file, "r", encoding="utf-8") as f:
                self.send_json(200, json.load(f))
            return

        # 5. Stream Reconstructed Mesh Sequence for Three.js
        if path.startswith("/api/jobs/") and path.endswith("/mesh"):
            job_id = path.split("/")[3]
            mesh_file = OUTPUTS_DIR / job_id / "mesh_sequence.json"
            if not mesh_file.exists():
                self.send_json(404, {"error": "Mesh sequence not found"})
                return
            with open(mesh_file, "r", encoding="utf-8") as f:
                self.send_json(200, json.load(f))
            return

        # 6. Sample Mesh Endpoint (instant demonstration from pre-computed sample run)
        if path in ["/api/sample", "/api/sample/mesh"]:
            sample_file = OUTPUTS_DIR / "sample_test" / "mesh_sequence.json"
            if sample_file.exists():
                with open(sample_file, "r", encoding="utf-8") as f:
                    self.send_json(200, json.load(f))
                return
            else:
                self.send_json(404, {"error": "No pre-computed sample found. Run a track job first."})
                return

        self.send_json(404, {"error": f"Endpoint {path} not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/track":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)

            try:
                data = json.loads(post_body.decode("utf-8")) if post_body else {}
            except Exception:
                data = {}

            video_path = data.get("video_path")
            visualize = data.get("visualize", True)
            head_only = data.get("head_only", False)
            device = data.get("device", "cuda")

            # Default to newest recorded clip in recordings/ if not specified
            if not video_path:
                recordings_dir = BASE_DIR.parent / "recordings"
                if recordings_dir.exists():
                    vids = list(recordings_dir.glob("*.mp4")) + list(recordings_dir.glob("*.webm"))
                    if vids:
                        vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        video_path = str(vids[0])

            if not video_path or not Path(video_path).exists():
                self.send_json(400, {"error": f"Invalid or missing video_path: {video_path}"})
                return

            job_id = f"emosh_{uuid.uuid4().hex[:8]}"
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "progress": 0,
                    "input_file": str(video_path),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

            thread = threading.Thread(
                target=process_tracking_job,
                args=(job_id, video_path, visualize, head_only, device),
                daemon=True,
            )
            thread.start()

            self.send_json(202, {
                "message": "EMOSH tracking job queued",
                "job_id": job_id,
                "status": "queued",
                "input_file": str(video_path),
            })
            return

        self.send_json(404, {"error": f"Endpoint {path} not found"})

def run_server(port=8000, host="127.0.0.1"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, EMOSHRequestHandler)
    print(f"==================================================")
    print(f"EMOSH-Tracker Microservice running at http://{host}:{port}")
    print(f"Health:  http://{host}:{port}/health")
    print(f"Sample:  http://{host}:{port}/api/sample")
    print(f"Track:   POST http://{host}:{port}/api/track")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down EMOSH service.")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port=port)
