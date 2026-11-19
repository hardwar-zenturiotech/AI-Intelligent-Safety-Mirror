"""
GPU Telemetry & Hardware Resource Monitor for EMOSH-Tracker Service
Monitors NVIDIA GPU, VRAM usage, and driver compatibility.
"""

import subprocess
import json
import os

def get_gpu_info():
    """
    Queries NVIDIA GPU information via nvidia-smi.
    Falls back gracefully if nvidia-smi is unavailable.
    """
    gpu_info = {
        "cuda_available": False,
        "device_name": "CPU",
        "total_memory_mb": 0,
        "free_memory_mb": 0,
        "used_memory_mb": 0,
        "driver_version": "N/A",
        "vram_status": "none",
        "warning": None,
    }

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            name, total, free, used, driver = parts
            total_mb = int(float(total))
            free_mb = int(float(free))
            used_mb = int(float(used))

            gpu_info["cuda_available"] = True
            gpu_info["device_name"] = name
            gpu_info["total_memory_mb"] = total_mb
            gpu_info["free_memory_mb"] = free_mb
            gpu_info["used_memory_mb"] = used_mb
            gpu_info["driver_version"] = driver

            if total_mb < 6000:
                gpu_info["vram_status"] = "limited"
                gpu_info["warning"] = (
                    f"GPU has {total_mb}MB VRAM. Full-resolution EMOSH temporal window "
                    "recommends >= 6GB VRAM. Sequence batch downsampling recommended."
                )
            else:
                gpu_info["vram_status"] = "optimal"
    except Exception as e:
        gpu_info["warning"] = f"nvidia-smi query failed: {str(e)}. Running on CPU mode."

    return gpu_info

if __name__ == "__main__":
    print(json.dumps(get_gpu_info(), indent=2))
