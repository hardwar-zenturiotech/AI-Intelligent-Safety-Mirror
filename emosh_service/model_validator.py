"""
Model Asset & Licensing Validator for EMOSH-Tracker Service
Verifies existence of SMPL-X, FLAME, and MANO parametric human models.
Provides explicit licensing documentation without violating third-party redistribution terms.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

REQUIRED_MODELS = {
    "smplx": {
        "name": "SMPL-X (Expressive Human Model)",
        "license_holder": "Max Planck Institute for Intelligent Systems",
        "url": "https://smpl-x.is.tue.mpg.de/",
        "license": "Custom non-commercial research license (registration required)",
        "files": ["SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL.pkl", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz"],
        "dir": ASSETS_DIR / "smplx",
    },
    "flame": {
        "name": "FLAME (3D Facial Model)",
        "license_holder": "Max Planck Institute for Intelligent Systems",
        "url": "https://flame.is.tue.mpg.de/",
        "license": "Custom non-commercial research license (registration required)",
        "files": ["generic_model.pkl", "flame2020.pkl"],
        "dir": ASSETS_DIR / "flame",
    },
    "mano": {
        "name": "MANO (Articulated Hand Model)",
        "license_holder": "Max Planck Institute for Intelligent Systems",
        "url": "https://mano.is.tue.mpg.de/",
        "license": "Custom non-commercial research license (registration required)",
        "files": ["MANO_LEFT.pkl", "MANO_RIGHT.pkl"],
        "dir": ASSETS_DIR / "mano",
    },
}

def validate_assets():
    """
    Checks whether required model assets exist on disk.
    Returns status dictionary with instructions for any missing models.
    """
    status = {
        "all_present": True,
        "models": {},
        "guidance": [],
    }

    for key, spec in REQUIRED_MODELS.items():
        model_dir = spec["dir"]
        present_files = []
        if model_dir.exists():
            for f in spec["files"]:
                if (model_dir / f).exists():
                    present_files.append(f)

        is_present = len(present_files) > 0
        if not is_present:
            status["all_present"] = False
            status["guidance"].append(
                f"Missing {spec['name']}. Register at {spec['url']} and download models into emosh_service/assets/{key}/"
            )

        status["models"][key] = {
            "name": spec["name"],
            "is_present": is_present,
            "present_files": present_files,
            "required_any_of": spec["files"],
            "license_url": spec["url"],
            "license": spec["license"],
            "target_dir": str(model_dir),
        }

    return status

if __name__ == "__main__":
    import json
    print(json.dumps(validate_assets(), indent=2))
