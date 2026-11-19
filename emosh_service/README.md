# EMOSH-Tracker Service (Engine D)

Official integration of **EMOSH-Tracker** (*Expressive Motion and Shape Disentanglement for Human Animation*, ECCV 2026) for the **AI Intelligent Safety Mirror**.

---

## 1. Overview & Capabilities

EMOSH-Tracker serves as the **fourth tracking and reconstruction engine** in the Smart Safety Mirror alongside:
- **Engine A**: MediaPipe 3D Pose
- **Engine B**: RTMW3D (SimCC-3D)
- **Engine C**: TensorFlow BodyPix (24-part semantic segmentation)
- **Engine D**: **EMOSH-Tracker** (Full 3D human body surface mesh, hand, facial, and shape reconstruction)

EMOSH reconstructs Expressive Human Model (EHM) parameters combining:
- **GVHMR** for full-body initialization
- **HaMeR** for detailed hand articulation
- **TEASER** for facial expressions
- **SMPL-X, FLAME, and MANO** parametric geometry
- Global trajectory and temporal optimization

---

## 2. Directory Layout

```text
emosh_service/
├── .venv/                   # Isolated virtual environment (Python 3)
├── server.py                # REST API microservice (port 8000)
├── run_tracking.py          # Official EMOSH CLI pipeline runner
├── emosh_adapter.py         # Converts optim_tracking.npy to Three.js JSON / GLB
├── gpu_telemetry.py         # Hardware GPU and VRAM memory telemetry
├── model_validator.py       # Validates SMPL-X / FLAME / MANO license assets
├── requirements.txt         # Standalone dependencies
├── assets/                  # Max Planck Institute model weights
│   ├── smplx/               # SMPLX_NEUTRAL.npz, etc.
│   ├── flame/               # generic_model.pkl
│   └── mano/                # MANO_LEFT.pkl, MANO_RIGHT.pkl
├── inputs/                  # Staged input clips
└── outputs/                 # Output directories containing optim_tracking.npy, metadata.json, mesh_sequence.json
```

---

## 3. Parametric Model Licensing & Downloads

Parametric human models are owned by the **Max Planck Institute for Intelligent Systems (MPI)**. Under their license agreements, these models cannot be redistributed directly. Users must register and download models individually:

1. **SMPL-X Body Model**:
   - Register at: [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
   - Download: `models_smplx_v1_1.zip`
   - Extract `SMPLX_NEUTRAL.npz` into `emosh_service/assets/smplx/`

2. **FLAME Facial Model**:
   - Register at: [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)
   - Download FLAME 2020 (`generic_model.pkl`) into `emosh_service/assets/flame/`

3. **MANO Hand Model**:
   - Register at: [https://mano.is.tue.mpg.de/](https://mano.is.tue.mpg.de/)
   - Download `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` into `emosh_service/assets/mano/`

---

## 4. Hardware & VRAM Recommendations

- **Recommended**: NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g. RTX 3080/4080).
- **Minimum**: NVIDIA GPU with $\ge 4\text{ GB}$ VRAM (e.g. RTX 3050 Laptop GPU).
- On GPUs with $\le 4\text{ GB}$ VRAM, temporal sequence windowing or frame downsampling is automatically applied to prevent Out-Of-Memory (OOM) errors.

---

## 5. Running the Service

### Start REST Microservice
```powershell
.\emosh_service\.venv\Scripts\python.exe emosh_service/server.py 8000
```

### Run Offline CLI Pipeline on Video
```powershell
.\emosh_service\.venv\Scripts\python.exe emosh_service/run_tracking.py --input recordings/sample.mp4 --output emosh_service/outputs/run_01 --visualize
```

---

## 6. Output File Structure

Each tracking run emits:
- `optim_tracking.npy`: Optimized per-frame camera parameters, 3D body pose, hand articulation, and facial expression.
- `id_share_params.npy`: Shared identity parameters (`betas` shape coefficients).
- `init_tracking.npy`: Pre-optimization initialization parameters.
- `metadata.json`: Video dimensions, FPS, frame counts, GPU device, execution duration, and asset verification.
- `mesh_sequence.json`: Three.js WebGL-ready vertex buffers ($x,y,z$), smooth normals ($nx,ny,nz$), and joint anchors (`head`, `chest`, `leftHand`, `rightHand`) for virtual SCBA equipment attachment.
