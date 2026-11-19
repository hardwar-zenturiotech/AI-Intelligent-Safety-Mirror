# AI Intelligent Safety Mirror

A computer-vision POC for visualising industrial PPE on a live camera feed.

## Development roadmap

1. **Stage 1 – Pose detection:** OpenCV webcam + MediaPipe Pose Landmarker.
2. **Stage 2 – 2D PPE overlays:** transparent PNG assets anchored to body landmarks.
3. **Stage 3 – Tracking improvements:** smoothing, orientation and tracking-loss recovery.
4. **Stage 4 – 3D PPE:** GLB/glTF assets rendered with Three.js.
5. **Stage 5 – Kiosk UI:** PPE selection and approved safety guidance.

The initial POC is for visual awareness and demonstration only. It is not a PPE fit-validation or safety certification system.

## Requirements

- Python 3.10–3.12 recommended (use Python 3.11 if you encounter package compatibility issues)
- Webcam
- Windows, macOS or Linux

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the **Pose Landmarker Lite** task model from the official MediaPipe documentation:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

Save it as:

`models/pose_landmarker_lite.task`

Download the MediaPipe Face Landmarker task model from:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Save it as:

`frontend/public/models/face_landmarker.task`

Place the SCBA mask FBX at:

`frontend/public/models/scba_mask.fbx`

## Desktop Executable Software (Windows)

The application can be launched directly as a native Windows desktop software without manually opening terminals or browser tabs:

1. **Launch Directly**: Double-click [`SafetyMirror.exe`](file:///c:/Users/nihal/Desktop/Zenturiotech/AI-Intelligent-Safety-Mirror/AI-Intelligent-Safety-Mirror/SafetyMirror.exe) or [`Start-Safety-Mirror.bat`](file:///c:/Users/nihal/Desktop/Zenturiotech/AI-Intelligent-Safety-Mirror/AI-Intelligent-Safety-Mirror/Start-Safety-Mirror.bat).
2. **Install Desktop Shortcut**: Double-click [`CreateDesktopShortcut.bat`](file:///c:/Users/nihal/Desktop/Zenturiotech/AI-Intelligent-Safety-Mirror/AI-Intelligent-Safety-Mirror/CreateDesktopShortcut.bat) to place a branded shortcut on your Windows desktop.
3. **Rebuild Executable**: Run `powershell -File build_executable.ps1` or `npm run build:exe`.

### Executable Architecture & Capabilities
- **Native GUI Launcher**: Compiled native `.exe` with embedded application icon, high-DPI scaling manifest, and no black console window popup.
- **Embedded Web Server**: Built-in HTTP listener that serves bundled production assets, handles media streaming, and processes `/api/recordings` uploads to the `recordings/` folder.
- **Dedicated Window Mode**: Launches in dedicated standalone window mode (`--app`) with zero browser address/tab clutter, custom 1440x900 resolution, and persistent camera permissions.
- **Hardware GPU Acceleration**: Pre-configured flags for WebGPU, WebGL 2.0, Vulkan/Direct3D 11 ANGLE, and SIMD WebAssembly for maximum ONNX Runtime Web and TensorFlow.js inference framerates.
- **Microservice Orchestration**: Automatically spawns and monitors the Python EMOSH reconstruction microservice on port 8000, and gracefully terminates all background processes upon closing the window.

## Run

From the repository root:

```powershell
python src/pose_detection.py
```

The app automatically creates a `recordings/` folder and saves each session as an MP4 file in that folder. You can override the folder with:

```powershell
python src/pose_detection.py --record-dir recordings
```

Press **Q** or **Ctrl+Q** to quit.

## Browser frontend

The browser UI provides a local camera preview, MediaPipe face and pose tracking,
a Three.js avatar overlay, an optional SCBA mask, and a manually selected PPE
checklist. Use the rendering-mode selector for the live-camera overlay or the 3D
environment. Pose and avatar skeletons, joint markers, pose-inference/render
FPS, and tracking coverage are available under **Motion capture diagnostics**.
The MediaPipe skeleton is hidden unless its debug checkbox is enabled. Camera
access is requested only after selecting **Start camera**.

The motion-capture avatar must be a rigged humanoid GLB at
`models/avatar.glb`, with at least one `SkinnedMesh` and a usable humanoid
skeleton. The app inspects and reports the bones it detects, maps, and cannot
find. Bone naming varies between assets, so check the runtime mapping report
after replacing the GLB. The MediaPipe Pose Landmarker task is expected at
`models/pose_landmarker_lite.task`.

Pose inference uses the unmirrored video element; selfie mirroring is CSS-only.
The camera-mode avatar canvas is mirrored with the video, while root placement
uses the unmirrored normalized landmarks. Debug landmarks use the displayed
mirror so anatomical labels stay attached to the same image joints. The
coordinate adapter treats world landmarks as camera-aligned (+X image-right,
+Y up, +Z away) and converts them to Three.js (+Y up, +Z toward the viewer).
MediaPipe documents hip-centred world coordinates in meters but does not state
all axis signs in the task API documentation, so verify signs against live
landmark labels and the calibration axes on the target camera. Synthetic input
is constructed in this convention; it is not evidence that the live feed uses
the same convention.

Segment rest directions are captured from the loaded rig in avatar-root
coordinates. Retargeted rotations are converted through each bone's current
parent transform rather than applying a generic bone-axis assumption. Before
segment alignment, MediaPipe directions are transformed through the inverse
fitted avatar-root transform, so camera-mode nonuniform scaling does not skew
the target direction. The model's reported forward direction is inferred from
its named left/right shoulders and vertical axis. Torso orientation and face orientation
from the nose/ear landmarks remain estimates; proportions, axis differences,
and head landmark noise may require model-specific calibration. In camera mode,
horizontal scale is fitted from the detected shoulder span (hip span is the
fallback), while vertical scale is fitted from nose-to-ankle height (torso or
nose-to-shoulder height are fallbacks). These are compared with dimensions
measured from the loaded rig rather than assumed bone lengths. The supplied
model's measured shoulder-to-height ratio differs substantially from the
synthetic reference pose, so the camera overlay currently uses separate
horizontal and vertical scale factors to align screen-space joints. World
landmarks drive bone orientation, while camera-mode root scale is fitted to
the detected body dimensions. This can stretch the avatar and does not
preserve the GLB's original proportions. Bone segment lengths retain the
loaded rig's proportions, so residual joint offsets can remain when its
proportions differ from the detected person. Use a model with closer body
proportions or calibrate its dimensions before treating the overlay as
anatomically aligned. The 3D environment mode keeps uniform scaling.
Hip-centred world landmarks do not provide absolute camera depth, so
screen placement and scale remain image-based approximations. Pose tracking
continues if the optional face/mask overlay cannot initialize; the face overlay
still requires its own task and mask assets.

After the face-tracked overlay is active, select **Record clip** and then
**Stop and save** to store the composited camera-and-mask video as a WebM file
in the repository's `recordings/` folder. Browser recording is available through
the local Vite development server; clips are limited to 250 MB.

From the `frontend/` directory, start the local development server:

```powershell
npm install
npm run dev
```

Open http://localhost:5173 in a browser and select **Start camera** to allow
camera access and start live pose inference. The pose task model must be present
at `models/pose_landmarker_lite.task`; the rigged avatar must be at
`models/avatar.glb`.

### Live webcam motion test

1. Optionally select **Synthetic motion test (debug only)** to confirm that the
   avatar cycles through neutral, separate arm raises, elbow bends, head/torso
   movement, knee bends, partial occlusion, and loss/reacquisition. The status
   explicitly labels this input as synthetic; it does not test the webcam or
   MediaPipe.
2. Turn the synthetic test off, select **Start camera**, grant permission, and
   stand far enough back for the required joints to remain in frame. Confirm
   the status changes to live tracking, landmark coverage is shown, and pose
   inference and render FPS appear.
3. With **Show MediaPipe skeleton** off, test neutral standing, raise the left
   and right arms independently, bend each elbow, turn the head, lean the
   torso, bend each knee, briefly occlude part of the body, then move out of
   view and return. The avatar should follow visible joints and ease toward rest
   during tracking loss. Enable the skeleton only when diagnosing landmark
   alignment.
4. Check **Detected bones / Mapped / Missing** in the diagnostic report. Record
   any pose loss, jitter, left/right mismatch, or delayed reacquisition along
   with the displayed FPS.
5. Enable **Show retargeting calibration** to see anatomical left/right joint
   labels, torso up/forward indicators, mapped avatar bone axes, head-forward
   direction, and torso/head yaw-pitch-roll readouts. Use these diagnostics to
   distinguish landmark interpretation errors from model rest-axis issues.

The synthetic sequence is an automated UI/debug aid, not a substitute for
testing with a real person and camera. Browser automation cannot validate body
poses, occlusion handling, or camera-specific tracking quality; report those as
untested unless physically exercised on the target device.

### Multi-Engine Tracking Architecture

The mirror integrates three independent tracking and vision engines running concurrently:

1. **MediaPipe 3D Pose Landmarker**:
   - 33 landmarks with root-relative metric 3D depth.
   - Smooth 1-Euro filtering and real-time retargeting for the rigged 3D avatar and SCBA PPE equipment.
2. **RTMW3D (Whole-Body 3D Pose)**:
   - 133-keypoint 3D pose detection powered by OpenMMLab RTMW3D-X and ONNX Runtime Web.
   - Real-time 3D diagnostic skeleton and 2D keypoint projections.
3. **TensorFlow.js BodyPix (24 Body-Part Semantic Segmentation)**:
   - Pixel-level body-part classification powered by `@tensorflow-models/body-segmentation` on the WebGL backend.
   - 24 anatomical part labels grouped into head, torso, left/right arms, left/right legs, and feet.
   - Live multi-mode canvas overlay aligned with the webcam feed.
   - Independent throttled inference (12–15 FPS) to preserve high-framerate Three.js rendering (30–60 FPS).

---

### BodyPix Body-Part Segmentation Setup & Usage

#### Installation & Dependencies

Inside `frontend/`:

```powershell
npm install @tensorflow-models/body-segmentation @tensorflow/tfjs-core @tensorflow/tfjs-converter @tensorflow/tfjs-backend-webgl
```

#### Architecture & Configuration

- **Module**: `frontend/src/bodyPartSegmentation.js`
- **Model**: `bodySegmentation.SupportedModels.BodyPix`
- **Architecture**: `MobileNetV1`
- **Output Stride**: `16`
- **Multiplier**: `0.75`
- **Quantization Bytes**: `2`
- **Internal Resolution**: `medium`
- **Segmentation Threshold**: `0.5`
- **Target Inference Frequency**: `12–15 FPS` (independent non-blocking asynchronous turns)

#### 24 Body-Part Label Mapping

| IDs | Anatomical Labels | Grouped Limb | Default Palette |
| :--- | :--- | :--- | :--- |
| **0, 1** | `left_face`, `right_face` | Head & Face | Golden Warm Coral |
| **2, 3** | `left_upper_arm_front`, `left_upper_arm_back` | Left Upper Arm | Vivid Magenta |
| **6, 7** | `left_lower_arm_front`, `left_lower_arm_back` | Left Lower Arm | Salmon / Peach |
| **10** | `left_hand` | Left Hand | Bright Rose Red |
| **4, 5** | `right_upper_arm_front`, `right_upper_arm_back` | Right Upper Arm | Deep Blue |
| **8, 9** | `right_lower_arm_front`, `right_lower_arm_back` | Right Lower Arm | Sky Blue |
| **11** | `right_hand` | Right Hand | Electric Cyan |
| **12, 13** | `torso_front`, `torso_back` | Torso | Safety Emerald Neon |
| **14, 15** | `left_upper_leg_front`, `left_upper_leg_back` | Left Upper Leg | Chartreuse Lime |
| **18, 19** | `left_lower_leg_front`, `left_lower_leg_back` | Left Lower Leg | Yellow Green |
| **16, 17** | `right_upper_leg_front`, `right_upper_leg_back` | Right Upper Leg | Leaf Green |
| **20, 21** | `right_lower_leg_front`, `right_lower_leg_back` | Right Lower Leg | Spring Green |
| **22, 23** | `left_foot`, `right_foot` | Feet | Royal Violet |

#### How to Activate & Test Segmentation Debug Modes

1. In the browser UI, open **Motion capture diagnostics**.
2. Ensure **BodyPix 24 body-part segmentation** is checked.
3. Select the desired **Visual mode**:
   - **Coloured overlay**: Semi-transparent (alpha ~0.65) rainbow overlay over the mirrored webcam.
   - **Mask only**: Opaque segmentation view over a dark background.
   - **Specific limb**: Isolates a single limb group using the **Limb** dropdown (e.g. `Left Arm`, `Torso`, `Head & Face`).
   - **Off**: Disables the visual overlay while keeping the engine available for other tracking modules.
4. Review live diagnostics:
   - `BodyPix Parts: XX FPS (XXms)`
   - `Dims: WxH`
   - `People: X`

#### Known Limitations

- **2D Semantic Segmentation Only**: BodyPix provides 2D pixel classifications. Front/back labels are statistical 2D classification outputs and must not be treated as true physical 3D metric depth coordinates.
- **Occlusion Dynamics**: When arms cross in front of the torso, 2D segmentation will group boundaries; MediaPipe or RTMW3D 3D joint depths remain the source of truth for avatar bone rotations.
- **GPU Resource Management**: Running simultaneous WebGL/WASM models (MediaPipe, RTMW3D, BodyPix, Three.js) shares GPU resources; BodyPix is throttled to 12–15 FPS to prevent WebGL context exhaustion or frame drops.

---

### Engine D: EMOSH-Tracker (Full 3D Human Body Mesh Reconstruction)

EMOSH-Tracker (*Expressive Motion and Shape Disentanglement for Human Animation*, ECCV 2026) operates as the **fourth tracking and reconstruction engine** in the AI Intelligent Safety Mirror.

#### Architecture & Capabilities

While MediaPipe and RTMW3D provide sparse 2D/3D skeletal keypoints and BodyPix provides 2D semantic masks, EMOSH-Tracker performs **volumetric 3D human body surface mesh reconstruction**:
- **SMPL-X, FLAME, and MANO** expressive geometry (body, face, and articulated hands).
- **GVHMR** body pose initialization + **HaMeR** hand initialization + **TEASER** facial expressions.
- **Motion-Shape Disentanglement**: Separates individual subject body shape (`betas`) from temporal motion dynamics.
- **Three.js Output Adapter**: Converts `optim_tracking.npy` and `id_share_params.npy` into per-frame vertex coordinates ($x,y,z$), smooth normals ($nx,ny,nz$), and anatomical SCBA equipment anchors (`head`, `chest`, `leftHand`, `rightHand`).

#### Visual Debugging Modes (Modes 1–6)

The application provides a dedicated visual mode selector:
1. **Mode 1: Camera Only** — Live webcam stream with overlays hidden.
2. **Mode 2: MediaPipe Skeleton** — 2D landmark connections and keypoint markers.
3. **Mode 3: RTMW3D 3D Skeleton** — SimCC-3D joint coordinates with metric depth.
4. **Mode 4: BodyPix Segmentation** — 24-part coloured semantic masks.
5. **Mode 5: EMOSH 3D Reconstruction** — Reconstructed 3D human body surface mesh rendered with WebGL PBR lighting and wireframe toggle.
6. **Mode 6: Combined View** — Side-by-side / multi-layer view with live tracking, segmentation, and 3D reconstruction.

#### Parametric Model Licensing & Downloads

Because parametric body assets are copyrighted by the **Max Planck Institute (MPI)**, their weights cannot be redistributed:
- **SMPL-X**: Register and download from [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/) into `emosh_service/assets/smplx/`
- **FLAME**: Register and download from [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/) into `emosh_service/assets/flame/`
- **MANO**: Register and download from [https://mano.is.tue.mpg.de/](https://mano.is.tue.mpg.de/) into `emosh_service/assets/mano/`

#### Running the Isolated EMOSH Microservice

```powershell
# Run REST Microservice (Port 8000)
.\emosh_service\.venv\Scripts\python.exe emosh_service/server.py 8000

# Run Offline CLI Tracking on Recorded Clip
.\emosh_service\.venv\Scripts\python.exe emosh_service/run_tracking.py --input recordings/sample.mp4 --output emosh_service/outputs/run_01 --visualize
```

---

## Project structure

```text
AI-Intelligent-Safety-Mirror/
├── assets/
│   └── ppe/                       # Transparent PNG PPE assets
├── models/
│   ├── avatar.glb                 # Rigged humanoid GLB for motion retargeting
│   └── pose_landmarker_lite.task  # MediaPipe pose model
├── src/
│   ├── pose_detection.py          # Webcam pose detection and recording
│   ├── rtmpose_tracker.py         # RTMPose Python reference
│   └── scba_tracking.py           # SCBA anchor from landmarks
├── emosh_service/                 # Engine D: EMOSH-Tracker Service
│   ├── .venv/                     # Isolated virtual environment
│   ├── server.py                  # REST API microservice
│   ├── run_tracking.py            # Official EMOSH CLI runner
│   ├── emosh_adapter.py           # Three.js JSON & GLB mesh converter
│   ├── gpu_telemetry.py           # GPU & VRAM monitor
│   ├── model_validator.py         # SMPL-X / FLAME / MANO license validator
│   ├── requirements.txt
│   ├── README.md
│   ├── assets/                    # Parametric model storage (SMPL-X, FLAME, MANO)
│   ├── inputs/                    # Staged video clips
│   └── outputs/                   # Tracking outputs (optim_tracking.npy, mesh_sequence.json)
├── frontend/
│   ├── public/
│   │   └── models/
│   │       ├── avatar.glb
│   │       ├── emosh_sample.json  # Pre-computed EMOSH 3D human body mesh
│   │       ├── face_landmarker.task
│   │       └── scba_mask.fbx
│   ├── src/
│   │   ├── tracking/
│   │   │   ├── emoshClient.js     # EMOSH service client
│   │   │   └── trackingManager.js # Unified 4-engine tracking manager
│   │   ├── visualization/
│   │   │   └── bodyMeshViewer.js  # Reconstructed 3D human mesh & PPE player
│   │   ├── bodyPartSegmentation.js # Engine C: BodyPix 24-part segmentation
│   │   └── rtmw3dTracker.js       # Engine B: RTMW3D SimCC-3D
│   ├── index.html                 # Kiosk UI with Unified Diagnostics Dashboard
│   ├── main.js                    # Application coordinator
│   ├── avatar.js                  # 3D avatar & PPE attachment assembly
│   ├── retargeting.js             # Motion retargeting & 360° turn orientation
│   ├── poseTracker.js             # Engine A: MediaPipe 3D Pose
│   ├── scbaApparatus.js           # 3D SCBA equipment models
│   ├── test_emosh.mjs             # EMOSH & TrackingManager test suite
│   ├── test_body_part_segmentation.mjs
│   ├── test_retargeting.mjs
│   └── package.json
├── .gitignore
├── requirements.txt
└── README.md
```

## Stage 1 acceptance checklist

- [ ] Webcam opens.
- [ ] A single person is detected.
- [ ] Body landmarks and skeleton are visible.
- [ ] Skeleton follows movement.
- [ ] Tracking recovers after temporary loss.
- [ ] FPS is displayed.

## Notes

- Downloaded `.task` model files are ignored by Git; provide them locally
  before running the relevant demos.
- Keep the camera view well lit and ensure the user's full body is visible.
- MediaPipe landmark coordinates are estimates. They are not precise anatomical measurements.
