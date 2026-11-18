"""OpenMMLab RTMPose Tracker with automatic MediaPipe Pose backup.

Executes RTMPose ONNX models using OpenCV's built-in DNN module (cv2.dnn)
or ONNX Runtime, with full sub-pixel SimCC coordinate decoding and
instant, seamless fallback to MediaPipe Pose.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


class Landmark(NamedTuple):
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


class PoseResult(NamedTuple):
    pose_landmarks: Optional[List[List[Landmark]]]
    pose_world_landmarks: Optional[List[List[Landmark]]]
    engine: str


# 17 COCO Keypoint IDs -> 33 MediaPipe Landmark IDs
COCO_TO_MEDIAPIPE = [
    (0, 0),   # nose
    (1, 2),   # left_eye
    (2, 5),   # right_eye
    (3, 7),   # left_ear
    (4, 8),   # right_ear
    (5, 11),  # left_shoulder
    (6, 12),  # right_shoulder
    (7, 13),  # left_elbow
    (8, 14),  # right_elbow
    (9, 15),  # left_wrist
    (10, 16), # right_wrist
    (11, 23), # left_hip
    (12, 24), # right_hip
    (13, 25), # left_knee
    (14, 26), # right_knee
    (15, 27), # left_ankle
    (16, 28), # right_ankle
]


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    split_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes SimCC 1D heatmaps into normalized [0, 1] coordinates and confidence scores.
    simcc_x: (1, 17, x_bins)
    simcc_y: (1, 17, y_bins)
    Returns:
        coords: (17, 2) in [0, 1] range
        scores: (17,) confidence values
    """
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    num_keypoints, x_bins = simcc_x.shape
    _, y_bins = simcc_y.shape

    max_val_x = np.max(simcc_x, axis=1)
    max_idx_x = np.argmax(simcc_x, axis=1)

    max_val_y = np.max(simcc_y, axis=1)
    max_idx_y = np.argmax(simcc_y, axis=1)

    # Sub-pixel refinement around peak
    refined_x = np.zeros(num_keypoints, dtype=np.float32)
    refined_y = np.zeros(num_keypoints, dtype=np.float32)

    for k in range(num_keypoints):
        sx0 = max(0, max_idx_x[k] - 3)
        sx1 = min(x_bins, max_idx_x[k] + 4)
        wx = np.exp(np.clip(simcc_x[k, sx0:sx1] - max_val_x[k], -20.0, 20.0))
        sum_wx = np.sum(wx)
        refined_x[k] = (
            np.sum(np.arange(sx0, sx1) * wx) / sum_wx if sum_wx > 0 else max_idx_x[k]
        )

        sy0 = max(0, max_idx_y[k] - 3)
        sy1 = min(y_bins, max_idx_y[k] + 4)
        wy = np.exp(np.clip(simcc_y[k, sy0:sy1] - max_val_y[k], -20.0, 20.0))
        sum_wy = np.sum(wy)
        refined_y[k] = (
            np.sum(np.arange(sy0, sy1) * wy) / sum_wy if sum_wy > 0 else max_idx_y[k]
        )

    norm_x = refined_x / float(x_bins)
    norm_y = refined_y / float(y_bins)

    # Reject boundary clamping artifacts: out-of-frame joints clamp against edge bins
    is_boundary = (max_idx_x <= 2) | (max_idx_x >= x_bins - 3) | (max_idx_y <= 2) | (max_idx_y >= y_bins - 3)
    mean_x = np.mean(simcc_x, axis=1)
    mean_y = np.mean(simcc_y, axis=1)
    prominence = np.minimum(max_val_x - mean_x, max_val_y - mean_y)

    scores = np.zeros(num_keypoints, dtype=np.float32)
    valid_mask = ~is_boundary & (prominence > 0.10)
    scores[valid_mask] = np.clip((prominence[valid_mask] - 0.10) / 0.35, 0.0, 1.0)

    coords = np.stack([norm_x, norm_y], axis=1)
    return coords, scores


def decode_simcc_3d(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_z: Optional[np.ndarray] = None,
    split_ratio: float = 2.0,
    depth_min: float = -1.2,
    depth_max: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes SimCC 1D heatmaps into 3D coordinates (x, y, metric_z) and confidence scores.
    simcc_x: (1, 17, x_bins)
    simcc_y: (1, 17, y_bins)
    simcc_z: Optional (1, 17, z_bins) root-relative depth in meters
    Returns:
        coords: (17, 3) where [:, 0:2] in [0, 1] range, and [:, 2] in meters
        scores: (17,) confidence values
    """
    coords_2d, scores = decode_simcc(simcc_x, simcc_y, split_ratio)
    num_keypoints = coords_2d.shape[0]

    coords_3d = np.zeros((num_keypoints, 3), dtype=np.float32)
    coords_3d[:, :2] = coords_2d

    if simcc_z is not None:
        if simcc_z.ndim == 3:
            simcc_z = simcc_z[0]
        z_bins = simcc_z.shape[1]
        max_idx_z = np.argmax(simcc_z, axis=1)
        max_val_z = np.max(simcc_z, axis=1)

        refined_z = np.zeros(num_keypoints, dtype=np.float32)
        for k in range(num_keypoints):
            sz0 = max(0, max_idx_z[k] - 3)
            sz1 = min(z_bins, max_idx_z[k] + 4)
            wz = np.exp(np.clip(simcc_z[k, sz0:sz1] - max_val_z[k], -20.0, 20.0))
            sum_wz = np.sum(wz)
            refined_z[k] = (
                np.sum(np.arange(sz0, sz1) * wz) / sum_wz if sum_wz > 0 else max_idx_z[k]
            )

        norm_z = refined_z / float(z_bins - 1)
        coords_3d[:, 2] = norm_z * (depth_max - depth_min) + depth_min

    return coords_3d, scores


class RTMPoseTracker:
    """
    OpenMMLab RTMPose / RTMPose-3D tracking engine with MediaPipe backup.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        backup_model_path: Optional[Path] = None,
        preferred_engine: str = "mediapipe",
    ):
        self.preferred_engine = preferred_engine
        self.active_engine = "mediapipe"
        self.net = None
        self.mediapipe_landmarker = None
        self.is_3d = (preferred_engine == "rtmpose3d")
        self.input_size = (256, 256) if self.is_3d else (192, 256)  # width, height
        self.start_time = 0

        # Model paths
        if model_path is not None:
            model_path = Path(model_path)
        elif self.is_3d:
            model_path = Path("models/rtmpose3d-m.onnx")
        else:
            model_path = Path("models/rtmpose-m.onnx")

        backup_model_path = Path(backup_model_path) if backup_model_path is not None else Path("models/pose_landmarker_lite.task")

        # 1. Initialize MediaPipe Pose as rock-solid engine / backup
        if _MEDIAPIPE_AVAILABLE and backup_model_path.is_file():
            try:
                base_options = python.BaseOptions(model_asset_path=str(backup_model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.mediapipe_landmarker = vision.PoseLandmarker.create_from_options(options)
                self.active_engine = "mediapipe"
            except Exception as e:
                print(f"[RTMPoseTracker] MediaPipe backup init note: {e}")

        # 2. Attempt RTMPose / RTMPose-3D initialization if requested
        if self.preferred_engine in ("rtmpose", "rtmpose3d") and model_path.is_file():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(model_path))
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.active_engine = self.preferred_engine
                print(f"[RTMPoseTracker] {self.active_engine.upper()} engine initialized from {model_path}")
            except Exception as e:
                print(f"[RTMPoseTracker] Could not load ONNX ({e}). Kept MediaPipe 3D Pose.")
                self.active_engine = "mediapipe"
        elif self.preferred_engine in ("rtmpose", "rtmpose3d"):
            print(f"[RTMPoseTracker] Model not found at {model_path}. Using MediaPipe 3D Pose.")

    def detect(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Runs pose detection on a BGR image frame.
        Returns a PoseResult compatible with MediaPipe landmark access.
        """
        if self.active_engine in ("rtmpose", "rtmpose3d") and self.net is not None:
            try:
                return self._detect_rtmpose(bgr_image)
            except Exception as e:
                print(f"[RTMPoseTracker] {self.active_engine} error, falling back to MediaPipe: {e}")
                if self.mediapipe_landmarker:
                    return self._detect_mediapipe(bgr_image)

        if self.mediapipe_landmarker:
            return self._detect_mediapipe(bgr_image)

        return PoseResult(pose_landmarks=None, pose_world_landmarks=None, engine="none")

    def _detect_rtmpose(self, bgr_image: np.ndarray) -> PoseResult:
        h, w = bgr_image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            bgr_image,
            scalefactor=1.0 / 58.395,
            size=self.input_size,
            mean=(123.675, 116.28, 103.53),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        out_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(out_names)

        simcc_x = outputs[0]
        simcc_y = outputs[1]
        simcc_z = outputs[2] if len(outputs) >= 3 else None
        coords, scores = decode_simcc_3d(simcc_x, simcc_y, simcc_z)

        # Convert 17 COCO keypoints to 33 MediaPipe landmarks
        landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.0, presence=0.0) for _ in range(33)]
        world_landmarks = [Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0) for _ in range(33)]

        for coco_idx, mp_idx in COCO_TO_MEDIAPIPE:
            kx, ky = coords[coco_idx, 0], coords[coco_idx, 1]
            kz = coords[coco_idx, 2] if coords.shape[1] > 2 else 0.0
            sc = float(scores[coco_idx])
            landmarks[mp_idx] = Landmark(x=float(kx), y=float(ky), z=float(kz), visibility=sc, presence=sc)

        # World landmarks calculation
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        ref_x = (l_sh.x + r_sh.x) / 2.0
        ref_y = (l_sh.y + r_sh.y) / 2.0
        sh_span = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        pixels_to_meters = 0.40 / max(sh_span, 0.05)

        for coco_idx, mp_idx in COCO_TO_MEDIAPIPE:
            lm = landmarks[mp_idx]
            if lm.visibility > 0.05:
                wx = (lm.x - ref_x) * pixels_to_meters
                wy = (lm.y - ref_y) * pixels_to_meters
                wz = lm.z if lm.z != 0.0 else 0.0
                world_landmarks[mp_idx] = Landmark(
                    x=wx, y=wy, z=wz, visibility=lm.visibility, presence=lm.presence
                )

        return PoseResult(
            pose_landmarks=[landmarks],
            pose_world_landmarks=[world_landmarks],
            engine=self.active_engine,
        )

    def _detect_mediapipe(self, bgr_image: np.ndarray) -> PoseResult:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.mediapipe_landmarker.detect(mp_image)
        return PoseResult(
            pose_landmarks=res.pose_landmarks if res.pose_landmarks else None,
            pose_world_landmarks=res.pose_world_landmarks if res.pose_world_landmarks else None,
            engine="mediapipe",
        )

    def close(self):
        if self.mediapipe_landmarker:
            self.mediapipe_landmarker.close()
