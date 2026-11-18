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

    min_val = np.minimum(max_val_x, max_val_y)
    scores = 1.0 / (1.0 + np.exp(-np.clip(min_val, -10.0, 10.0)))

    coords = np.stack([norm_x, norm_y], axis=1)
    return coords, scores


class RTMPoseTracker:
    """
    OpenMMLab RTMPose tracking engine with MediaPipe backup.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        backup_model_path: Optional[Path] = None,
        preferred_engine: str = "rtmpose",
    ):
        self.preferred_engine = preferred_engine
        self.active_engine = "mediapipe"
        self.net = None
        self.mediapipe_landmarker = None
        self.input_size = (192, 256)  # width, height
        self.start_time = 0

        # Model paths
        if model_path is None:
            model_path = Path("models/rtmpose-m.onnx")
        if backup_model_path is None:
            backup_model_path = Path("models/pose_landmarker_lite.task")

        # 1. Initialize MediaPipe Pose as backup
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

        # 2. Attempt RTMPose initialization
        if self.preferred_engine == "rtmpose" and model_path.is_file():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(model_path))
                # Enable OpenCV DNN acceleration if available
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.active_engine = "rtmpose"
                print(f"[RTMPoseTracker] RTMPose engine initialized from {model_path} (SOTA)")
            except Exception as e:
                print(f"[RTMPoseTracker] Could not load RTMPose ONNX ({e}). Kept MediaPipe backup.")
                self.active_engine = "mediapipe"
        elif self.preferred_engine == "rtmpose":
            print(f"[RTMPoseTracker] RTMPose model not found at {model_path}. Using MediaPipe backup.")

    def detect(self, bgr_image: np.ndarray) -> PoseResult:
        """
        Runs pose detection on a BGR image frame.
        Returns a PoseResult compatible with MediaPipe landmark access.
        """
        if self.active_engine == "rtmpose" and self.net is not None:
            try:
                return self._detect_rtmpose(bgr_image)
            except Exception as e:
                print(f"[RTMPoseTracker] RTMPose error, falling back to MediaPipe: {e}")
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
        coords, scores = decode_simcc(simcc_x, simcc_y)

        # Convert 17 COCO keypoints to 33 MediaPipe landmarks
        landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=0.0, presence=0.0) for _ in range(33)]
        world_landmarks = [Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0, presence=0.0) for _ in range(33)]

        for coco_idx, mp_idx in COCO_TO_MEDIAPIPE:
            kx, ky = coords[coco_idx]
            sc = float(scores[coco_idx])
            landmarks[mp_idx] = Landmark(x=float(kx), y=float(ky), z=0.0, visibility=sc, presence=sc)

        # World landmarks calculation
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        ref_x = (l_sh.x + r_sh.x) / 2.0
        ref_y = (l_sh.y + r_sh.y) / 2.0
        sh_span = np.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        pixels_to_meters = 0.40 / max(sh_span, 0.05)

        for _, mp_idx in COCO_TO_MEDIAPIPE:
            lm = landmarks[mp_idx]
            if lm.visibility > 0.05:
                wx = (lm.x - ref_x) * pixels_to_meters
                wy = (lm.y - ref_y) * pixels_to_meters
                world_landmarks[mp_idx] = Landmark(
                    x=wx, y=wy, z=0.0, visibility=lm.visibility, presence=lm.presence
                )

        return PoseResult(
            pose_landmarks=[landmarks],
            pose_world_landmarks=[world_landmarks],
            engine="rtmpose",
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
