
import numpy as np


def normalise(vector):
    length = np.linalg.norm(vector)
    if length < 1e-6:
        return None
    return vector / length


def calculate_scba_anchor(world_landmarks):
    """
    Calculate an approximate SCBA cylinder attachment frame.

    world_landmarks: MediaPipe pose_world_landmarks
    Returns position and torso orientation axes.
    """

    # Extract 3D coordinates
    def point(index):
        lm = world_landmarks[index]
        return np.array([lm.x, lm.y, lm.z], dtype=float)

    left_shoulder = point(11)
    right_shoulder = point(12)
    left_hip = point(23)
    right_hip = point(24)
    nose = point(0)

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2

    # Torso vertical and lateral directions
    up = normalise(shoulder_mid - hip_mid)
    side = normalise(right_shoulder - left_shoulder)

    if up is None or side is None:
        return None

    # Calculate the normal to the torso plane
    forward = normalise(np.cross(side, up))

    if forward is None:
        return None

    # Resolve the direction: nose should be on the front side
    if np.dot(nose - shoulder_mid, forward) < 0:
        forward = -forward

    back = -forward

    # Estimate SCBA cylinder position
    shoulder_width = np.linalg.norm(
        right_shoulder - left_shoulder
    )

    position = (
        shoulder_mid
        + back * shoulder_width * 0.55
        - up * shoulder_width * 0.25
    )

    return {
        "position": position,
        "up": up,
        "side": side,
        "forward": forward,
        "back": back,
        "shoulder_width": shoulder_width
    }
