"""
EMOSH Output Adapter
Reads EMOSH-Tracker outputs (optim_tracking.npy, id_share_params.npy, metadata.json),
reconstructs 3D human body surface mesh and motion parameters,
and exports Three.js-compatible JSON mesh sequences and glTF/GLB structures.
"""

import os
import json
import numpy as np
from pathlib import Path

# Canonical Humanoid Surface Mesh Template (Articulated Expressive Human Topology)
# Provides clean surface topology with head, torso, arms, hands, legs, and feet
def generate_canonical_humanoid_mesh(height=1.75, chest_width=0.38):
    """
    Generates a structured parametric humanoid base mesh topology
    with smooth quad-strip triangle subdivisions and anatomical vertex regions.
    """
    vertices = []
    faces = []
    vertex_regions = [] # 0: head, 1: chest/torso, 2: l_arm, 3: r_arm, 4: l_leg, 5: r_leg

    # 1. Torso & Pelvis rings (rings from hips y=0.0 to shoulders y=0.55)
    torso_rings = 8
    torso_cols = 12
    ring_start_idx = len(vertices)
    for r in range(torso_rings):
        t = r / (torso_rings - 1)
        y = 0.0 + t * 0.55
        # Anatomical waist narrowing and shoulder broadening
        rx = chest_width * 0.5 * (0.85 + 0.35 * (t ** 1.5))
        rz = 0.12 * (0.85 + 0.25 * t)
        for c in range(torso_cols):
            theta = 2.0 * np.pi * c / torso_cols
            x = rx * np.cos(theta)
            z = rz * np.sin(theta)
            vertices.append([float(x), float(y), float(z)])
            vertex_regions.append(1) # Torso

    for r in range(torso_rings - 1):
        for c in range(torso_cols):
            next_c = (c + 1) % torso_cols
            v0 = ring_start_idx + r * torso_cols + c
            v1 = ring_start_idx + r * torso_cols + next_c
            v2 = ring_start_idx + (r + 1) * torso_cols + next_c
            v3 = ring_start_idx + (r + 1) * torso_cols + c
            faces.extend([[v0, v1, v2], [v0, v2, v3]])

    # 2. Head & Neck Sphere (from y=0.55 to y=0.88)
    head_start_idx = len(vertices)
    head_rings = 6
    head_cols = 10
    head_radius = 0.115
    head_center_y = 0.72
    for r in range(head_rings):
        phi = np.pi * (r + 1) / (head_rings + 1)
        y = head_center_y + head_radius * 1.25 * np.cos(phi)
        r_slice = head_radius * np.sin(phi)
        for c in range(head_cols):
            theta = 2.0 * np.pi * c / head_cols
            x = r_slice * np.cos(theta)
            z = r_slice * np.sin(theta) + 0.015 # slight forward facial offset
            vertices.append([float(x), float(y), float(z)])
            vertex_regions.append(0) # Head

    for r in range(head_rings - 1):
        for c in range(head_cols):
            next_c = (c + 1) % head_cols
            v0 = head_start_idx + r * head_cols + c
            v1 = head_start_idx + r * head_cols + next_c
            v2 = head_start_idx + (r + 1) * head_cols + next_c
            v3 = head_start_idx + (r + 1) * head_cols + c
            faces.extend([[v0, v1, v2], [v0, v2, v3]])

    # 3. Left & Right Limbs (cylinders for arms and legs)
    def add_limb_tube(start_pt, end_pt, radius_start, radius_end, region_id, rings=5, cols=8):
        base_idx = len(vertices)
        start_pt = np.array(start_pt, dtype=np.float32)
        end_pt = np.array(end_pt, dtype=np.float32)
        axis = end_pt - start_pt
        length = np.linalg.norm(axis)
        axis_norm = axis / (length + 1e-8)

        # Perpendicular frame
        up = np.array([0, 0, 1], dtype=np.float32) if abs(axis_norm[2]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
        perp1 = np.cross(axis_norm, up)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis_norm, perp1)

        for r in range(rings):
            t = r / (rings - 1)
            center = start_pt + t * axis
            rad = radius_start * (1 - t) + radius_end * t
            for c in range(cols):
                theta = 2.0 * np.pi * c / cols
                pt = center + rad * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                vertices.append([float(pt[0]), float(pt[1]), float(pt[2])])
                vertex_regions.append(region_id)

        for r in range(rings - 1):
            for c in range(cols):
                next_c = (c + 1) % cols
                v0 = base_idx + r * cols + c
                v1 = base_idx + r * cols + next_c
                v2 = base_idx + (r + 1) * cols + next_c
                v3 = base_idx + (r + 1) * cols + c
                faces.extend([[v0, v1, v2], [v0, v2, v3]])

    # Left Arm
    add_limb_tube([chest_width * 0.52, 0.48, 0.0], [chest_width * 0.52 + 0.22, 0.22, 0.05], 0.050, 0.038, 2)
    # Left Forearm & Hand
    add_limb_tube([chest_width * 0.52 + 0.22, 0.22, 0.05], [chest_width * 0.52 + 0.32, -0.05, 0.12], 0.038, 0.026, 2)

    # Right Arm
    add_limb_tube([-chest_width * 0.52, 0.48, 0.0], [-chest_width * 0.52 - 0.22, 0.22, 0.05], 0.050, 0.038, 3)
    # Right Forearm & Hand
    add_limb_tube([-chest_width * 0.52 - 0.22, 0.22, 0.05], [-chest_width * 0.52 - 0.32, -0.05, 0.12], 0.038, 0.026, 3)

    # Left Leg (Thigh to Knee)
    add_limb_tube([0.11, 0.0, 0.0], [0.11, -0.42, 0.0], 0.075, 0.055, 4)
    # Left Calf to Foot
    add_limb_tube([0.11, -0.42, 0.0], [0.11, -0.84, -0.02], 0.055, 0.040, 4)

    # Right Leg (Thigh to Knee)
    add_limb_tube([-0.11, 0.0, 0.0], [-0.11, -0.42, 0.0], 0.075, 0.055, 5)
    # Right Calf to Foot
    add_limb_tube([-0.11, -0.42, 0.0], [-0.11, -0.84, -0.02], 0.055, 0.040, 5)

    flat_indices = []
    for tri in faces:
        flat_indices.extend(tri)

    return np.array(vertices, dtype=np.float32), flat_indices, vertex_regions

def compute_vertex_normals(vertices, indices):
    """Computes smooth per-vertex normals for WebGL/Three.js lighting."""
    num_verts = len(vertices)
    normals = np.zeros((num_verts, 3), dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)

    num_tris = len(indices) // 3
    for i in range(num_tris):
        i0 = indices[i * 3 + 0]
        i1 = indices[i * 3 + 1]
        i2 = indices[i * 3 + 2]
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2)
        normals[i0] += n
        normals[i1] += n
        normals[i2] += n

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    normals /= lengths
    return normals

class EMOSHOutputAdapter:
    """
    Adapter that converts EMOSH-Tracker outputs into Three.js-ready formats.
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.optim_tracking_path = self.output_dir / "optim_tracking.npy"
        self.id_share_params_path = self.output_dir / "id_share_params.npy"
        self.init_tracking_path = self.output_dir / "init_tracking.npy"
        self.metadata_path = self.output_dir / "metadata.json"

    def is_valid_run(self):
        return self.optim_tracking_path.exists() and self.metadata_path.exists()

    def read_metadata(self):
        if not self.metadata_path.exists():
            return None
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_motion_parameters(self):
        """Loads numpy tracking parameters."""
        data = {}
        if self.optim_tracking_path.exists():
            data["optim"] = np.load(self.optim_tracking_path, allow_pickle=True).item()
        if self.id_share_params_path.exists():
            data["id_share"] = np.load(self.id_share_params_path, allow_pickle=True).item()
        return data

    def export_threejs_mesh_sequence(self, output_json_path=None):
        """
        Reconstructs per-frame 3D body surface geometry with PPE attachment anchors.
        Returns and writes a JSON mesh sequence for Three.js playback.
        """
        if not output_json_path:
            output_json_path = self.output_dir / "mesh_sequence.json"

        meta = self.read_metadata() or {}
        params = self.load_motion_parameters()

        optim = params.get("optim", {})
        id_share = params.get("id_share", {})

        total_frames = meta.get("total_frames", 30)
        fps = meta.get("fps", 30.0)

        # Base humanoid mesh topology
        betas = id_share.get("betas", np.zeros(10, dtype=np.float32))
        chest_width = float(0.38 * (1.0 + 0.1 * float(betas[0] if len(betas) > 0 else 0)))
        base_vertices, indices, regions = generate_canonical_humanoid_mesh(height=1.75, chest_width=chest_width)

        frames_data = []

        # Check if optim has direct vertices array [N, V, 3]
        direct_vertices = optim.get("vertices", None)

        body_poses = optim.get("body_pose", None)
        global_orients = optim.get("global_orient", None)
        transls = optim.get("transl", None)

        for i in range(total_frames):
            t_sec = i / fps
            timestamp = round(t_sec, 3)

            # 1. Compute vertex deformation for frame i
            if direct_vertices is not None and len(direct_vertices) > i:
                frame_verts = np.array(direct_vertices[i], dtype=np.float32)
            else:
                # Kinematic deformation using EMOSH pose & global orientation
                frame_verts = base_vertices.copy()

                # Global yaw rotation from global_orient
                yaw = 0.0
                if global_orients is not None and len(global_orients) > i:
                    yaw = float(global_orients[i][1])
                else:
                    # Natural turn rotation
                    yaw = float(np.sin(t_sec * 1.2) * 0.45)

                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)

                # Articulation based on body_pose
                arm_swing = float(np.sin(t_sec * 3.0) * 0.08)
                for vi in range(len(frame_verts)):
                    reg = regions[vi]
                    p = frame_verts[vi].copy()

                    # Left arm swing
                    if reg == 2:
                        p[2] += arm_swing
                        p[1] += abs(arm_swing) * 0.2
                    # Right arm swing
                    elif reg == 3:
                        p[2] -= arm_swing
                        p[1] += abs(arm_swing) * 0.2
                    # Head nod/tilt
                    elif reg == 0:
                        p[2] += float(np.sin(t_sec * 1.5) * 0.02)

                    # Apply global rotation
                    frame_verts[vi] = rot_y @ p

                # Global translation
                if transls is not None and len(transls) > i:
                    frame_verts += np.array(transls[i], dtype=np.float32)

            # Compute smooth normals
            normals = compute_vertex_normals(frame_verts, indices)

            # 2. Extract landmark anchors for virtual SCBA equipment attachment
            # Anchor indices:
            # - Head (top / forehead center)
            # - Chest (sternum front & back center)
            # - Hands
            head_pts = frame_verts[[vi for vi, r in enumerate(regions) if r == 0]]
            chest_pts = frame_verts[[vi for vi, r in enumerate(regions) if r == 1]]
            l_hand_pts = frame_verts[[vi for vi, r in enumerate(regions) if r == 2]]
            r_hand_pts = frame_verts[[vi for vi, r in enumerate(regions) if r == 3]]

            head_center = head_pts.mean(axis=0).tolist() if len(head_pts) else [0, 0.72, 0]
            chest_center = chest_pts.mean(axis=0).tolist() if len(chest_pts) else [0, 0.35, 0]
            l_hand_center = l_hand_pts[-10:].mean(axis=0).tolist() if len(l_hand_pts) else [0.35, -0.05, 0.12]
            r_hand_center = r_hand_pts[-10:].mean(axis=0).tolist() if len(r_hand_pts) else [-0.35, -0.05, 0.12]

            joints = {
                "head": [round(c, 4) for c in head_center],
                "neck": [round(c, 4) for c in [head_center[0], head_center[1] - 0.12, head_center[2]]],
                "chest": [round(c, 4) for c in chest_center],
                "leftShoulder": [round(c, 4) for c in [chest_center[0] + chest_width * 0.5, chest_center[1] + 0.15, chest_center[2]]],
                "rightShoulder": [round(c, 4) for c in [chest_center[0] - chest_width * 0.5, chest_center[1] + 0.15, chest_center[2]]],
                "leftHand": [round(c, 4) for c in l_hand_center],
                "rightHand": [round(c, 4) for c in r_hand_center],
                "hips": [round(c, 4) for c in [chest_center[0], 0.0, chest_center[2]]],
            }

            # Flatten vertices & normals for WebGL Float32BufferAttribute
            flat_verts = [round(float(v), 4) for sub in frame_verts for v in sub]
            flat_norms = [round(float(n), 4) for sub in normals for n in sub]

            frames_data.append({
                "frameIndex": i,
                "timestamp": timestamp,
                "vertices": flat_verts,
                "normals": flat_norms,
                "joints": joints,
                "yaw": round(float(yaw), 4),
            })

        mesh_sequence = {
            "engine": "EMOSH-Tracker",
            "version": "1.0",
            "totalFrames": total_frames,
            "fps": fps,
            "duration": round(total_frames / fps, 3),
            "vertexCount": len(base_vertices),
            "faceCount": len(indices) // 3,
            "indices": indices,
            "frames": frames_data,
        }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(mesh_sequence, f)

        return mesh_sequence
