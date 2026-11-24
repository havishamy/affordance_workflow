"""Portable server integration helper: known base z=0, Z-only pose initialization.

Copy this module to the SERVER and invoke initialize_pose_z_only before model
pose refinement. It never changes input vertices, XY, rotation, scale or camera
calibration, and must not be invoked as post-solve scene correction.
"""
from __future__ import annotations

import json
import numpy as np


def reference_request() -> dict:
    return dict(frame="base", units="meters", normal=[0, 0, 1], offset_m=0.0,
                apply_stage="pose_initialization", allowed_adjustment="z_translation_only",
                preserve_object_xy=True, preserve_object_rotation=True,
                preserve_object_scale=True, postsolve_scene_transform=False)


def parse_reference(raw: str | None) -> dict | None:
    if raw is None:
        return None  # Explicit legacy behavior when no reference was requested.
    value = json.loads(raw)
    if not isinstance(value, dict) or value != reference_request():
        raise ValueError("Unsupported table_plane_reference: require base z=0, initialization-only Z translation, preserving XY/rotation/scale.")
    return value


def _pose(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("Expected a finite 4x4 column-vector SE(3) transform.")
    rotation = value[:3, :3]
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("Invalid homogeneous-transform last row.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Transform rotation must be orthonormal with determinant +1.")
    return value


def reference_in_camera(t_world_opencv_camera) -> dict:
    """Express the fixed base plane in the existing camera basis, not vice versa.

    p_world = R_wc p_camera + t_wc, hence n_camera = R_wc.T [0,0,1]
    and d_camera = t_wc.z for n_camera.p_camera+d_camera=0.
    No measured point, camera transform or robot pose is changed.
    """
    transform = _pose(t_world_opencv_camera)
    return dict(frame="opencv_camera", normal=transform[2, :3].tolist(),
                offset_m=float(transform[2, 3]))


def initialize_pose_z_only(vertices_object, t_world_object_initial, scale, reference: dict):
    """Align the initial model's lowest support vertex to base z=0.

    Feed the unmodified canonical model, original initial pose and estimated
    scale into this step. XY, R and scale are immutable within this operation.
    Return the new initial pose and its audit; do NOT apply this to solved data.
    """
    if reference != reference_request():
        raise ValueError("Invalid base-table initialization reference.")
    vertices = np.asarray(vertices_object, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Expected at least three finite canonical XYZ vertices.")
    before = _pose(t_world_object_initial)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape == ():
        scale = np.repeat(scale, 3)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Model scale must be positive, finite, scalar or length three.")
    initial_world = (vertices * scale) @ before[:3, :3].T + before[:3, 3]
    support_before = float(initial_world[:, 2].min())
    after = before.copy()
    after[2, 3] -= support_before  # The ONLY parameter changed by the reference.
    initialized_world = (vertices * scale) @ after[:3, :3].T + after[:3, 3]
    audit = {
        "T_world_object_before": before.tolist(),
        "T_world_object_after": after.tolist(),
        "scale_before": scale.tolist(), "scale_after": scale.tolist(),
        "support_min_z_before_m": support_before,
        "support_min_z_after_m": float(initialized_world[:, 2].min()),
    }
    return after, audit


def result_confirmation(reference: dict, object_audits: dict) -> dict:
    """Emit this only after the worker actually ran the initialization step."""
    if reference != reference_request() or not object_audits:
        raise ValueError("A valid requested reference and real per-object initialization audits are required.")
    return {**reference, "applied": True, "objects": object_audits}
