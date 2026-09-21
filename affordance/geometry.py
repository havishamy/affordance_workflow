from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    """Convert a UR axis-angle rotation vector to a 3x3 matrix."""
    vector = np.asarray(rotvec, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("Rotation vector must contain three finite values.")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    k = _skew(axis)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Convert [x, y, z, rx, ry, rz] into T_parent_child."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Pose must contain six finite values.")
    transform = np.eye(4)
    transform[:3, :3] = rotvec_to_matrix(values[3:])
    transform[:3, 3] = values[:3]
    return transform


def validate_transform(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid last row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"{name} rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError(f"{name} rotation determinant must be +1.")
    return transform


def transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    xyz = np.asarray(point, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("3D point must contain three finite values.")
    return transform[:3, :3] @ xyz + transform[:3, 3]
