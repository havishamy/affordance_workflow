"""Known base-table reference for server pose initialization only.

No post-solve scene rotation, XY translation, object rescaling or point-cloud
leveling is allowed here. Original server/depth coordinates remain authoritative.
"""
from __future__ import annotations

import numpy as np

from .geometry import validate_transform

TABLE_REFERENCE_PARAMETER = "table_plane_reference"


def table_initialization_reference() -> dict:
    return {
        "frame": "base",
        "units": "meters",
        "normal": [0, 0, 1],
        "offset_m": 0.0,
        "apply_stage": "pose_initialization",
        "allowed_adjustment": "z_translation_only",
        "preserve_object_xy": True,
        "preserve_object_rotation": True,
        "preserve_object_scale": True,
        "postsolve_scene_transform": False,
    }


def validate_table_initialization_reference(value: dict) -> dict:
    expected = table_initialization_reference()
    if not isinstance(value, dict) or value != expected:
        raise ValueError("Table reference must be base z=0 at pose initialization, Z translation only; preserve XY, rotation and scale.")
    return expected


def validate_table_initialization_result(result: dict, object_names, reference: dict | None = None) -> dict:
    """Require server confirmation and per-object proof of Z-only initialization."""
    expected = validate_table_initialization_reference(reference or table_initialization_reference())
    acknowledgement = result.get(TABLE_REFERENCE_PARAMETER)
    if not isinstance(acknowledgement, dict) or acknowledgement.get("applied") is not True:
        raise ValueError("Server did not confirm applying table_plane_reference at pose initialization.")
    if any(acknowledgement.get(key) != value for key, value in expected.items()):
        raise ValueError("Server table initialization confirmation disagrees with the requested Z-only/XY-preserving contract.")
    audits = acknowledgement.get("objects")
    if not isinstance(audits, dict):
        raise ValueError("Server table initialization confirmation lacks per-object audit data.")
    for name in object_names:
        audit = audits.get(name)
        if not isinstance(audit, dict):
            raise ValueError(f"Missing table initialization audit for {name}.")
        before = validate_transform(audit.get("T_world_object_before"), f"{name} initialization before")
        after = validate_transform(audit.get("T_world_object_after"), f"{name} initialization after")
        unchanged = after.copy()
        unchanged[2, 3] = before[2, 3]
        if not np.allclose(unchanged, before, atol=1e-9, rtol=0):
            raise ValueError(f"{name}: table initialization changed XY or rotation; only Z translation is allowed.")
        scales = []
        for key in ("scale_before", "scale_after"):
            scale = np.asarray(audit.get(key), dtype=float)
            if scale.shape == ():
                scale = np.repeat(scale, 3)
            if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"{name}: invalid {key} in table initialization audit.")
            scales.append(scale)
        if not np.array_equal(scales[0], scales[1]):
            raise ValueError(f"{name}: table initialization changed model scale.")
        bottom_before = float(audit.get("support_min_z_before_m", float("nan")))
        bottom_after = float(audit.get("support_min_z_after_m", float("nan")))
        if not np.all(np.isfinite([bottom_before, bottom_after])) or abs(bottom_after) > 1e-6:
            raise ValueError(f"{name}: initialization support is not on base z=0.")
        if abs((after[2,3]-before[2,3]) + bottom_before-bottom_after) > 1e-6:
            raise ValueError(f"{name}: initialization Z change disagrees with the support-point audit.")
    return acknowledgement


def server_accepts_table_reference(openapi: dict) -> bool:
    """Require a declared request field; never rely on ignored extra form data."""
    try:
        content = openapi["paths"]["/v1/complex-demo/jobs"]["post"]["requestBody"]["content"]
        schema = content["multipart/form-data"]["schema"]
        visited = set()
        while "$ref" in schema:
            ref = schema["$ref"]
            if ref in visited or not ref.startswith("#/components/schemas/"):
                return False
            visited.add(ref)
            schema = openapi["components"]["schemas"][ref.rsplit("/", 1)[-1]]
        return TABLE_REFERENCE_PARAMETER in schema.get("properties", {})
    except (KeyError, TypeError):
        return False
