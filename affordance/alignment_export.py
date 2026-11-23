from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import validate_transform
from .three_stage_result import _read_ply_xyz


DEPTH_COLOR = (170, 170, 170)
MODEL_COLORS = {
    "wide_mouth_bottle": (255, 60, 60),
    "beaker": (60, 120, 255),
    "dropper": (255, 60, 220),
}
SOURCE_IDS = {
    "depth": 0,
    "wide_mouth_bottle": 1,
    "beaker": 2,
    "dropper": 3,
}


@dataclass(frozen=True)
class AlignmentExport:
    output_path: Path
    glb_path: Path
    manifest_path: Path
    depth_points: int
    model_points: dict[str, int]
    transform_source: str
    local_transform_max_abs_difference: float | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _stage_json(log_text: str, stage: str) -> dict[str, Any]:
    marker = f"[STAGE] {stage}"
    marker_at = log_text.find(marker)
    if marker_at < 0:
        raise ValueError(f"pipeline.log does not contain {marker}.")
    object_at = log_text.find("{", marker_at + len(marker))
    if object_at < 0:
        raise ValueError(f"pipeline.log stage {stage} has no JSON object.")
    try:
        value, _ = json.JSONDecoder().raw_decode(log_text[object_at:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot decode JSON for pipeline stage {stage}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"pipeline.log stage {stage} must contain a JSON object.")
    return value


def server_world_camera(result_dir: Path) -> np.ndarray:
    log_path = result_dir / "pipeline.log"
    try:
        stage = _stage_json(log_path.read_text(encoding="utf-8", errors="replace"), "calibration_validation")
    except OSError as exc:
        raise ValueError(f"Cannot read {log_path}: {exc}") from exc
    return validate_transform(stage.get("T_world_opencv_camera"), "T_world_opencv_camera")


def local_world_camera(capture_dir: Path, calibration_path: Path) -> np.ndarray:
    t_world_ee = validate_transform(
        _read_json(capture_dir / "T_world_ee.json").get("T_world_ee"), "T_world_ee",
    )
    t_ee_camera = validate_transform(
        _read_json(calibration_path).get("T_ee_cam"), "T_ee_cam",
    )
    return validate_transform(t_world_ee @ t_ee_camera, "local T_world_opencv_camera")


def depth_to_world(
    depth_u16: np.ndarray,
    intrinsics: dict[str, Any],
    depth_scale: float,
    t_world_camera: np.ndarray,
    *,
    stride: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    if depth_u16.ndim != 2 or depth_u16.dtype != np.uint16:
        raise ValueError("Depth image must be a two-dimensional uint16 array.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be finite and positive.")
    if not 0 <= min_depth_m < max_depth_m:
        raise ValueError("Depth limits must satisfy 0 <= min_depth_m < max_depth_m.")

    height, width = depth_u16.shape
    if int(intrinsics.get("width", -1)) != width or int(intrinsics.get("height", -1)) != height:
        raise ValueError("Camera dimensions do not match the depth image.")
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    if not all(np.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("Camera intrinsics are invalid.")

    rows = np.arange(0, height, stride)
    columns = np.arange(0, width, stride)
    u, v = np.meshgrid(columns, rows)
    z = depth_u16[np.ix_(rows, columns)].astype(np.float64) * depth_scale
    valid = (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        raise ValueError("No valid depth points remain after filtering.")
    z = z[valid]
    camera_points = np.column_stack((
        (u[valid] - cx) * z / fx,
        (v[valid] - cy) * z / fy,
        z,
    ))
    world_points = camera_points @ t_world_camera[:3, :3].T + t_world_camera[:3, 3]
    if not np.all(np.isfinite(world_points)):
        raise ValueError("Depth projection produced non-finite world points.")
    return world_points


def write_colored_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    source_ids: np.ndarray,
    comments: list[str],
) -> None:
    xyz = np.asarray(points, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.uint8)
    sources = np.asarray(source_ids, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 1 or not np.all(np.isfinite(xyz)):
        raise ValueError("PLY points must be a non-empty finite [N,3] array.")
    if rgb.shape != xyz.shape or sources.shape != (len(xyz),):
        raise ValueError("PLY colors/source IDs do not match the points.")

    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("source_id", "u1"),
    ])
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index].astype(np.float32)
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, index]
    vertices["source_id"] = sources

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        *[f"comment {comment}" for comment in comments],
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar source_id",
        "end_header",
        "",
    ]
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        vertices.tofile(handle)


def write_colored_glb(path: Path, components: list[tuple[str, np.ndarray, tuple[int, int, int]]]) -> None:
    """Write colored point primitives as a dependency-free glTF 2.0 binary file."""
    binary = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    primitives: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []

    def append_buffer(data: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        return index

    for name, points_value, color_value in components:
        points = np.asarray(points_value, dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.all(np.isfinite(points)):
            raise ValueError(f"GLB component {name} must contain finite [N,3] points.")
        color = np.asarray((*color_value, 255), dtype=np.uint8)
        colors = np.tile(color, (len(points), 1))
        position_view = append_buffer(points.astype("<f4", copy=False).tobytes(), 34962)
        color_view = append_buffer(colors.tobytes(), 34962)
        position_accessor = len(accessors)
        accessors.append({
            "bufferView": position_view,
            "componentType": 5126,
            "count": len(points),
            "type": "VEC3",
            "min": points.min(axis=0).astype(float).tolist(),
            "max": points.max(axis=0).astype(float).tolist(),
        })
        color_accessor = len(accessors)
        accessors.append({
            "bufferView": color_view,
            "componentType": 5121,
            "normalized": True,
            "count": len(points),
            "type": "VEC4",
        })
        material_index = len(materials)
        materials.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        })
        primitives.append({
            "attributes": {"POSITION": position_accessor, "COLOR_0": color_accessor},
            "material": material_index,
            "mode": 0,
        })

    gltf = {
        "asset": {"version": "2.0", "generator": "affordance.export_alignment_ply"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "depth_model_alignment_world", "mesh": 0}],
        "meshes": [{"name": "depth_model_alignment_world", "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary.extend(b"\x00" * ((-len(binary)) % 4))
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total_length))
        handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        handle.write(json_chunk)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)


def export_alignment_ply(
    capture_dir: Path,
    result_dir: Path,
    output_path: Path,
    calibration_path: Path,
    *,
    stride: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> AlignmentExport:
    capture_dir = capture_dir.resolve()
    result_dir = result_dir.resolve()
    output_path = output_path.resolve()
    calibration_path = calibration_path.resolve()
    result = _read_json(result_dir / "result.json")
    if str(result.get("capture_id", "")) != capture_dir.name:
        raise ValueError("Result capture_id does not match the capture directory.")
    if result.get("frame") != "world" or result.get("units") != "meters":
        raise ValueError("Result must use world coordinates in meters.")

    depth = cv2.imread(str(capture_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Cannot read {capture_dir / 'depth.png'}.")
    camera = _read_json(capture_dir / "camera.json")
    depth_scale = float(camera.get("depth_scale_m_per_unit", float("nan")))
    server_transform = server_world_camera(result_dir)
    local_transform = local_world_camera(capture_dir, calibration_path)
    transform_difference = float(np.max(np.abs(server_transform - local_transform)))
    depth_points = depth_to_world(
        depth, camera, depth_scale, server_transform,
        stride=stride, min_depth_m=min_depth_m, max_depth_m=max_depth_m,
    )

    point_blocks = [depth_points]
    color_blocks = [np.tile(np.asarray(DEPTH_COLOR, dtype=np.uint8), (len(depth_points), 1))]
    source_blocks = [np.full(len(depth_points), SOURCE_IDS["depth"], dtype=np.uint8)]
    model_counts: dict[str, int] = {}
    components: list[tuple[str, np.ndarray, tuple[int, int, int]]] = [
        ("observed_depth", depth_points, DEPTH_COLOR),
    ]
    objects = result.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Result objects must be a mapping.")
    for name in ("wide_mouth_bottle", "beaker", "dropper"):
        obj = objects.get(name)
        if not isinstance(obj, dict):
            raise ValueError(f"Result is missing object {name}.")
        artifact = obj.get("model_points_world_artifact")
        if not isinstance(artifact, str) or Path(artifact).is_absolute() or ".." in Path(artifact).parts:
            raise ValueError(f"{name} has an unsafe model artifact path.")
        model_points = _read_ply_xyz(result_dir / artifact)
        model_counts[name] = len(model_points)
        components.append((f"fitted_{name}", model_points, MODEL_COLORS[name]))
        point_blocks.append(model_points)
        color_blocks.append(np.tile(np.asarray(MODEL_COLORS[name], dtype=np.uint8), (len(model_points), 1)))
        source_blocks.append(np.full(len(model_points), SOURCE_IDS[name], dtype=np.uint8))

    all_points = np.vstack(point_blocks)
    all_colors = np.vstack(color_blocks)
    all_sources = np.concatenate(source_blocks)
    comments = [
        "coordinates: UR base/world, meters",
        f"capture_id: {capture_dir.name}",
        "source_id 0: observed aligned depth, gray rgb=170,170,170",
        "source_id 1: fitted wide-mouth-bottle model, red rgb=255,60,60",
        "source_id 2: fitted beaker model, blue rgb=60,120,255",
        "source_id 3: fitted dropper model, magenta rgb=255,60,220",
        "depth projection uses server pipeline T_world_opencv_camera",
    ]
    write_colored_ply(output_path, all_points, all_colors, all_sources, comments)
    glb_path = output_path.with_suffix(".glb")
    write_colored_glb(glb_path, components)

    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps({
        "capture_id": capture_dir.name,
        "result_dir": str(result_dir),
        "output_ply": str(output_path),
        "output_glb": str(glb_path),
        "coordinate_frame": "world/base",
        "units": "meters",
        "depth": {
            "points": len(depth_points),
            "color_rgb": list(DEPTH_COLOR),
            "source_id": SOURCE_IDS["depth"],
            "stride": stride,
            "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m,
        },
        "models": {
            name: {
                "points": model_counts[name],
                "color_rgb": list(MODEL_COLORS[name]),
                "source_id": SOURCE_IDS[name],
            }
            for name in model_counts
        },
        "transform": {
            "source": "pipeline.log calibration_validation.T_world_opencv_camera",
            "T_world_opencv_camera": server_transform.tolist(),
            "local_transform_max_abs_difference": transform_difference,
        },
        "total_points": len(all_points),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return AlignmentExport(
        output_path=output_path,
        glb_path=glb_path,
        manifest_path=manifest_path,
        depth_points=len(depth_points),
        model_points=model_counts,
        transform_source="pipeline.log calibration_validation.T_world_opencv_camera",
        local_transform_max_abs_difference=transform_difference,
    )
