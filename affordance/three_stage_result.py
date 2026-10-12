from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import validate_transform
from .three_stage_config import ThreeStageConfig

LOG = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _points(value: Any, label: str, minimum: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1:] != (3,) or len(result) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} three-dimensional points.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains NaN or infinity.")
    return result


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Read the binary vertex-only PLY exported by this server, without modifying it."""
    type_map = {
        "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
        "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    }
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        if handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")
        count = None
        vertex_properties: list[tuple[str, str]] = []
        current_element = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            parts = line.decode("ascii").strip().split()
            if not parts:
                continue
            if parts[0] == "end_header":
                break
            if parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] not in type_map:
                    raise ValueError(f"Unsupported PLY vertex property in {path}: {parts}")
                vertex_properties.append((parts[2], type_map[parts[1]]))
        if count is None or count < 3 or not {"x", "y", "z"}.issubset(name for name, _ in vertex_properties):
            raise ValueError(f"PLY has invalid vertex metadata: {path}")
        data = np.fromfile(handle, dtype=np.dtype(vertex_properties), count=count)
        if len(data) != count:
            raise ValueError(f"Truncated PLY vertex data: {path}")
    xyz = np.column_stack((data["x"], data["y"], data["z"])).astype(float)
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"PLY contains non-finite vertices: {path}")
    return xyz


def _rms_radius(points: np.ndarray) -> float:
    median = np.median(points, axis=0)
    return float(np.sqrt(np.mean(np.sum((points - median) ** 2, axis=1))))


def _circle_from_partial_rim(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Fit XY circle to the returned rim arc; do not use its biased median as cup center."""
    xy = points[:, :2]
    origin = np.mean(xy, axis=0)
    centered = xy - origin
    design = np.column_stack((2.0 * centered, np.ones(len(centered))))
    rhs = np.sum(centered * centered, axis=1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        raise ValueError("Beaker mouth points do not span a circle arc.")
    center = coefficients[:2] + origin
    radius_sq = float(coefficients[2] + coefficients[:2] @ coefficients[:2])
    if radius_sq <= 0:
        raise ValueError("Beaker mouth circle radius is invalid.")
    radius = float(np.sqrt(radius_sq))
    distances = np.linalg.norm(xy - center, axis=1)
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)))
    angles = np.sort(np.mod(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    arc_span = float(2 * np.pi - np.max(gaps))
    return center, radius, residual, arc_span


@dataclass(frozen=True)
class SceneGeometry:
    stopper_grasp: np.ndarray
    bottle_body_grasp: np.ndarray
    bottle_axis_out: np.ndarray
    bottle_mouth_approx: np.ndarray
    beaker_mouth_center: np.ndarray
    beaker_inner_radius_estimate_m: float
    dropper_bulb_grasp: np.ndarray
    quality_report: dict[str, Any]
    source_job_id: str
    source_capture_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_job_id": self.source_job_id,
            "source_capture_id": self.source_capture_id,
            "stopper_grasp_base_m": self.stopper_grasp.tolist(),
            "bottle_body_grasp_base_m": self.bottle_body_grasp.tolist(),
            "bottle_axis_out_base": self.bottle_axis_out.tolist(),
            "bottle_mouth_approx_base_m": self.bottle_mouth_approx.tolist(),
            "beaker_mouth_center_base_m": self.beaker_mouth_center.tolist(),
            "beaker_inner_radius_estimate_m": self.beaker_inner_radius_estimate_m,
            "dropper_bulb_grasp_base_m": self.dropper_bulb_grasp.tolist(),
            "quality_report": self.quality_report,
            "method_note": "Beaker center is locally fitted to server mouth-affordance world arc; bottle mouth and tip dimensions remain unverified physical offsets.",
        }


def load_scene(config: ThreeStageConfig) -> SceneGeometry:
    root = config.result_dir
    capture = _read_json(config.capture_dir / "capture_info.json")
    job = _read_json(root / "job.json")
    result = _read_json(root / "result.json")
    capture_id = config.capture_dir.name
    if capture.get("capture_id") != capture_id or result.get("capture_id") != capture_id:
        raise ValueError("Capture/result IDs do not match the configured directory.")
    if capture.get("world_frame") != "base" or capture.get("ee_frame") != "flange":
        raise ValueError("Capture coordinate frames do not match the tool0 calibration convention.")
    if job.get("state") != "completed" or job.get("job_id") != result.get("job_id"):
        raise ValueError("Result job is not completed or job IDs disagree.")
    if result.get("frame") != "world" or result.get("units") != "meters":
        raise ValueError("Result must be in world/base frame and meters.")
    if result.get("model_transform_convention") != "p_world = T_world_object @ [scale * p_object, 1]":
        raise ValueError("Unrecognized model transform convention.")
    if result.get("calibration", {}).get("ee_frame") != "tool0":
        raise ValueError("Server calibration must report ee_frame=tool0.")
    if result.get("calibration", {}).get("world_frame") != "base":
        raise ValueError("Server calibration must report world_frame=base.")
    required_names = ("wide_mouth_bottle", "beaker", "dropper")
    objects = result.get("objects")
    if not isinstance(objects, dict) or any(name not in objects for name in required_names):
        raise ValueError("Result is missing a required object.")

    model_clouds: dict[str, np.ndarray] = {}
    model_quality: dict[str, Any] = {}
    warnings: list[str] = []
    for name in required_names:
        obj = objects[name]
        transform = validate_transform(obj.get("T_world_object"), f"{name}.T_world_object")
        scale = np.asarray(obj.get("scale"), dtype=float)
        if scale.shape == ():
            scale = np.repeat(scale, 3)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"{name} has invalid scale.")
        artifact = obj.get("model_points_world_artifact")
        if not isinstance(artifact, str) or artifact.startswith("/") or ".." in Path(artifact).parts:
            raise ValueError(f"{name} has an unsafe model artifact path.")
        model_clouds[name] = _read_ply_xyz(root / artifact)
        canonical_path = root / "models" / f"{name}_canonical.ply"
        world_mesh_path = root / "models" / f"{name}_world_mesh.ply"
        canonical = _read_ply_xyz(canonical_path)
        world_mesh = _read_ply_xyz(world_mesh_path)
        if canonical.shape != world_mesh.shape:
            raise ValueError(f"{name} canonical and world mesh vertex counts disagree.")
        reproduced = (transform[:3, :3] @ (canonical * scale).T).T + transform[:3, 3]
        reproduction_error = np.linalg.norm(reproduced - world_mesh, axis=1)
        maximum_error = float(np.max(reproduction_error))
        if maximum_error > 1e-4:
            raise ValueError(f"{name} pose/scale does not reproduce its world mesh ({maximum_error:.5f} m).")
        pose_path = root / "02_sam3d" / name / "pose_result.json"
        pose = _read_json(pose_path)
        iou = float(pose.get("silhouette_iou", float("nan")))
        if not np.isfinite(iou) or iou < config.quality.minimum_silhouette_iou:
            warnings.append(
                f"{name} silhouette IoU {iou:.3f} is below advisory value "
                f"{config.quality.minimum_silhouette_iou:.3f}; inspect its overlay."
            )
        model_quality[name] = {
            "silhouette_iou": iou,
            "model_vertices": len(model_clouds[name]),
            "canonical_to_world_mesh_max_error_m": maximum_error,
        }

    region_paths = {
        "stopper": ("wide_mouth_bottle", "stopper", config.quality.maximum_stopper_rms_m),
        "body_grasp": ("wide_mouth_bottle", "body_grasp", config.quality.maximum_body_rms_m),
        "beaker_mouth": ("beaker", "mouth", config.quality.maximum_beaker_mouth_rms_m),
        "bulb": ("dropper", "bulb", config.quality.maximum_bulb_rms_m),
    }
    regions: dict[str, np.ndarray] = {}
    region_quality: dict[str, Any] = {}
    for alias, (object_name, region_name, maximum_rms) in region_paths.items():
        region = objects[object_name].get("affordance_regions", {}).get(region_name)
        if not isinstance(region, dict):
            raise ValueError(f"Missing {object_name}.{region_name} affordance.")
        # At least three beaker rim points are mathematically needed to fit a
        # circle. Other small regions can still yield a center, with a warning.
        points = _points(region.get("points_world"), alias, 3 if alias == "beaker_mouth" else 1)
        if len(points) < config.quality.minimum_region_points:
            warnings.append(
                f"{alias} has {len(points)} points, below advisory support "
                f"{config.quality.minimum_region_points}; inspect the region overlay."
            )
        artifact = region.get("points_world_artifact")
        if not isinstance(artifact, str) or artifact.startswith("/") or ".." in Path(artifact).parts:
            raise ValueError(f"{alias} has an unsafe region artifact path.")
        if not (root / artifact).is_file():
            raise ValueError(f"Missing affordance PLY: {artifact}")
        metrics = _read_json(root / "01_affordance" / object_name / region_name / "quality_metrics.json")
        if metrics.get("valid") is not True:
            warnings.append(f"{alias} server affordance quality flag is not valid; inspect before any motion.")
        rms = _rms_radius(points)
        if rms > maximum_rms:
            warnings.append(
                f"{alias} region RMS radius {rms:.4f} m exceeds advisory value "
                f"{maximum_rms:.4f} m; inspect the region overlay."
            )
        regions[alias] = points
        region_quality[alias] = {
            "points": len(points), "rms_radius_m": rms,
            "argmax_uv": metrics.get("argmax_uv"), "target_peak": metrics.get("target_peak"),
        }

    separation_path = root / "01_affordance/wide_mouth_bottle/separation_metrics.json"
    separation = _read_json(separation_path) if separation_path.is_file() else {}
    if separation.get("valid") is not True or float(separation.get("top_region_iou", 1.0)) > 0.20:
        warnings.append("Stopper/body image affordances may overlap; inspect both regions before grasping.")
    stopper_region_center = np.median(regions["stopper"], axis=0)
    body_region_center = np.median(regions["body_grasp"], axis=0)
    separation_m = float(np.linalg.norm(stopper_region_center - body_region_center))
    if separation_m < config.quality.minimum_stopper_body_separation_m:
        warnings.append(
            f"Stopper/body centers are {separation_m:.4f} m apart, below advisory separation "
            f"{config.quality.minimum_stopper_body_separation_m:.4f} m."
        )

    bottle_model = model_clouds["wide_mouth_bottle"]
    top = bottle_model[bottle_model[:, 2] >= np.quantile(bottle_model[:, 2], 0.95)]
    bottle_xy = np.median(top[:, :2], axis=0)
    if np.linalg.norm(bottle_xy - stopper_region_center[:2]) > 0.025:
        warnings.append("Stopper region is over 25 mm from the reconstructed bottle-top axis.")
    stopper_grasp = np.array([*bottle_xy, stopper_region_center[2]], dtype=float)
    body_grasp = np.array([*bottle_xy, body_region_center[2]], dtype=float)
    axis = stopper_grasp - body_grasp
    axis_length = float(np.linalg.norm(axis))
    if axis_length < 1e-9:
        raise ValueError("Bottle body and stopper centers coincide; no extraction axis can be constructed.")
    axis /= axis_length
    if axis[2] < config.quality.minimum_bottle_axis_world_z:
        warnings.append("Bottle axis is less upright than the configured advisory value.")

    beaker_center_xy, radius, circle_residual, arc_span = _circle_from_partial_rim(regions["beaker_mouth"])
    quality = config.quality
    if not quality.minimum_beaker_radius_m <= radius <= quality.maximum_beaker_radius_m:
        warnings.append(f"Fitted beaker mouth radius {radius:.4f} m is outside advisory bounds.")
    if circle_residual > quality.maximum_beaker_circle_residual_m or arc_span < quality.minimum_beaker_arc_rad:
        warnings.append(
            f"Beaker mouth arc fit is weak (RMS {circle_residual:.4f} m, "
            f"span {arc_span:.2f} rad); inspect the rim overlay."
        )
    beaker_model = model_clouds["beaker"]
    model_xy = np.median(beaker_model[:, :2], axis=0)
    if np.linalg.norm(beaker_center_xy - model_xy) > 0.015:
        warnings.append("Fitted beaker mouth center is over 15 mm from the reconstructed beaker center.")
    mouth_z = float(np.quantile(regions["beaker_mouth"][:, 2], 0.98))
    beaker_mouth = np.array([*beaker_center_xy, mouth_z], dtype=float)

    bulb = np.median(regions["bulb"], axis=0)
    bottle_mouth = stopper_grasp + axis * config.bottle.stopper_center_to_mouth_m
    report = {
        "models": model_quality,
        "regions": region_quality,
        "stopper_body_center_distance_m": separation_m,
        "stopper_body_top_region_iou": separation.get("top_region_iou"),
        "beaker_circle_fit": {
            "radius_m": radius, "residual_rms_m": circle_residual,
            "arc_span_rad": arc_span,
            "mouth_z_from_region_98th_percentile_m": mouth_z,
            "raw_region_median_base_m": np.median(regions["beaker_mouth"], axis=0).tolist(),
        },
        "canonical_model_stability": "not guaranteed by server; no cross-capture canonical anchors used",
        "warnings": warnings,
    }
    for warning in warnings:
        LOG.warning("Scene quality advisory: %s", warning)
    return SceneGeometry(
        stopper_grasp=stopper_grasp,
        bottle_body_grasp=body_grasp,
        bottle_axis_out=axis,
        bottle_mouth_approx=bottle_mouth,
        beaker_mouth_center=beaker_mouth,
        beaker_inner_radius_estimate_m=radius,
        dropper_bulb_grasp=bulb,
        quality_report=report,
        source_job_id=str(result["job_id"]),
        source_capture_id=capture_id,
    )
