"""Independent color-gated raw-depth bulb center and guarded grasp test."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .camera import RealSenseWristCamera
from .geometry import pose_to_matrix, rotvec_to_matrix, validate_transform
from .hardware import DHGripper, URRobot
from .three_stage_config import ThreeStageConfig


@dataclass(frozen=True)
class BulbEstimate:
    surface_center_base_m: np.ndarray
    surface_center_camera_m: np.ndarray
    median_pixel_uv: tuple[float, float]
    color_pixels: int
    valid_depth_pixels: int
    depth_inlier_pixels: int
    depth_median_m: float
    world_bounds_m: tuple[np.ndarray, np.ndarray]
    mask: np.ndarray

    def as_dict(self) -> dict:
        return {
            "meaning": "median of the visible, color-gated, valid raw-depth bulb surface points; not a full-volume center",
            "surface_center_base_m": self.surface_center_base_m.tolist(),
            "surface_center_camera_m": self.surface_center_camera_m.tolist(),
            "median_pixel_uv": list(self.median_pixel_uv),
            "color_pixels": self.color_pixels,
            "valid_depth_pixels": self.valid_depth_pixels,
            "depth_inlier_pixels": self.depth_inlier_pixels,
            "depth_median_m": self.depth_median_m,
            "world_bounds_m": [self.world_bounds_m[0].tolist(), self.world_bounds_m[1].tolist()],
        }


def estimate_bulb_from_raw_depth(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    intrinsics: dict,
    depth_scale_m: float,
    t_base_camera: np.ndarray,
    roi_xyxy: Sequence[int],
    depth_min_m: float,
    depth_max_m: float,
) -> BulbEstimate:
    """Use RGB only to isolate rubber; coordinates come solely from raw depth."""
    if color_bgr.dtype != np.uint8 or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("Color must be uint8 BGR.")
    if depth_u16.dtype != np.uint16 or depth_u16.shape != color_bgr.shape[:2]:
        raise ValueError("Depth must be aligned uint16 with the RGB dimensions.")
    height, width = depth_u16.shape
    if int(intrinsics.get("width", -1)) != width or int(intrinsics.get("height", -1)) != height:
        raise ValueError("Camera intrinsics dimensions do not match RGB-D.")
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    if not np.all(np.isfinite([fx, fy, cx, cy, depth_scale_m, depth_min_m, depth_max_m])):
        raise ValueError("Intrinsics and depth limits must be finite.")
    if fx <= 0 or fy <= 0 or depth_scale_m <= 0 or not 0 < depth_min_m < depth_max_m:
        raise ValueError("Invalid focal length or depth range.")
    transform = validate_transform(t_base_camera, "T_base_camera")
    if len(roi_xyxy) != 4:
        raise ValueError("ROI must be [x0,y0,x1,y1].")
    x0, y0, x1, y1 = [int(value) for value in roi_xyxy]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI is outside the image.")

    crop = color_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bgr = crop.astype(np.int16)
    # The observed bulb is dim red/pink (OpenCV H ~4..10); background/glass
    # is mostly green/cyan (H ~75..85). These are task-specific color cues.
    red = (
        (hsv[:, :, 0] <= 20)
        & (hsv[:, :, 1] >= 35)
        & ((bgr[:, :, 2] - bgr[:, :, 1]) >= 12)
        & ((bgr[:, :, 2] - bgr[:, :, 0]) >= 15)
    ).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    if components <= 1:
        raise ValueError("No red/pink bulb component found in ROI; do not move the robot.")
    label = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    local_mask = labels == label
    color_pixels = int(np.count_nonzero(local_mask))
    if color_pixels < 50:
        raise ValueError(f"Bulb color component has only {color_pixels} pixels.")

    local_v, local_u = np.nonzero(local_mask)
    u = local_u + x0
    v = local_v + y0
    raw_z = depth_u16[v, u].astype(float) * depth_scale_m
    valid = (raw_z >= depth_min_m) & (raw_z <= depth_max_m)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 50:
        raise ValueError(f"Bulb has only {valid_count} valid non-background depth pixels.")
    u, v, raw_z = u[valid], v[valid], raw_z[valid]
    median_depth = float(np.median(raw_z))
    inlier = np.abs(raw_z - median_depth) <= 0.025
    inlier_count = int(np.count_nonzero(inlier))
    if inlier_count < 50:
        raise ValueError(f"Bulb has only {inlier_count} depth-continuous pixels.")
    u, v, raw_z = u[inlier], v[inlier], raw_z[inlier]

    camera_points = np.column_stack(((u - cx) / fx * raw_z, (v - cy) / fy * raw_z, raw_z))
    world_points = (transform[:3, :3] @ camera_points.T).T + transform[:3, 3]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[v, u] = 255
    return BulbEstimate(
        surface_center_base_m=np.median(world_points, axis=0),
        surface_center_camera_m=np.median(camera_points, axis=0),
        median_pixel_uv=(float(np.median(u)), float(np.median(v))),
        color_pixels=color_pixels,
        valid_depth_pixels=valid_count,
        depth_inlier_pixels=inlier_count,
        depth_median_m=float(np.median(raw_z)),
        world_bounds_m=(np.min(world_points, axis=0), np.max(world_points, axis=0)),
        mask=full_mask,
    )


def load_saved_capture(capture_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, float, np.ndarray]:
    color = cv2.imread(str(capture_dir / "rgb.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise ValueError(f"Missing readable rgb.png or depth.png in {capture_dir}.")
    camera = json.loads((capture_dir / "camera.json").read_text(encoding="utf-8"))
    t_base_flange = validate_transform(
        json.loads((capture_dir / "T_world_ee.json").read_text(encoding="utf-8"))["T_world_ee"],
        "T_base_flange",
    )
    hand_eye = validate_transform(
        json.loads((Path(__file__).resolve().parents[1] / "config/T_ee_cam.json").read_text(encoding="utf-8"))["T_ee_cam"],
        "T_flange_camera",
    )
    return color, depth, camera, float(camera["depth_scale_m_per_unit"]), t_base_flange @ hand_eye


def save_debug(capture_dir: Path, color: np.ndarray, estimate: BulbEstimate, target_xyz: np.ndarray, model_xyz=None) -> Path:
    output = capture_dir / "raw_depth_bulb_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    overlay = color.copy()
    tint = overlay.copy()
    tint[estimate.mask > 0] = (255, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.6, tint, 0.4, 0)
    cv2.circle(overlay, tuple(int(round(v)) for v in estimate.median_pixel_uv), 4, (0, 255, 0), -1)
    cv2.imwrite(str(output / "raw_depth_bulb_mask_overlay.png"), overlay)
    report = estimate.as_dict()
    report["grasp_target_base_m"] = target_xyz.tolist()
    if model_xyz is not None:
        model = np.asarray(model_xyz, dtype=float)
        report["server_model_affordance_center_base_m"] = model.tolist()
        report["raw_minus_model_center_mm"] = ((estimate.surface_center_base_m - model) * 1000).tolist()
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def capture_current_view(config: ThreeStageConfig, robot: URRobot, camera: RealSenseWristCamera) -> Path:
    """Capture from the current pose; no automatic movement or server request."""
    capture_dir = config.base.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_dir.mkdir(parents=True, exist_ok=False)
    camera.connect()
    try:
        time.sleep(config.observation.settle_s)
        before = robot.actual_tcp_pose()
        frame = camera.capture()
        after = robot.actual_tcp_pose()
        if np.linalg.norm(np.asarray(after[:3]) - np.asarray(before[:3])) > config.base.robot.position_tolerance_m:
            raise RuntimeError("Robot moved during depth diagnostic capture.")
        relative_rotation = rotvec_to_matrix(before[3:]).T @ rotvec_to_matrix(after[3:])
        angle = float(np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)))
        if angle > config.orientation_tolerance_rad:
            raise RuntimeError("Robot rotated during depth diagnostic capture.")
        t_base_flange = pose_to_matrix(after) @ np.linalg.inv(pose_to_matrix(config.base.robot.tcp_offset))
        from .server import build_upload_files

        for filename, content, _ in build_upload_files(frame, t_base_flange).values():
            (capture_dir / filename).write_bytes(content)
        (capture_dir / "capture_info.json").write_text(json.dumps({
            "capture_id": capture_dir.name,
            "actual_tcp_pose": after,
            "tcp_offset": list(config.base.robot.tcp_offset),
            "camera_serial": frame.camera_serial,
            "captured_at_unix_s": frame.captured_at_unix_s,
            "depth_scale_m_per_unit": frame.depth_scale_m,
            "world_frame": "base",
            "ee_frame": "flange",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return capture_dir
    finally:
        camera.disconnect()


def grasp_from_current_pose(
    config: ThreeStageConfig,
    robot: URRobot,
    gripper: DHGripper,
    target_xyz: np.ndarray,
    confirm: Callable[[str], None],
) -> None:
    """Approach and close on bulb; deliberately do not pull the pipette out."""
    target = np.asarray(target_xyz, dtype=float)
    orientation = list(config.dropper.orientation_rotvec)
    current = robot.actual_tcp_pose()
    transit_z = max(config.transit_z_m, current[2], target[2] + config.dropper.approach_m)
    waypoints = [
        [current[0], current[1], transit_z, *current[3:]],
        [target[0], target[1], transit_z, *orientation],
        [target[0], target[1], target[2] + config.dropper.approach_m, *orientation],
        [*target.tolist(), *orientation],
    ]
    lower = np.asarray(config.base.motion.workspace_min)
    upper = np.asarray(config.base.motion.workspace_max)
    for waypoint in [current, *waypoints]:
        xyz = np.asarray(waypoint[:3], dtype=float)
        if not np.all(np.isfinite(waypoint)) or not np.all((xyz >= lower) & (xyz <= upper)):
            raise ValueError(f"Raw-depth grasp waypoint is outside the configured workspace: {waypoint}")
    confirm("Check the actual bulb center, open jaws, free space above it and emergency stop before moving")
    gripper.connect()
    try:
        gripper.open()
        for waypoint in waypoints:
            robot.move_linear(waypoint)
        confirm("Check finger-to-glass clearance at grasp height before closing")
        gripper.set_position(config.dropper.hold_position)
        confirm("Check whether the bulb is physically held; this diagnostic will not extract it")
    finally:
        gripper.disconnect()
