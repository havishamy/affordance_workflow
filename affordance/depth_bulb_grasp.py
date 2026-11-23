"""Independent color-gated raw-depth bulb center and guarded grasp test."""

from __future__ import annotations

# Support the exact file path users naturally run, as well as package imports.
# The wrapper imports this file under its package name, so definitions are not
# duplicated and all CLI paths share the same implementation.
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from run_depth_bulb_grasp import main as _main

    raise SystemExit(_main())

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .camera import RealSenseWristCamera
from .geometry import pose_to_matrix, rotvec_to_matrix, validate_transform
from .hardware import DHGripper, URRobot
from .three_stage_config import ThreeStageConfig
from .three_stage_observation import observation_pose_for_current_tcp
from .target_geometry import GraspCenterConfig, weighted_modal_center


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
    center_estimation: dict = field(default_factory=dict)
    roi_px: tuple[int, int, int, int] | None = None
    points_base_m: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "meaning": "axial-mode local weighted center of visible color-gated raw-depth points; not a full-volume center",
            "center_estimation": self.center_estimation,
            "roi_px": list(self.roi_px) if self.roi_px is not None else None,
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
    roi_xyxy: Sequence[int] | None,
    depth_min_m: float,
    depth_max_m: float,
    *, center_config: GraspCenterConfig | None = None,
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
    if roi_xyxy is None:
        roi_xyxy = [0, 0, width, height]
    if len(roi_xyxy) != 4:
        raise ValueError("ROI must be [x0,y0,x1,y1].")
    x0, y0, x1, y1 = [int(value) for value in roi_xyxy]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI is outside the image.")

    crop = color_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bgr = crop.astype(np.int16)
    # The current bulb is dim red/pink with large neutral specular gaps.
    # A close operation deliberately reconnects those gaps inside the tight ROI.
    red = (
        ((hsv[:, :, 0] <= 20) | (hsv[:, :, 0] >= 160))
        & (hsv[:, :, 1] >= 15)
        & ((bgr[:, :, 2] - bgr[:, :, 1]) >= 5)
        & ((bgr[:, :, 2] - bgr[:, :, 0]) >= 8)
    ).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    if components <= 1:
        raise ValueError("No red/pink bulb component found in ROI; do not move the robot.")
    candidates = sorted(range(1, components), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    if len(candidates) > 1 and stats[candidates[1], cv2.CC_STAT_AREA] >= max(50, 0.6 * stats[candidates[0], cv2.CC_STAT_AREA]):
        raise ValueError("Multiple comparable red/pink components found; use --roi to isolate the intended bulb.")
    label = int(candidates[0])
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
    center, center_report = weighted_modal_center(world_points, config=center_config)
    center_camera = transform[:3, :3].T @ (center - transform[:3, 3])
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[v, u] = 255
    return BulbEstimate(
        surface_center_base_m=center,
        surface_center_camera_m=center_camera,
        median_pixel_uv=(float(np.median(u)), float(np.median(v))),
        color_pixels=color_pixels,
        valid_depth_pixels=valid_count,
        depth_inlier_pixels=inlier_count,
        depth_median_m=float(np.median(raw_z)),
        world_bounds_m=(np.min(world_points, axis=0), np.max(world_points, axis=0)),
        mask=full_mask,
        center_estimation=center_report,
        roi_px=(x0, y0, x1, y1),
        points_base_m=world_points,
    )


def latest_complete_depth_capture(root: Path) -> Path:
    """Select a complete saved RGB-D capture without loading any server result."""
    required = ("rgb.png", "depth.png", "camera.json", "T_world_ee.json", "capture_info.json")
    candidates = [p for p in root.iterdir() if p.is_dir() and all((p / name).is_file() for name in required)]
    if not candidates:
        raise ValueError(f"No complete RGB-D capture is available under {root}.")
    return max(candidates, key=lambda p: p.name)


def validate_depth_calibration(config: ThreeStageConfig) -> Path:
    """Catch inconsistent flange/TCP camera files before a calibration trial."""
    path = Path(__file__).resolve().parents[1] / "config/T_ee_cam.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("ee_frame") != "tool0" or raw.get("units") != "meters":
        raise ValueError("Raw-depth hand-eye file must declare ee_frame=tool0 and units=meters.")
    flange_camera = validate_transform(raw["T_ee_cam"], "T_tool0_camera")
    reproduced = pose_to_matrix(config.base.robot.tcp_offset) @ config.base.calibration.transform_parent_camera
    if not np.allclose(reproduced, flange_camera, atol=1e-6, rtol=0):
        raise ValueError("TCP offset + hand_eye.yaml disagree with T_ee_cam.json; synchronize calibration files before this trial.")
    return path


def load_saved_capture(capture_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, float, np.ndarray]:
    color = cv2.imread(str(capture_dir / "rgb.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(capture_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise ValueError(f"Missing readable rgb.png or depth.png in {capture_dir}.")
    camera = json.loads((capture_dir / "camera.json").read_text(encoding="utf-8"))
    info = json.loads((capture_dir / "capture_info.json").read_text(encoding="utf-8"))
    if info.get("world_frame") != "base" or info.get("ee_frame") not in ("flange", "tool0"):
        raise ValueError("Raw-depth capture must declare world_frame=base and ee_frame=flange/tool0.")
    t_base_flange = validate_transform(
        json.loads((capture_dir / "T_world_ee.json").read_text(encoding="utf-8"))["T_world_ee"],
        "T_base_flange",
    )
    hand_eye = validate_transform(
        json.loads((Path(__file__).resolve().parents[1] / "config/T_ee_cam.json").read_text(encoding="utf-8"))["T_ee_cam"],
        "T_flange_camera",
    )
    return color, depth, camera, float(camera["depth_scale_m_per_unit"]), t_base_flange @ hand_eye


def save_debug(
    capture_dir: Path, color: np.ndarray, estimate: BulbEstimate, target_xyz: np.ndarray,
    model_xyz=None, roi_xyxy: Sequence[int] | None = None,
    *, intrinsics: dict | None = None, t_base_camera: np.ndarray | None = None,
    calibration_path: Path | None = None, tcp_offset: Sequence[float] | None = None,
) -> Path:
    output = capture_dir / "raw_depth_bulb_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    overlay = color.copy()
    tint = overlay.copy()
    tint[estimate.mask > 0] = (255, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.6, tint, 0.4, 0)
    roi_xyxy = roi_xyxy if roi_xyxy is not None else estimate.roi_px
    if roi_xyxy is not None:
        x0, y0, x1, y1 = [int(value) for value in roi_xyxy]
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    center_pixel = estimate.median_pixel_uv
    if intrinsics is not None:
        point = estimate.surface_center_camera_m
        center_pixel = (point[0] / point[2] * intrinsics["fx"] + intrinsics["cx"], point[1] / point[2] * intrinsics["fy"] + intrinsics["cy"])
    cv2.circle(overlay, tuple(int(round(v)) for v in center_pixel), 4, (0, 255, 0), -1)
    cv2.imwrite(str(output / "raw_depth_bulb_mask_overlay.png"), overlay)
    report = estimate.as_dict()
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["capture_id"] = capture_dir.name
    report["grasp_target_base_m"] = target_xyz.tolist()
    report["center_pixel_uv"] = list(center_pixel)
    report["grasp_offset_base_m"] = (target_xyz - estimate.surface_center_base_m).tolist()
    report["coordinate_chain"] = "p_base = T_base_tool0 @ T_tool0_camera @ backproject(aligned_depth,u,v)"
    report["world_frame"] = "base"
    if t_base_camera is not None:
        report["T_base_opencv_camera"] = t_base_camera.tolist()
    if calibration_path is not None:
        report["hand_eye_path"] = str(calibration_path.resolve())
        report["hand_eye_sha256"] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        report["hand_eye"] = json.loads(calibration_path.read_text(encoding="utf-8"))
    if tcp_offset is not None:
        report["current_tcp_offset"] = list(tcp_offset)
    if estimate.points_base_m is not None:
        points = estimate.points_base_m.astype("<f4")
        ply = output / "raw_depth_bulb_points_world.ply"
        header = f"ply\nformat binary_little_endian 1.0\ncomment observed raw-depth bulb surface in UR base\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        ply.write_bytes(header.encode("ascii") + points.tobytes())
        report["observed_points_ply"] = str(ply)
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


def move_to_observation_pose(
    config: ThreeStageConfig,
    robot: URRobot,
    confirm: Callable[[str], None],
) -> dict[str, list[float] | list[list[float]]]:
    """Move through a guarded high transit to the configured fixed observation pose."""
    current = robot.actual_tcp_pose()
    target = observation_pose_for_current_tcp(config)
    high = max(current[2], target[2], config.transit_z_m)
    waypoints = [
        [*current[:2], high, *current[3:]],
        [*target[:2], high, *target[3:]],
        target,
    ]
    lower = np.asarray(config.base.motion.workspace_min)
    upper = np.asarray(config.base.motion.workspace_max)
    for index, pose in enumerate([current, *waypoints]):
        values = np.asarray(pose, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Observation waypoint {index} is not a finite six-value pose.")
        if not np.all((values[:3] >= lower) & (values[:3] <= upper)):
            raise ValueError(f"Observation waypoint {index} is outside workspace: {pose}")
    for index, pose in enumerate(waypoints):
        if not robot.has_inverse_kinematics(pose):
            raise RuntimeError(f"No IK solution for observation waypoint {index}: {pose}")
        if not robot.joints_within_safety_limits(robot.inverse_kinematics(pose)):
            raise RuntimeError(f"Joint safety limits reject observation waypoint {index}: {pose}")
    confirm(
        "Review the fixed-observation lift and horizontal path; gripper must be empty and emergency stop reachable"
    )
    for pose in waypoints:
        robot.move_linear(pose)
    return {"start": current, "target": target, "waypoints": waypoints}


def grasp_from_current_pose(
    config: ThreeStageConfig,
    robot: URRobot,
    gripper: DHGripper,
    target_xyz: np.ndarray,
    confirm: Callable[[str], None],
    *, record_dir: Path | None = None,
) -> None:
    """Approach and close on bulb; deliberately do not pull the pipette out."""
    target = np.asarray(target_xyz, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Raw-depth grasp target must contain three finite XYZ values.")
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
    for index, waypoint in enumerate(waypoints):
        if not robot.has_inverse_kinematics(waypoint):
            raise RuntimeError(f"No IK solution for raw-depth bulb waypoint {index}: {waypoint}")
        if not robot.joints_within_safety_limits(robot.inverse_kinematics(waypoint)):
            raise RuntimeError(f"Joint safety limits reject raw-depth bulb waypoint {index}: {waypoint}")
    if record_dir is not None:
        (record_dir / "grasp_plan.json").write_text(json.dumps({
            "start_tcp_pose": current, "target_tcp_pose": waypoints[-1], "waypoints": waypoints,
            "tcp_offset": list(config.base.robot.tcp_offset), "speed_mps": 0.02,
            "acceleration_mps2": 0.02, "extract_after_close": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    confirm("Check the actual bulb center, open jaws, free space above it and emergency stop before moving")
    gripper.connect()
    try:
        gripper.set_force(config.base.gripper.force)
        gripper.open()
        for waypoint in waypoints[:-1]:
            robot.move_linear(waypoint, speed=0.02, acceleration=0.02)
        confirm("At pregrasp: check finger-to-glass clearance before descending")
        robot.move_linear(waypoints[-1], speed=0.02, acceleration=0.02)
        confirm("At grasp height: confirm both jaws are centered on the rubber bulb before closing")
        gripper.set_position(config.dropper.hold_position)
        if record_dir is not None:
            actual = robot.actual_tcp_pose()
            (record_dir / "grasp_completed.json").write_text(json.dumps({
                "target_tcp_pose": waypoints[-1], "actual_tcp_pose": actual,
                "tcp_position_error_m": float(np.linalg.norm(np.asarray(actual[:3]) - target)),
                "tcp_orientation_error_rad": float(np.arccos(np.clip((np.trace(
                    rotvec_to_matrix(waypoints[-1][3:]).T @ rotvec_to_matrix(actual[3:])
                ) - 1) / 2, -1, 1))),
                "gripper_position": config.dropper.hold_position,
                "note": "Controller pose agreement does not measure visual/physical calibration error; inspect actual jaw contact.",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        gripper.disconnect()
