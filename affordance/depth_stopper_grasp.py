from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from .camera import CapturedFrame, RealSenseWristCamera
from .geometry import pose_to_matrix, validate_transform
from .hardware import DHGripper, URRobot
from .three_stage_config import ThreeStageConfig
from .three_stage_observation import observation_pose_for_current_tcp
from .target_geometry import weighted_modal_center


@dataclass(frozen=True)
class DirectStopperConfig:
    source_path: Path
    three_stage: ThreeStageConfig
    roi_px: tuple[int, int, int, int]
    saturation_max: int
    value_min: int
    nearest_depth_percentile: float
    depth_below_percentile_m: float
    depth_above_percentile_m: float
    minimum_component_pixels: int
    depth_min_m: float
    depth_max_m: float
    grasp_offset_base_m: tuple[float, float, float]
    grasp_speed_mps: float
    grasp_acceleration_mps2: float
    settle_s: float

    @classmethod
    def load(cls, path: str | Path) -> "DirectStopperConfig":
        source = Path(path).expanduser().resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Direct stopper config must be a mapping.")
        three_stage_path = Path(str(raw["three_stage_config"])).expanduser()
        if not three_stage_path.is_absolute():
            three_stage_path = source.parent / three_stage_path
        segmentation = raw["segmentation"]
        motion = raw["motion"]
        roi = tuple(int(value) for value in raw["roi_px"])
        offset = tuple(float(value) for value in motion["grasp_offset_base_m"])
        if len(roi) != 4 or len(offset) != 3:
            raise ValueError("roi_px must have four values and grasp_offset_base_m three values.")
        result = cls(
            source_path=source,
            three_stage=ThreeStageConfig.load(three_stage_path),
            roi_px=roi,
            saturation_max=int(segmentation["saturation_max"]),
            value_min=int(segmentation["value_min"]),
            nearest_depth_percentile=float(segmentation["nearest_depth_percentile"]),
            depth_below_percentile_m=float(segmentation["depth_below_percentile_m"]),
            depth_above_percentile_m=float(segmentation["depth_above_percentile_m"]),
            minimum_component_pixels=int(segmentation["minimum_component_pixels"]),
            depth_min_m=float(segmentation["depth_min_m"]),
            depth_max_m=float(segmentation["depth_max_m"]),
            grasp_offset_base_m=offset,
            grasp_speed_mps=float(motion["grasp_speed_mps"]),
            grasp_acceleration_mps2=float(motion["grasp_acceleration_mps2"]),
            settle_s=float(motion["settle_s"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        x0, y0, x1, y1 = self.roi_px
        width, height = self.three_stage.base.camera.width, self.three_stage.base.camera.height
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("roi_px is outside the configured image.")
        if not 0 <= self.saturation_max <= 255 or not 0 <= self.value_min <= 255:
            raise ValueError("HSV limits must be within [0,255].")
        if not 0 <= self.nearest_depth_percentile <= 100:
            raise ValueError("nearest_depth_percentile must be within [0,100].")
        if self.depth_below_percentile_m < 0 or self.depth_above_percentile_m <= 0:
            raise ValueError("Depth cluster margins must be nonnegative/positive.")
        if self.minimum_component_pixels < 20:
            raise ValueError("minimum_component_pixels is too small.")
        if not 0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("Depth range is invalid.")
        if not all(np.isfinite(self.grasp_offset_base_m)):
            raise ValueError("grasp_offset_base_m must be finite.")
        if not 0 < self.grasp_speed_mps <= 0.05 or not 0 < self.grasp_acceleration_mps2 <= 0.05:
            raise ValueError("Direct grasp speed and acceleration must be within (0,0.05].")
        if self.settle_s < 0:
            raise ValueError("settle_s cannot be negative.")


@dataclass(frozen=True)
class StopperEstimate:
    surface_center_camera_m: np.ndarray
    surface_center_base_m: np.ndarray
    grasp_target_base_m: np.ndarray
    median_pixel_uv: tuple[float, float]
    median_depth_m: float
    white_depth_pixels: int
    selected_pixels: int
    mask: np.ndarray
    roi_px: tuple[int, int, int, int]
    center_estimation: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface_center_camera_m": self.surface_center_camera_m.tolist(),
            "surface_center_base_m": self.surface_center_base_m.tolist(),
            "grasp_target_base_m": self.grasp_target_base_m.tolist(),
            "median_pixel_uv": list(self.median_pixel_uv),
            "median_depth_m": self.median_depth_m,
            "white_depth_pixels": self.white_depth_pixels,
            "selected_pixels": self.selected_pixels,
            "roi_px": list(self.roi_px),
            "center_estimation": self.center_estimation,
        }


def estimate_stopper(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    intrinsics: dict[str, Any],
    depth_scale: float,
    t_base_camera: np.ndarray,
    config: DirectStopperConfig,
) -> StopperEstimate:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3 or color_bgr.dtype != np.uint8:
        raise ValueError("Color image must be uint8 BGR.")
    if depth_u16.shape != color_bgr.shape[:2] or depth_u16.dtype != np.uint16:
        raise ValueError("Depth must be aligned uint16 with the color image.")
    height, width = depth_u16.shape
    if int(intrinsics.get("width", -1)) != width or int(intrinsics.get("height", -1)) != height:
        raise ValueError("Camera intrinsics dimensions do not match the images.")
    fx, fy, cx, cy = (float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy"))
    if fx <= 0 or fy <= 0 or not all(np.isfinite(value) for value in (fx, fy, cx, cy, depth_scale)):
        raise ValueError("Camera intrinsics/depth scale are invalid.")

    x0, y0, x1, y1 = config.roi_px
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    depth_m = depth_u16.astype(np.float64) * depth_scale
    white = (hsv[:, :, 1] <= config.saturation_max) & (hsv[:, :, 2] >= config.value_min)
    valid_depth = (
        (depth_u16 > 0)
        & (depth_m >= config.depth_min_m)
        & (depth_m <= config.depth_max_m)
    )
    roi = np.zeros((height, width), dtype=bool)
    roi[y0:y1, x0:x1] = True
    candidates = white & valid_depth & roi
    candidate_depths = depth_m[candidates]
    if len(candidate_depths) < config.minimum_component_pixels:
        raise ValueError(
            f"Only {len(candidate_depths)} white valid-depth pixels in ROI; adjust roi_px/HSV/depth limits."
        )
    anchor = float(np.percentile(candidate_depths, config.nearest_depth_percentile))
    cluster = (
        candidates
        & (depth_m >= anchor - config.depth_below_percentile_m)
        & (depth_m <= anchor + config.depth_above_percentile_m)
    ).astype(np.uint8) * 255
    cluster = cv2.morphologyEx(cluster, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cluster = cv2.morphologyEx(cluster, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cluster)
    components = [
        index for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= config.minimum_component_pixels
    ]
    if not components:
        raise ValueError("No sufficiently large stopper component remains after depth clustering.")
    selected = max(components, key=lambda index: int(stats[index, cv2.CC_STAT_AREA]))
    mask = (labels == selected)
    v, u = np.where(mask)
    z = depth_m[v, u]
    camera_points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    world_points = camera_points @ t_base_camera[:3, :3].T + t_base_camera[:3, 3]
    surface_base, center_report = weighted_modal_center(world_points, config=config.three_stage.grasp_center)
    surface_camera = t_base_camera[:3, :3].T @ (surface_base - t_base_camera[:3, 3])
    grasp_target = surface_base + np.asarray(config.grasp_offset_base_m)
    return StopperEstimate(
        surface_center_camera_m=surface_camera,
        surface_center_base_m=surface_base,
        grasp_target_base_m=grasp_target,
        median_pixel_uv=(float(np.median(u)), float(np.median(v))),
        median_depth_m=float(np.median(z)),
        white_depth_pixels=len(candidate_depths),
        selected_pixels=len(z),
        mask=(mask.astype(np.uint8) * 255),
        roi_px=config.roi_px,
        center_estimation=center_report,
    )


def _rotation_distance(first_pose: list[float], second_pose: list[float]) -> float:
    first = pose_to_matrix(first_pose)[:3, :3]
    second = pose_to_matrix(second_pose)[:3, :3]
    relative = first.T @ second
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


class DirectDepthStopperWorkflow:
    def __init__(
        self,
        config: DirectStopperConfig,
        robot: URRobot | None = None,
        camera: RealSenseWristCamera | None = None,
        gripper: DHGripper | None = None,
    ):
        self.config = config
        self.stage = config.three_stage
        self.robot = robot or URRobot(self.stage.base.robot)
        self.camera = camera or RealSenseWristCamera(self.stage.base.camera)
        self.gripper = gripper or DHGripper(self.stage.base.gripper)

    def _validate_poses(self, poses: list[list[float]], label: str) -> None:
        lower = np.asarray(self.stage.base.motion.workspace_min)
        upper = np.asarray(self.stage.base.motion.workspace_max)
        for index, pose in enumerate(poses):
            values = np.asarray(pose, dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{label} waypoint {index} is not a finite six-value pose.")
            if not np.all((values[:3] >= lower) & (values[:3] <= upper)):
                raise ValueError(f"{label} waypoint {index} xyz {values[:3].tolist()} is outside workspace.")

    def _assert_ik(self, poses: list[list[float]], label: str) -> None:
        for index, pose in enumerate(poses):
            if not self.robot.has_inverse_kinematics(pose):
                raise RuntimeError(f"No IK solution for {label} waypoint {index}: {pose}")
            joints = self.robot.inverse_kinematics(pose)
            if not self.robot.joints_within_safety_limits(joints):
                raise RuntimeError(f"Joint safety limits reject {label} waypoint {index}: {pose}")

    def _observation_plan(self, current: list[float]) -> tuple[list[float], list[list[float]]]:
        target = observation_pose_for_current_tcp(self.stage)
        high = max(current[2], target[2], self.stage.transit_z_m)
        poses = [
            [*current[:2], high, *current[3:]],
            [*target[:2], high, *target[3:]],
            target,
        ]
        self._validate_poses([current, *poses], "observation")
        return target, poses

    def _grasp_plan(self, current: list[float], target_xyz: np.ndarray) -> dict[str, list[float]]:
        orientation = list(self.stage.stopper.orientation_rotvec)
        approach = self.stage.stopper.approach_m
        pull = self.stage.stopper.pull_distance_m
        transit = max(self.stage.transit_z_m, current[2], float(target_xyz[2] + approach), float(target_xyz[2] + pull))
        result = {
            "raise_to_transit": [current[0], current[1], transit, *current[3:]],
            "over_stopper": [float(target_xyz[0]), float(target_xyz[1]), transit, *orientation],
            "pregrasp": [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2] + approach), *orientation],
            "grasp": [*target_xyz.astype(float).tolist(), *orientation],
            "pull": [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2] + pull), *orientation],
        }
        self._validate_poses([current, *result.values()], "grasp")
        return result

    def _save_capture(
        self, directory: Path, frame: CapturedFrame, tcp_pose: list[float], t_world_ee: np.ndarray,
    ) -> None:
        cv2.imwrite(str(directory / "rgb.png"), frame.color_bgr)
        cv2.imwrite(str(directory / "depth.png"), frame.depth_u16)
        camera = {
            **frame.intrinsics,
            "depth_scale_m_per_unit": frame.depth_scale_m,
        }
        (directory / "camera.json").write_text(json.dumps(camera, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "T_world_ee.json").write_text(
            json.dumps({"T_world_ee": t_world_ee.tolist()}, indent=2), encoding="utf-8",
        )
        (directory / "capture_info.json").write_text(json.dumps({
            "capture_id": directory.name,
            "captured_at_unix_s": frame.captured_at_unix_s,
            "actual_tcp_pose": tcp_pose,
            "tcp_offset": list(self.stage.base.robot.tcp_offset),
            "camera_serial": frame.camera_serial,
            "world_frame": "base",
            "ee_frame": "flange",
            "rgb_frame": self.stage.base.calibration.camera_frame,
            "mode": "direct_raw_depth_stopper",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_diagnostic(self, directory: Path, frame: CapturedFrame, estimate: StopperEstimate) -> Path:
        overlay = frame.color_bgr.copy()
        tint = overlay.copy()
        tint[estimate.mask > 0] = (255, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
        x0, y0, x1, y1 = estimate.roi_px
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
        center = tuple(int(round(value)) for value in estimate.median_pixel_uv)
        cv2.drawMarker(overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        output = directory / "direct_stopper_mask_overlay.png"
        cv2.imwrite(str(output), overlay)
        (directory / "direct_stopper_result.json").write_text(
            json.dumps(estimate.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return output

    def run(self, execute: bool, confirm: Callable[[str, str], None]) -> dict[str, Any]:
        if not self.stage.physical_execution_validated or not self.stage.base.calibration.validated:
            raise RuntimeError("Physical execution and hand-eye calibration gates must be validated.")
        capture_dir = self.stage.base.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_dir.mkdir(parents=True, exist_ok=False)
        robot_connected = camera_connected = gripper_connected = False
        try:
            self.robot.connect()
            robot_connected = True
            current = self.robot.actual_tcp_pose()
            observation_target, observation_poses = self._observation_plan(current)
            self._assert_ik(observation_poses, "observation")
            (capture_dir / "observation_plan.json").write_text(json.dumps({
                "execute": execute,
                "target": observation_target,
                "waypoints": observation_poses,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            confirm(
                "observe stopper",
                f"Review {capture_dir / 'observation_plan.json'}; gripper must be empty and the lift/travel path clear",
            )
            for pose in observation_poses:
                self.robot.move_linear(pose)
            time.sleep(self.config.settle_s)

            self.camera.connect()
            camera_connected = True
            before = self.robot.actual_tcp_pose()
            frame = self.camera.capture()
            after = self.robot.actual_tcp_pose()
            if np.linalg.norm(np.asarray(before[:3]) - np.asarray(after[:3])) > self.stage.base.robot.position_tolerance_m:
                raise RuntimeError("Robot translated during direct-depth capture.")
            if _rotation_distance(before, after) > self.stage.orientation_tolerance_rad:
                raise RuntimeError("Robot rotated during direct-depth capture.")
            self.camera.disconnect()
            camera_connected = False

            t_base_tcp = pose_to_matrix(after)
            t_tool0_tcp = pose_to_matrix(self.stage.base.robot.tcp_offset)
            t_world_ee = validate_transform(t_base_tcp @ np.linalg.inv(t_tool0_tcp), "T_world_ee")
            t_ee_camera = validate_transform(
                json.loads((self.config.source_path.parent / "T_ee_cam.json").read_text(encoding="utf-8"))["T_ee_cam"],
                "T_ee_cam",
            )
            t_base_camera = validate_transform(t_world_ee @ t_ee_camera, "T_base_camera")
            self._save_capture(capture_dir, frame, after, t_world_ee)
            estimate = estimate_stopper(
                frame.color_bgr, frame.depth_u16,
                {**frame.intrinsics, "width": frame.depth_u16.shape[1], "height": frame.depth_u16.shape[0]},
                frame.depth_scale_m, t_base_camera, self.config,
            )
            overlay = self._save_diagnostic(capture_dir, frame, estimate)
            plan = self._grasp_plan(after, estimate.grasp_target_base_m)
            self._assert_ik(list(plan.values()), "grasp")
            (capture_dir / "direct_stopper_plan.json").write_text(json.dumps({
                "execute": execute,
                "estimate": estimate.as_dict(),
                "plan": plan,
                "gripper_position": self.stage.stopper.gripper_position,
                "gripper_force": self.stage.base.gripper.force,
                "warnings": [
                    "ROI/color/depth segmentation is scene-specific; inspect the overlay before motion.",
                    "No collision model or grasp-success feedback is available.",
                    "After a successful pull the script intentionally keeps holding the stopper.",
                ],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            if not execute:
                return {
                    "capture_dir": str(capture_dir),
                    "overlay": str(overlay),
                    "estimate": estimate.as_dict(),
                    "plan": plan,
                    "executed": False,
                }

            confirm(
                "grasp stopper",
                f"Inspect {overlay} and {capture_dir / 'direct_stopper_plan.json'}; continue only if the mask covers the stopper top",
            )
            self.gripper.connect()
            gripper_connected = True
            self.gripper.set_force(self.stage.base.gripper.force)
            self.gripper.open()
            for name in ("raise_to_transit", "over_stopper", "pregrasp"):
                self.robot.move_linear(
                    plan[name], speed=self.config.grasp_speed_mps,
                    acceleration=self.config.grasp_acceleration_mps2,
                )
            confirm("descend stopper", "Inspect jaw-to-bottle clearance at pregrasp before descending")
            self.robot.move_linear(
                plan["grasp"], speed=self.config.grasp_speed_mps,
                acceleration=self.config.grasp_acceleration_mps2,
            )
            self.gripper.set_position(self.stage.stopper.gripper_position)
            time.sleep(self.stage.stopper.settle_s)
            confirm("pull stopper", "Confirm the jaws are centered on the stopper before the vertical pull")
            self.robot.move_linear(
                plan["pull"], speed=self.config.grasp_speed_mps,
                acceleration=self.config.grasp_acceleration_mps2,
            )
            (capture_dir / "completed.json").write_text(json.dumps({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "final_tcp_pose": self.robot.actual_tcp_pose(),
                "stopper_remains_held": True,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "capture_dir": str(capture_dir),
                "overlay": str(overlay),
                "estimate": estimate.as_dict(),
                "plan": plan,
                "executed": True,
                "stopper_remains_held": True,
            }
        except BaseException as exc:
            if robot_connected:
                self.robot.stop()
            (capture_dir / "failure.json").write_text(json.dumps({
                "error_type": type(exc).__name__, "error": str(exc),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        finally:
            if camera_connected:
                self.camera.disconnect()
            if gripper_connected:
                self.gripper.disconnect()
            if robot_connected:
                self.robot.disconnect()

