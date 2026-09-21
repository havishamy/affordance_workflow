from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .camera import CapturedFrame, RealSenseWristCamera
from .config import AppConfig
from .geometry import pose_to_matrix, transform_point
from .hardware import DHGripper, URRobot
from .server import LocatedPoint, LocalizationClient, SceneLocations, build_upload_files

LOG = logging.getLogger(__name__)


class TitrationWorkflow:
    def __init__(
        self,
        config: AppConfig,
        robot: URRobot | None = None,
        gripper: DHGripper | None = None,
        camera: RealSenseWristCamera | None = None,
        client: LocalizationClient | None = None,
    ):
        self.config = config
        self.robot = robot or URRobot(config.robot)
        self.gripper = gripper or DHGripper(config.gripper)
        self.camera = camera or RealSenseWristCamera(config.camera)
        self.client = client or LocalizationClient(config.server)

    def run(
        self, execute: bool, capture_only: bool = False,
        before_execute: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if execute and capture_only:
            raise ValueError("capture_only and execute cannot be combined.")
        if self.config.server.ee_frame not in {"tcp", "flange"}:
            raise RuntimeError("Set server.ee_frame to tcp or flange after confirming ee in the server calibration.")
        if capture_only and not self.config.save_capture:
            raise ValueError("Capture-only mode requires data.save_capture: true.")
        if execute and not self.config.calibration.validated:
            raise RuntimeError(
                "Hand-eye calibration is not marked validated; verify config/hand_eye.yaml before real execution."
            )
        capture_dir = None
        try:
            self.robot.connect()
            self.camera.connect()
            time.sleep(self.config.motion.settle_s)
            pose_before = self.robot.actual_tcp_pose()
            pose_before_time = time.time()
            frame = self.camera.capture()
            tcp_pose = self.robot.actual_tcp_pose()
            pose_after_time = time.time()
            self._assert_same_pose(pose_before, tcp_pose, "during capture")
            t_world_ee = self._world_ee_matrix(tcp_pose)
            capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            capture_info = {
                "capture_id": capture_id,
                "captured_at_unix_s": frame.captured_at_unix_s,
                "rgb_timestamp_ms": frame.rgb_timestamp_ms,
                "depth_timestamp_ms": frame.depth_timestamp_ms,
                "rgb_timestamp_domain": frame.rgb_timestamp_domain,
                "depth_timestamp_domain": frame.depth_timestamp_domain,
                "pose_before_read_at_unix_s": pose_before_time,
                "pose_after_read_at_unix_s": pose_after_time,
                "timestamp_note": "SDK frame times have their own domains; pose times are host read times. Stationary capture, not hardware synchronization.",
                "actual_tcp_pose": tcp_pose,
                "tcp_offset": list(self.config.robot.tcp_offset),
                "depth_scale_m_per_unit": frame.depth_scale_m,
                "camera_serial": frame.camera_serial,
                "world_frame": "base",
                "ee_frame": self.config.server.ee_frame,
                "rgb_frame": self.config.calibration.camera_frame,
            }
            if self.config.save_capture:
                capture_dir = self._save_capture(frame, t_world_ee, capture_info)
                LOG.info("Capture saved before upload: %s", capture_dir)
            if capture_only:
                return {**capture_info, "capture_dir": str(capture_dir), "execute": False}
            locations = self.client.locate(
                frame, t_world_ee, capture_id,
                output_dir=capture_dir / "server_result" if capture_dir else None,
            )
            base_points = self._locations_in_base(locations, tcp_pose)
            plan = self.build_plan(base_points["rubber_bulb"], base_points["beaker_mouth"], tcp_pose)
            record = self._make_record(frame, tcp_pose, locations, base_points, plan, execute)
            record["T_world_ee"] = t_world_ee.tolist()
            record["ee_frame"] = self.config.server.ee_frame
            if capture_dir is not None:
                record["capture_dir"] = str(capture_dir)
                self._write_json(capture_dir / "metadata.json", record)
            self._log_plan(plan, base_points, execute)
            if execute:
                if before_execute is not None:
                    before_execute()
                self._assert_same_pose(tcp_pose, self.robot.actual_tcp_pose(), "while waiting for localization/confirmation")
                self.gripper.connect()
                self.execute_plan(plan)
            return record
        except BaseException as exc:
            if capture_dir is not None:
                try:
                    self._write_json(capture_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
                except OSError:
                    LOG.exception("Failed to save error details.")
            self.robot.stop()
            raise
        finally:
            for name, cleanup in (
                ("camera", self.camera.disconnect),
                ("gripper", self.gripper.disconnect),
                ("robot", self.robot.disconnect),
            ):
                try:
                    cleanup()
                except Exception as exc:
                    LOG.warning("Failed to disconnect %s cleanly: %s", name, exc)

    def run_saved(
        self,
        capture_dir: str | Path,
        execute: bool,
        before_execute: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Rebuild a plan from saved base-frame targets without camera/server work."""
        if execute and not self.config.calibration.validated:
            raise RuntimeError(
                "Hand-eye calibration is not marked validated; verify config/hand_eye.yaml before real execution."
            )
        source_dir = Path(capture_dir).expanduser().resolve()
        locations, source_info = self._load_saved_locations(source_dir)
        saved_offset = np.asarray(source_info.get("tcp_offset"), dtype=float)
        current_offset = np.asarray(self.config.robot.tcp_offset, dtype=float)
        if saved_offset.shape != (6,) or not np.all(np.isfinite(saved_offset)):
            raise ValueError("Saved capture does not contain a valid six-value TCP offset.")
        offset_changed = not np.allclose(saved_offset, current_offset, atol=1e-9)
        if offset_changed:
            LOG.warning(
                "TCP offset changed since localization: saved=%s, current=%s. "
                "The saved base-frame targets remain unchanged; rebuilding every waypoint for the current TCP.",
                saved_offset.tolist(), current_offset.tolist(),
            )

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = source_dir / "reuse_runs" / run_stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            self.robot.connect()
            current_tcp_pose = self.robot.actual_tcp_pose()
            base_points = self._locations_in_base(locations, current_tcp_pose)
            plan = self.build_plan(
                base_points["rubber_bulb"], base_points["beaker_mouth"], current_tcp_pose,
            )
            record = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mode": "EXECUTE_SAVED" if execute else "PLAN_SAVED",
                "source_capture_dir": str(source_dir),
                "source_capture_id": source_info.get("capture_id", source_dir.name),
                "source_job_id": locations.raw_response.get("job_id"),
                "scene_reuse_requires_unchanged_objects": True,
                "current_tcp_pose": current_tcp_pose,
                "source_tcp_offset": saved_offset.tolist(),
                "current_tcp_offset": current_offset.tolist(),
                "tcp_offset_changed_since_capture": offset_changed,
                "positions_base_m": {key: value.tolist() for key, value in base_points.items()},
                "motion_plan": [{"name": name, "pose": pose} for name, pose in plan],
                "execute": execute,
            }
            self._write_json(run_dir / "plan.json", record)
            self._log_plan(plan, base_points, execute)
            if execute:
                if before_execute is not None:
                    before_execute()
                self._assert_same_pose(
                    current_tcp_pose,
                    self.robot.actual_tcp_pose(),
                    "while confirming saved-result execution",
                )
                self.gripper.connect()
                self.execute_plan(plan)
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                record["status"] = "completed"
                self._write_json(run_dir / "execution.json", record)
            return {**record, "capture_dir": str(source_dir), "reuse_run_dir": str(run_dir)}
        except BaseException as exc:
            try:
                self._write_json(
                    run_dir / "failure.json",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
            except OSError:
                LOG.exception("Failed to save saved-result execution error details.")
            self.robot.stop()
            raise
        finally:
            for name, cleanup in (("gripper", self.gripper.disconnect), ("robot", self.robot.disconnect)):
                try:
                    cleanup()
                except Exception as exc:
                    LOG.warning("Failed to disconnect %s cleanly: %s", name, exc)

    @staticmethod
    def _load_json_object(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        return payload

    def _load_saved_locations(self, source_dir: str | Path) -> tuple[SceneLocations, dict[str, Any]]:
        source_dir = Path(source_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise ValueError(f"Saved capture directory does not exist: {source_dir}")
        source_info = self._load_json_object(source_dir / "capture_info.json")
        if source_info.get("world_frame") != "base":
            raise ValueError("Saved capture must declare world_frame=base.")
        job = self._load_json_object(source_dir / "server_result" / "job.json")
        if job.get("state") != "completed":
            raise ValueError(f"Saved server job is not completed: {job.get('state')!r}.")
        result = self._load_json_object(source_dir / "server_result" / "result.json")
        if job.get("job_id") != result.get("job_id"):
            raise ValueError("Saved job.json and result.json have different job IDs.")
        locations = LocalizationClient._parse_world_response(result)
        metadata_path = source_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = self._load_json_object(metadata_path)
            positions = metadata.get("positions_base_m", {})
            for key, point in (
                ("rubber_bulb", locations.rubber_bulb.position),
                ("beaker_mouth", locations.beaker_mouth.position),
            ):
                saved = np.asarray(positions.get(key), dtype=float)
                if saved.shape != (3,) or not np.allclose(saved, point, atol=1e-9):
                    raise ValueError(f"Saved metadata and result disagree for {key}.")
        return locations, source_info

    def _world_ee_matrix(self, tcp_pose: list[float]) -> np.ndarray:
        t_base_tcp = pose_to_matrix(tcp_pose)
        if self.config.server.ee_frame == "tcp":
            return t_base_tcp
        if self.config.server.ee_frame == "flange":
            return t_base_tcp @ np.linalg.inv(pose_to_matrix(self.config.robot.tcp_offset))
        raise ValueError("Unknown server ee_frame.")

    def _assert_same_pose(self, before: list[float], after: list[float], stage: str) -> None:
        first, second = pose_to_matrix(before), pose_to_matrix(after)
        distance = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        relative = first[:3, :3].T @ second[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        if distance > self.config.robot.position_tolerance_m or angle > 0.02:
            raise RuntimeError(f"Robot pose changed {stage} ({distance:.4f} m, {angle:.4f} rad); recapture required.")

    def _point_in_base(self, point: LocatedPoint, tcp_pose: list[float]) -> np.ndarray:
        normalized = point.frame.lower()
        calibration = self.config.calibration
        if normalized in {"base", "robot_base", "base_link"}:
            return point.position.copy()
        if normalized == "tcp":
            return transform_point(pose_to_matrix(tcp_pose), point.position)
        if normalized in {
            "camera",
            "color_camera",
            "wrist_camera",
            calibration.camera_frame.lower(),
        }:
            t_base_tcp = pose_to_matrix(tcp_pose)
            t_base_camera = t_base_tcp @ calibration.transform_parent_camera
            return transform_point(t_base_camera, point.position)
        raise ValueError(f"Unsupported coordinate frame returned by server: {point.frame!r}.")

    def _locations_in_base(self, result: SceneLocations, tcp_pose: list[float]) -> dict[str, np.ndarray]:
        threshold = self.config.motion.min_confidence
        for name, point in (("rubber_bulb", result.rubber_bulb), ("beaker_mouth", result.beaker_mouth)):
            if point.confidence is None:
                LOG.info("Server supplies no confidence for %s; inspect the downloaded overlays before execution.", name)
            elif point.confidence < threshold:
                raise RuntimeError(
                    f"Server confidence for {name} is {point.confidence:.3f}, below {threshold:.3f}; refusing motion."
                )
        return {
            "rubber_bulb": self._point_in_base(result.rubber_bulb, tcp_pose),
            "beaker_mouth": self._point_in_base(result.beaker_mouth, tcp_pose),
        }

    def _pose(self, xyz: np.ndarray) -> list[float]:
        return [*np.asarray(xyz, dtype=float).tolist(), *self.config.motion.tool_orientation_rotvec]

    def _validate_xyz(self, xyz: np.ndarray, label: str) -> None:
        lower = np.asarray(self.config.motion.workspace_min)
        upper = np.asarray(self.config.motion.workspace_max)
        if not np.all((xyz >= lower) & (xyz <= upper)):
            raise RuntimeError(
                f"Waypoint {label}={xyz.tolist()} is outside workspace {lower.tolist()}..{upper.tolist()}."
            )

    def build_plan(
        self,
        rubber_bulb_base: np.ndarray,
        beaker_mouth_base: np.ndarray,
        current_tcp_pose: list[float],
    ) -> list[tuple[str, list[float]]]:
        motion = self.config.motion
        grasp = np.asarray(rubber_bulb_base, dtype=float) + np.asarray(motion.grasp_offset_m)
        dispense = np.asarray(beaker_mouth_base, dtype=float) + np.asarray(motion.beaker_offset_m)
        dispense[2] += motion.dispense_height_m

        current_safe = np.asarray(current_tcp_pose[:3], dtype=float)
        current_safe[2] = max(current_safe[2], motion.transit_z_m)
        above_grasp = grasp.copy()
        above_grasp[2] = max(grasp[2] + motion.grasp_approach_m, motion.transit_z_m)
        lift_grasp = grasp.copy()
        lift_grasp[2] = max(grasp[2] + motion.grasp_approach_m, motion.transit_z_m)
        above_beaker = dispense.copy()
        above_beaker[2] = max(dispense[2], motion.transit_z_m)
        retreat = dispense.copy()
        retreat[2] = max(dispense[2] + motion.lift_after_titration_m, motion.transit_z_m)

        named_xyz = [
            ("raise_to_transit", current_safe),
            ("above_rubber_bulb", above_grasp),
            ("grasp_rubber_bulb", grasp),
            ("lift_rubber_bulb", lift_grasp),
            ("above_beaker", above_beaker),
            ("dispense", dispense),
            ("retreat", retreat),
        ]
        for label, xyz in named_xyz:
            self._validate_xyz(xyz, label)
        return [(label, self._pose(xyz)) for label, xyz in named_xyz]

    def execute_plan(self, plan: list[tuple[str, list[float]]]) -> None:
        waypoints = dict(plan)
        self.gripper.open()
        for label in ("raise_to_transit", "above_rubber_bulb", "grasp_rubber_bulb"):
            LOG.info("Moving to %s", label)
            self.robot.move_linear(waypoints[label])
        self.gripper.grasp()
        time.sleep(self.config.motion.grasp_settle_s)
        for label in ("lift_rubber_bulb", "above_beaker", "dispense"):
            LOG.info("Moving to %s", label)
            self.robot.move_linear(waypoints[label])
        for cycle in range(self.config.motion.squeeze_cycles):
            LOG.info("Titration squeeze %d/%d", cycle + 1, self.config.motion.squeeze_cycles)
            self.gripper.squeeze()
            time.sleep(self.config.motion.squeeze_hold_s)
            self.gripper.grasp()
            time.sleep(self.config.motion.squeeze_hold_s)
        self.robot.move_linear(waypoints["retreat"])

    @staticmethod
    def _log_plan(plan: list[tuple[str, list[float]]], points: dict[str, np.ndarray], execute: bool) -> None:
        LOG.info("Rubber bulb in base: %s", np.round(points["rubber_bulb"], 5).tolist())
        LOG.info("Beaker mouth in base: %s", np.round(points["beaker_mouth"], 5).tolist())
        LOG.info("Mode: %s", "EXECUTE" if execute else "PLAN ONLY (no arm/gripper motion)")
        for name, pose in plan:
            LOG.info("  %-20s %s", name, [round(v, 5) for v in pose])

    def _make_record(
        self,
        frame: CapturedFrame,
        tcp_pose: list[float],
        locations: SceneLocations,
        base_points: dict[str, np.ndarray],
        plan: list[tuple[str, list[float]]],
        execute: bool,
    ) -> dict[str, Any]:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execute": execute,
            "camera_serial": frame.camera_serial,
            "captured_at_unix_s": frame.captured_at_unix_s,
            "depth_scale_m_per_unit": frame.depth_scale_m,
            "intrinsics": frame.intrinsics,
            "actual_tcp_pose": tcp_pose,
            "hand_eye": self.config.calibration.as_json(),
            "server_response": locations.raw_response,
            "positions_base_m": {key: value.tolist() for key, value in base_points.items()},
            "motion_plan": [{"name": name, "pose": pose} for name, pose in plan],
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_capture(self, frame: CapturedFrame, t_world_ee: np.ndarray, record: dict[str, Any]) -> Path:
        files = build_upload_files(frame, t_world_ee)
        directory = self.config.output_dir / record["capture_id"]
        directory.mkdir(parents=True, exist_ok=False)
        for name, content, _ in files.values():
            (directory / name).write_bytes(content)
        self._write_json(directory / "capture_info.json", record)
        return directory
