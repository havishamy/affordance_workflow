from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import validate_transform


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values.")
    return result


@dataclass(frozen=True)
class RobotConfig:
    ip: str
    speed: float
    acceleration: float
    payload_mass: float
    payload_cog: tuple[float, float, float]
    tcp_offset: tuple[float, ...]
    position_tolerance_m: float


@dataclass(frozen=True)
class GripperConfig:
    port: str
    init_feedback: bool
    force: int
    speed: int
    open_position: int
    grasp_position: int
    squeeze_position: int
    command_timeout_s: float


@dataclass(frozen=True)
class CameraConfig:
    serial: str
    width: int
    height: int
    fps: int
    warmup_frames: int
    timeout_ms: int


@dataclass(frozen=True)
class ServerConfig:
    url: str
    timeout_s: float
    retries: int
    verify_tls: bool
    bearer_token_env: str | None
    poll_seconds: float = 5.0
    job_timeout_s: float = 1800.0
    bundle_scope: str = "results"
    download_bundle: bool = True
    gpu: int | None = None
    ee_frame: str | None = None


@dataclass(frozen=True)
class MotionConfig:
    tool_orientation_rotvec: tuple[float, float, float]
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    transit_z_m: float
    grasp_offset_m: tuple[float, float, float]
    grasp_approach_m: float
    beaker_offset_m: tuple[float, float, float]
    dispense_height_m: float
    lift_after_titration_m: float
    settle_s: float
    grasp_settle_s: float
    squeeze_hold_s: float
    squeeze_cycles: int
    min_confidence: float


@dataclass(frozen=True)
class Calibration:
    parent_frame: str
    camera_frame: str
    transform_parent_camera: np.ndarray
    validated: bool

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Calibration file must contain a mapping.")
        parent = str(raw.get("parent_frame", "tcp"))
        if parent != "tcp":
            raise ValueError("Only parent_frame: tcp is supported in this workflow.")
        camera = str(raw.get("camera_frame", "wrist_camera_color_optical_frame"))
        matrix = validate_transform(raw["T_parent_camera"], "T_parent_camera")
        return cls(parent, camera, matrix, bool(raw.get("validated", False)))

    def as_json(self) -> dict[str, Any]:
        return {
            "convention": "T_parent_camera maps camera coordinates into parent coordinates",
            "parent_frame": self.parent_frame,
            "camera_frame": self.camera_frame,
            "T_parent_camera": self.transform_parent_camera.tolist(),
            "validated": self.validated,
        }


@dataclass(frozen=True)
class AppConfig:
    robot: RobotConfig
    gripper: GripperConfig
    camera: CameraConfig
    server: ServerConfig
    motion: MotionConfig
    calibration: Calibration
    output_dir: Path
    save_capture: bool

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Configuration file must contain a mapping.")

        robot = raw["robot"]
        gripper = raw["gripper"]
        camera = raw["camera"]
        server = raw["server"]
        motion = raw["motion"]
        data = raw.get("data", {})

        tcp_offset = tuple(float(v) for v in robot.get("tcp_offset", [0] * 6))
        if len(tcp_offset) != 6:
            raise ValueError("robot.tcp_offset must contain six values.")
        robot_cfg = RobotConfig(
            ip=str(robot["ip"]),
            speed=float(robot.get("speed", 0.2)),
            acceleration=float(robot.get("acceleration", 0.2)),
            payload_mass=float(robot.get("payload_mass", 1.601)),
            payload_cog=_triple(robot.get("payload_cog", [0.011, -0.002, 0.052]), "payload_cog"),
            tcp_offset=tcp_offset,
            position_tolerance_m=float(robot.get("position_tolerance_m", 0.005)),
        )
        gripper_cfg = GripperConfig(
            port=str(gripper["port"]),
            init_feedback=bool(gripper.get("init_feedback", True)),
            force=int(gripper.get("force", 20)),
            speed=int(gripper.get("speed", 100)),
            open_position=int(gripper.get("open_position", 1000)),
            grasp_position=int(gripper.get("grasp_position", 350)),
            squeeze_position=int(gripper.get("squeeze_position", 0)),
            command_timeout_s=float(gripper.get("command_timeout_s", 3.0)),
        )
        camera_cfg = CameraConfig(
            serial=str(camera["serial"]),
            width=int(camera.get("width", 640)),
            height=int(camera.get("height", 480)),
            fps=int(camera.get("fps", 15)),
            warmup_frames=int(camera.get("warmup_frames", 30)),
            timeout_ms=int(camera.get("timeout_ms", 5000)),
        )
        server_cfg = ServerConfig(
            url=str(server["url"]),
            timeout_s=float(server.get("timeout_s", 30.0)),
            retries=int(server.get("retries", 2)),
            verify_tls=bool(server.get("verify_tls", True)),
            bearer_token_env=server.get("bearer_token_env"),
            poll_seconds=float(server.get("poll_seconds", 5.0)),
            job_timeout_s=float(server.get("job_timeout_s", 1800.0)),
            bundle_scope=str(server.get("bundle_scope", "results")),
            download_bundle=bool(server.get("download_bundle", True)),
            gpu=None if server.get("gpu") is None else int(server["gpu"]),
            ee_frame=server.get("ee_frame"),
        )
        motion_cfg = MotionConfig(
            tool_orientation_rotvec=_triple(motion["tool_orientation_rotvec"], "tool_orientation_rotvec"),
            workspace_min=_triple(motion["workspace_min"], "workspace_min"),
            workspace_max=_triple(motion["workspace_max"], "workspace_max"),
            transit_z_m=float(motion["transit_z_m"]),
            grasp_offset_m=_triple(motion.get("grasp_offset_m", [0, 0, 0]), "grasp_offset_m"),
            grasp_approach_m=float(motion.get("grasp_approach_m", 0.10)),
            beaker_offset_m=_triple(motion.get("beaker_offset_m", [0, 0, 0]), "beaker_offset_m"),
            dispense_height_m=float(motion.get("dispense_height_m", 0.08)),
            lift_after_titration_m=float(motion.get("lift_after_titration_m", 0.10)),
            settle_s=float(motion.get("settle_s", 0.5)),
            grasp_settle_s=float(motion.get("grasp_settle_s", 1.0)),
            squeeze_hold_s=float(motion.get("squeeze_hold_s", 0.5)),
            squeeze_cycles=int(motion.get("squeeze_cycles", 1)),
            min_confidence=float(motion.get("min_confidence", 0.5)),
        )

        calibration_path = Path(raw["calibration_file"]).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = config_path.parent / calibration_path
        output_dir = Path(data.get("output_dir", "captures")).expanduser()
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir

        result = cls(
            robot_cfg,
            gripper_cfg,
            camera_cfg,
            server_cfg,
            motion_cfg,
            Calibration.load(calibration_path.resolve()),
            output_dir.resolve(),
            bool(data.get("save_capture", True)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.robot.ip or not self.camera.serial or not self.server.url:
            raise ValueError("Robot IP, camera serial and server URL are required.")
        if any(v <= 0 for v in (self.robot.speed, self.robot.acceleration, self.server.timeout_s)):
            raise ValueError("Motion speed, acceleration and server timeout must be positive.")
        if any(v < 0 or v > 1000 for v in (
            self.gripper.open_position,
            self.gripper.grasp_position,
            self.gripper.squeeze_position,
        )):
            raise ValueError("Gripper positions must be within [0, 1000].")
        if np.any(np.asarray(self.motion.workspace_min) >= np.asarray(self.motion.workspace_max)):
            raise ValueError("workspace_min must be smaller than workspace_max on every axis.")
        if not self.motion.workspace_min[2] <= self.motion.transit_z_m <= self.motion.workspace_max[2]:
            raise ValueError("transit_z_m must be inside the configured workspace.")
        if self.motion.squeeze_cycles < 1:
            raise ValueError("squeeze_cycles must be at least 1.")
        if not 0.0 <= self.motion.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        if any(not np.isfinite(v) or v <= 0 for v in (
            self.server.timeout_s, self.server.poll_seconds, self.server.job_timeout_s,
        )) or self.server.retries < 0:
            raise ValueError("Server timeouts/poll interval must be finite and positive; retries must be nonnegative.")
        if self.server.bundle_scope not in {"results", "all"}:
            raise ValueError("server.bundle_scope must be results or all.")
        if self.server.ee_frame not in {None, "tcp", "flange"}:
            raise ValueError("server.ee_frame must be tcp or flange, matching the server calibration.")
        if self.server.download_bundle and not self.save_capture:
            raise ValueError("Downloading the result bundle requires data.save_capture: true.")
