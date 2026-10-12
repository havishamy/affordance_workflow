from __future__ import annotations

import argparse
import json
import logging
import os
import time
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import requests

from .camera import RealSenseWristCamera
from .geometry import pose_to_matrix, rotvec_to_matrix
from .hardware import URRobot
from .server import build_upload_files
from .three_stage_config import ThreeStageConfig
from .three_stage_result import _read_ply_xyz, load_scene

LOG = logging.getLogger(__name__)


def _pose_from_transform(transform: np.ndarray) -> list[float]:
    rotvec, _ = cv2.Rodrigues(transform[:3, :3])
    return [*transform[:3, 3].tolist(), *rotvec.reshape(3).tolist()]


def observation_pose_for_current_tcp(config: ThreeStageConfig) -> list[float]:
    """Reproduce the captured flange pose even when the TCP offset changed."""
    reference = json.loads(config.observation.reference_capture_info.read_text(encoding="utf-8"))
    saved_pose = np.asarray(reference.get("actual_tcp_pose"), dtype=float)
    saved_offset = np.asarray(reference.get("tcp_offset"), dtype=float)
    if saved_pose.shape != (6,) or saved_offset.shape != (6,):
        raise ValueError("Observation reference requires six-value TCP pose and offset.")
    if not np.all(np.isfinite(saved_pose)) or not np.all(np.isfinite(saved_offset)):
        raise ValueError("Observation reference pose and offset must be finite.")
    t_base_capture_tcp = pose_to_matrix(saved_pose)
    t_flange_capture_tcp = pose_to_matrix(saved_offset)
    t_flange_current_tcp = pose_to_matrix(config.base.robot.tcp_offset)
    t_base_current_tcp = t_base_capture_tcp @ np.linalg.inv(t_flange_capture_tcp) @ t_flange_current_tcp
    return _pose_from_transform(t_base_current_tcp)


def _rotation_distance(first: Sequence[float], second: Sequence[float]) -> float:
    relative = rotvec_to_matrix(first[3:]).T @ rotvec_to_matrix(second[3:])
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def _safe_extract_bundle(bundle: Path, destination: Path) -> None:
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("Server returned an empty result ZIP.")
        for member in members:
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"Unsafe ZIP path: {member.filename!r}")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlink is not permitted: {member.filename!r}")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)


def reuse_latest_result(config: ThreeStageConfig) -> ThreeStageConfig:
    """Mode 0: local-only reuse, falling back to YAML paths if no pointer exists."""
    if not config.observation.reuse_latest_result:
        return config
    manifest = config.observation.latest_result_manifest
    if not manifest.is_file():
        return config
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read latest three-stage result manifest: {manifest}") from exc
    if not isinstance(record, dict):
        raise ValueError("Latest three-stage result manifest must contain an object.")
    if not isinstance(record.get("capture_dir"), str) or not isinstance(record.get("result_dir"), str):
        raise ValueError("Latest manifest requires capture_dir and result_dir strings.")
    capture_dir = Path(record.get("capture_dir", "")).expanduser().resolve()
    result_dir = Path(record.get("result_dir", "")).expanduser().resolve()
    if not capture_dir.is_dir() or not result_dir.is_dir():
        raise ValueError(f"Latest result referenced by {manifest} is missing; do not silently use stale data.")
    updated = replace(config, capture_dir=capture_dir, result_dir=result_dir)
    scene = load_scene(updated)
    if scene.source_job_id != record.get("job_id"):
        raise ValueError("Latest manifest job ID differs from its result.json.")
    LOG.info("Reusing latest validated three-stage result: %s", result_dir)
    return updated


class ThreeStageObservation:
    """Observe, upload and download once; never execute chemistry stages itself."""

    def __init__(
        self,
        config: ThreeStageConfig,
        robot: URRobot | None = None,
        camera: RealSenseWristCamera | None = None,
        submitter: Callable[[argparse.Namespace], int] | None = None,
        health_get: Callable[..., requests.Response] | None = None,
    ):
        self.config = config
        self.robot = robot or URRobot(config.base.robot)
        self.camera = camera or RealSenseWristCamera(config.base.camera)
        self.submitter = submitter
        self.health_get = health_get or requests.get
        self._transit_warning: str | None = None

    def _server_token(self) -> str:
        token_name = self.config.base.server.bearer_token_env
        if not token_name:
            raise ValueError("The complex-demo server requires server.bearer_token_env.")
        token = os.getenv(token_name)
        if not token:
            raise ValueError(f"Set {token_name} before enabling observation mode.")
        return token

    def _preflight_server(self, token: str) -> None:
        server = self.config.base.server
        with self.health_get(
            server.url.rstrip("/") + "/health",
            headers={"Authorization": f"Bearer {token}"},
            timeout=min(server.timeout_s, 8.0),
            verify=server.verify_tls,
        ) as response:
            response.raise_for_status()

    def _validate_observation_target(self, current: Sequence[float], target: Sequence[float]) -> list[list[float]]:
        lower = np.asarray(self.config.base.motion.workspace_min)
        upper = np.asarray(self.config.base.motion.workspace_max)
        highest_model = max(
            float(np.max(_read_ply_xyz(self.config.result_dir / "models" / f"{name}_world.ply")[:, 2]))
            for name in ("wide_mouth_bottle", "beaker", "dropper")
        )
        required_transit = highest_model + 0.080
        self._transit_warning = None
        if self.config.transit_z_m < required_transit:
            self._transit_warning = (
                f"Observation transit_z_m={self.config.transit_z_m:.3f} m is below model-top+80 mm "
                f"advisory ({required_transit:.3f} m); inspect the physical route."
            )
            LOG.warning(self._transit_warning)
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        if current.shape != (6,) or target.shape != (6,) or not np.all(np.isfinite(current)):
            raise ValueError("Observation start/target poses must be finite six-value poses.")
        high = max(current[2], target[2], self.config.transit_z_m)
        steps = [
            [*current[:2], high, *current[3:]],
            [*target[:2], high, *target[3:]],
            target.tolist(),
        ]
        for index, pose in enumerate([current.tolist(), *steps]):
            position = np.asarray(pose[:3])
            if not np.all((position >= lower) & (position <= upper)):
                raise ValueError(f"Observation waypoint {index} is outside workspace: {pose}.")
        return steps

    def _save_capture(self, directory: Path, frame, tcp_pose: list[float]) -> None:
        t_base_tcp = pose_to_matrix(tcp_pose)
        t_base_flange = t_base_tcp @ np.linalg.inv(pose_to_matrix(self.config.base.robot.tcp_offset))
        for filename, contents, _ in build_upload_files(frame, t_base_flange).values():
            (directory / filename).write_bytes(contents)
        record = {
            "capture_id": directory.name,
            "captured_at_unix_s": frame.captured_at_unix_s,
            "rgb_timestamp_ms": frame.rgb_timestamp_ms,
            "depth_timestamp_ms": frame.depth_timestamp_ms,
            "rgb_timestamp_domain": frame.rgb_timestamp_domain,
            "depth_timestamp_domain": frame.depth_timestamp_domain,
            "actual_tcp_pose": tcp_pose,
            "tcp_offset": list(self.config.base.robot.tcp_offset),
            "depth_scale_m_per_unit": frame.depth_scale_m,
            "camera_serial": frame.camera_serial,
            "world_frame": "base",
            "ee_frame": "flange",
            "rgb_frame": self.config.base.calibration.camera_frame,
        }
        (directory / "capture_info.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _submit(self, capture_dir: Path, output_dir: Path, token: str) -> None:
        submitter = self.submitter
        if submitter is None:
            from complex_capture_client import submit_and_wait

            submitter = submit_and_wait
        server = self.config.base.server
        args = argparse.Namespace(
            capture_dir=str(capture_dir),
            objects="wide_mouth_bottle,beaker,dropper",
            output_dir=str(output_dir),
            server=server.url,
            token=token,
            request_timeout=server.timeout_s,
            insecure=not server.verify_tls,
            gpu=server.gpu,
            scope="all",
            poll_seconds=server.poll_seconds,
            job_timeout=server.job_timeout_s,
            bundle_timeout=max(server.timeout_s, 600.0),
        )
        if submitter(args) != 0:
            raise RuntimeError("Complex-demo upload client did not finish successfully.")

    def run(self, confirm: Callable[[str, str], None]) -> ThreeStageConfig:
        if not self.config.physical_execution_validated:
            raise RuntimeError("Observation mode moves the robot; physical_execution_validated must be true.")
        if self.config.base.server.ee_frame != "flange":
            raise ValueError("Observation upload requires server.ee_frame=flange/tool0.")
        token = self._server_token()
        self._preflight_server(token)
        target = observation_pose_for_current_tcp(self.config)
        capture_dir = self.config.base.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_dir.mkdir(parents=True, exist_ok=False)
        robot_connected = False
        camera_connected = False
        try:
            self.robot.connect()
            robot_connected = True
            start = self.robot.actual_tcp_pose()
            waypoints = self._validate_observation_target(start, target)
            (capture_dir / "observation_plan.json").write_text(
                json.dumps({
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "reference_capture_info": str(self.config.observation.reference_capture_info),
                    "current_tcp_offset": list(self.config.base.robot.tcp_offset),
                    "start_tcp_pose": start,
                    "target_tcp_pose": target,
                    "waypoints": waypoints,
                    "warnings": [
                        "Cartesian workspace/object height proxy only; no arm, gripper, cable or obstacle collision model.",
                        *([self._transit_warning] if self._transit_warning else []),
                    ],
                }, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            confirm(
                "observe",
                f"Review {capture_dir / 'observation_plan.json'}; confirm gripper is empty, "
                "vertical lift and horizontal travel to the observation pose are clear"
                + ("\nWARNING: " + self._transit_warning if self._transit_warning else ""),
            )
            after_confirmation = self.robot.actual_tcp_pose()
            if np.linalg.norm(np.asarray(after_confirmation[:3]) - np.asarray(start[:3])) > self.config.base.robot.position_tolerance_m:
                raise RuntimeError("Robot moved during observation confirmation; replan required.")
            if _rotation_distance(start, after_confirmation) > self.config.orientation_tolerance_rad:
                raise RuntimeError("Robot rotated during observation confirmation; replan required.")
            for waypoint in waypoints:
                self.robot.move_linear(waypoint)
            actual = self.robot.actual_tcp_pose()
            if _rotation_distance(actual, target) > self.config.orientation_tolerance_rad:
                raise RuntimeError("Robot reached observation position but not the requested orientation.")
            time.sleep(self.config.observation.settle_s)
            self.camera.connect()
            camera_connected = True
            before = self.robot.actual_tcp_pose()
            frame = self.camera.capture()
            after = self.robot.actual_tcp_pose()
            if np.linalg.norm(np.asarray(before[:3]) - np.asarray(after[:3])) > self.config.base.robot.position_tolerance_m:
                raise RuntimeError("Robot moved during RGB-D capture.")
            if _rotation_distance(before, after) > self.config.orientation_tolerance_rad:
                raise RuntimeError("Robot rotated during RGB-D capture.")
            self._save_capture(capture_dir, frame, after)
            self.camera.disconnect()
            camera_connected = False
            self.robot.disconnect()
            robot_connected = False

            output_dir = capture_dir / "complex_server_result_with_beaker"
            self._submit(capture_dir, output_dir, token)
            bundles = list(output_dir.glob("complex_*_all.zip"))
            if len(bundles) != 1:
                raise RuntimeError(f"Expected one scope=all result ZIP, found {len(bundles)}.")
            result_dir = output_dir / bundles[0].stem
            _safe_extract_bundle(bundles[0], result_dir)
            # The saved API files are authoritative if the bundle omits copies.
            for name in ("job.json", "result.json"):
                bundled = result_dir / name
                if not bundled.is_file():
                    bundled.write_bytes((output_dir / name).read_bytes())
            updated = replace(self.config, capture_dir=capture_dir, result_dir=result_dir)
            scene = load_scene(updated)
            manifest = self.config.observation.latest_result_manifest
            manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
            temporary_manifest.write_text(json.dumps({
                "capture_dir": str(capture_dir),
                "result_dir": str(result_dir),
                "job_id": scene.source_job_id,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary_manifest.replace(manifest)
            LOG.info("Fresh complex-demo result validated: %s", result_dir)
            return updated
        except BaseException as exc:
            (capture_dir / "failure.json").write_text(
                json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if robot_connected:
                self.robot.stop()
            raise
        finally:
            if camera_connected:
                self.camera.disconnect()
            if robot_connected:
                self.robot.disconnect()
