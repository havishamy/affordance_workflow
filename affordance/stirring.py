"""Glass-rod perception client, guarded planner and fourth-stage executor."""

from __future__ import annotations

import json
import math
import os
import re
import time
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import requests

from .camera import RealSenseWristCamera
from .geometry import rotvec_to_matrix, validate_transform
from .hardware import DHGripper, URRobot
from .three_stage_config import ThreeStageConfig
from .three_stage_observation import ThreeStageObservation, observation_pose_for_current_tcp
from .three_stage_result import SceneGeometry, _read_ply_xyz


TERMINAL_STATES = {"completed", "failed", "interrupted"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite values.")
    return vector


def _safe_artifact(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty artifact path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} contains an unsafe artifact path.")
    return value


def _rotation_distance(first: Sequence[float], second: Sequence[float]) -> float:
    relative = rotvec_to_matrix(first[3:]).T @ rotvec_to_matrix(second[3:])
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass(frozen=True)
class RodGraspCandidate:
    id: str
    grasp_point_world: np.ndarray
    opposite_tip_id: str
    opposite_tip_world: np.ndarray
    direction_grasp_to_tip_world: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "grasp_point_world": self.grasp_point_world.tolist(),
            "opposite_tip_id": self.opposite_tip_id,
            "opposite_tip_world": self.opposite_tip_world.tolist(),
            "rod_direction_grasp_to_tip_world": self.direction_grasp_to_tip_world.tolist(),
        }


@dataclass(frozen=True)
class StirringGeometry:
    job_id: str
    capture_id: str
    endpoint_a_world: np.ndarray
    endpoint_b_world: np.ndarray
    center_world: np.ndarray
    axis_world_unoriented: np.ndarray
    length_m: float
    endpoint_uncertainty_m: float
    rod_radius_m: float
    candidates: tuple[RodGraspCandidate, RodGraspCandidate]
    quality: dict[str, Any]
    geometry_overlay_artifact: str
    artifact_paths: tuple[str, ...]
    raw_result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "capture_id": self.capture_id,
            "endpoint_a_world": self.endpoint_a_world.tolist(),
            "endpoint_b_world": self.endpoint_b_world.tolist(),
            "center_world": self.center_world.tolist(),
            "axis_world_unoriented": self.axis_world_unoriented.tolist(),
            "length_m": self.length_m,
            "endpoint_uncertainty_m": self.endpoint_uncertainty_m,
            "rod_radius_m": self.rod_radius_m,
            "grasp_candidates": [candidate.as_dict() for candidate in self.candidates],
            "quality": self.quality,
            "geometry_overlay_artifact": self.geometry_overlay_artifact,
            "artifact_paths": list(self.artifact_paths),
        }


def validate_stirring_result(
    result: dict[str, Any], config: ThreeStageConfig, expected_capture_id: str,
    expected_t_world_camera: np.ndarray,
) -> StirringGeometry:
    """Validate the complete stirring-geometry-v1 contract before planning."""
    if result.get("schema_version") != "stirring-geometry-v1":
        raise ValueError("Stirring result schema_version must be stirring-geometry-v1.")
    if result.get("capture_id") != expected_capture_id:
        raise ValueError("Stirring result capture_id does not match the uploaded capture.")
    if result.get("frame") != "world" or result.get("units") != "meters":
        raise ValueError("Stirring result must use world/base frame and meters.")
    assumptions = result.get("assumptions")
    if not isinstance(assumptions, dict) or assumptions.get("glass_rod_lies_on_table") is not True:
        raise ValueError("Stirring result does not assert that the rod lies on the table.")
    if assumptions.get("endpoint_a_b_have_no_task_semantics") is not True:
        raise ValueError("Stirring result endpoint semantics are incompatible.")
    world_up = _vector(result.get("world_up"), "world_up")
    world_down = _vector(result.get("world_down"), "world_down")
    if not np.allclose(world_up, [0, 0, 1], atol=1e-6) or not np.allclose(world_down, [0, 0, -1], atol=1e-6):
        raise ValueError("Stirring result world up/down axes are incompatible with UR base.")
    calibration = result.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Stirring result is missing calibration details.")
    reported_camera = validate_transform(
        calibration.get("T_world_opencv_camera"), "calibration.T_world_opencv_camera",
    )
    expected_camera = validate_transform(expected_t_world_camera, "expected_T_world_opencv_camera")
    if not np.allclose(reported_camera, expected_camera, atol=1e-6):
        translation_error = float(np.linalg.norm(reported_camera[:3, 3] - expected_camera[:3, 3]))
        raise ValueError(
            f"Server/local hand-eye composition differs (camera translation error {translation_error:.6f} m)."
        )

    rod = result.get("glass_rod")
    if not isinstance(rod, dict) or rod.get("class_id") != 12:
        raise ValueError("Stirring result is missing YOLO class 12 glass_rod.")
    quality = rod.get("quality")
    if not isinstance(quality, dict) or quality.get("valid") is not True:
        raise ValueError("Server marked glass-rod geometry invalid.")
    perception = config.stirring_perception
    metrics = {
        "segmentation_confidence": float(rod.get("segmentation_confidence", float("nan"))),
        "mask_elongation": float(quality.get("mask_elongation", float("nan"))),
        "table_plane_inlier_fraction": float(quality.get("table_plane_inlier_fraction", float("nan"))),
        "table_plane_rmse_m": float(quality.get("table_plane_rmse_m", float("nan"))),
        "endpoint_uncertainty_m": float(quality.get("endpoint_uncertainty_m", float("nan"))),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Stirring result quality metrics contain missing or non-finite values.")
    gates = (
        (metrics["segmentation_confidence"] >= perception.min_segmentation_confidence, "segmentation confidence"),
        (metrics["mask_elongation"] >= perception.min_mask_elongation, "mask elongation"),
        (
            metrics["table_plane_inlier_fraction"] >= perception.min_table_plane_inlier_fraction,
            "table-plane inlier fraction",
        ),
        (metrics["table_plane_rmse_m"] <= perception.max_table_plane_rmse_m, "table-plane RMSE"),
        (
            metrics["endpoint_uncertainty_m"] <= perception.max_endpoint_uncertainty_m,
            "endpoint uncertainty",
        ),
    )
    failed = [label for passed, label in gates if not passed]
    if failed:
        raise ValueError(f"Stirring perception quality gates failed: {failed}")

    endpoint_a = _vector(rod.get("endpoint_a_world"), "glass_rod.endpoint_a_world")
    endpoint_b = _vector(rod.get("endpoint_b_world"), "glass_rod.endpoint_b_world")
    center = _vector(rod.get("center_world"), "glass_rod.center_world")
    axis = _vector(rod.get("axis_world_unoriented"), "glass_rod.axis_world_unoriented")
    axis_norm = float(np.linalg.norm(axis))
    if not math.isclose(axis_norm, 1.0, abs_tol=0.02):
        raise ValueError("Glass-rod axis is not unit length.")
    axis /= axis_norm
    length_m = float(rod.get("length_m", float("nan")))
    if not math.isfinite(length_m) or not perception.min_length_m <= length_m <= perception.max_length_m:
        raise ValueError("Glass-rod length is outside request bounds.")
    if abs(length_m - perception.expected_rod_length_m) > perception.rod_length_tolerance_m:
        raise ValueError("Glass-rod length differs from the measured expected length.")
    endpoint_distance = float(np.linalg.norm(endpoint_b - endpoint_a))
    if abs(endpoint_distance - length_m) > max(0.01, 2 * metrics["endpoint_uncertainty_m"]):
        raise ValueError("Glass-rod endpoints disagree with reported length.")
    if np.linalg.norm(center - (endpoint_a + endpoint_b) / 2) > max(0.01, 2 * metrics["endpoint_uncertainty_m"]):
        raise ValueError("Glass-rod center disagrees with its endpoints.")

    raw_candidates = rod.get("grasp_candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
        raise ValueError("Stirring result must provide exactly two grasp candidates.")
    candidates: list[RodGraspCandidate] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ValueError("Each glass-rod grasp candidate must be an object.")
        candidate_id = raw.get("id")
        tip_id = raw.get("opposite_tip_id")
        if not isinstance(candidate_id, str) or not candidate_id or tip_id not in {"endpoint_a", "endpoint_b"}:
            raise ValueError("Glass-rod candidate IDs are invalid.")
        grasp = _vector(raw.get("grasp_point_world"), f"candidate[{index}].grasp_point_world")
        tip = _vector(raw.get("opposite_tip_world"), f"candidate[{index}].opposite_tip_world")
        direction = _vector(
            raw.get("rod_direction_grasp_to_tip_world"),
            f"candidate[{index}].rod_direction_grasp_to_tip_world",
        )
        actual = tip - grasp
        actual_length = float(np.linalg.norm(actual))
        if actual_length <= 0.03:
            raise ValueError("Glass-rod grasp-to-tip length is implausibly short.")
        actual /= actual_length
        if not math.isclose(float(np.linalg.norm(direction)), 1.0, abs_tol=0.02):
            raise ValueError("Glass-rod candidate direction is not unit length.")
        direction /= np.linalg.norm(direction)
        if float(direction @ actual) < 0.995:
            raise ValueError("Glass-rod candidate direction disagrees with grasp and tip points.")
        expected_tip = endpoint_a if tip_id == "endpoint_a" else endpoint_b
        if np.linalg.norm(tip - expected_tip) > max(0.01, 2 * metrics["endpoint_uncertainty_m"]):
            raise ValueError("Glass-rod candidate opposite tip disagrees with the named endpoint.")
        candidates.append(RodGraspCandidate(candidate_id, grasp, tip_id, tip, direction))
    if len({candidate.id for candidate in candidates}) != 2 or len({candidate.opposite_tip_id for candidate in candidates}) != 2:
        raise ValueError("Glass-rod grasp candidates are not the two symmetric alternatives.")

    artifacts = (
        _safe_artifact(rod.get("mask_artifact"), "mask_artifact"),
        _safe_artifact(rod.get("mask_probability_artifact"), "mask_probability_artifact"),
        _safe_artifact(rod.get("segmentation_overlay_artifact"), "segmentation_overlay_artifact"),
        _safe_artifact(rod.get("geometry_overlay_artifact"), "geometry_overlay_artifact"),
    )
    overlay = artifacts[-1]
    rod_radius = float(assumptions.get("centerline_is_offset_from_table_by_rod_radius_m", float("nan")))
    if not math.isfinite(rod_radius) or abs(rod_radius - perception.rod_radius_m) > 1e-6:
        raise ValueError("Server rod radius differs from the local measured request value.")
    job_id = result.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Stirring result has an invalid job_id.")
    return StirringGeometry(
        job_id=job_id,
        capture_id=expected_capture_id,
        endpoint_a_world=endpoint_a,
        endpoint_b_world=endpoint_b,
        center_world=center,
        axis_world_unoriented=axis,
        length_m=length_m,
        endpoint_uncertainty_m=metrics["endpoint_uncertainty_m"],
        rod_radius_m=rod_radius,
        candidates=(candidates[0], candidates[1]),
        quality={**quality, "segmentation_confidence": metrics["segmentation_confidence"]},
        geometry_overlay_artifact=overlay,
        artifact_paths=artifacts,
        raw_result=result,
    )


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Server returned non-JSON HTTP {response.status_code}: {response.text[:500]}") from exc
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(value, ensure_ascii=False)}")
    if not isinstance(value, dict):
        raise ValueError("Server response must be a JSON object.")
    return value


def _get_with_retries(url: str, *, headers: dict[str, str], config: ThreeStageConfig, **kwargs) -> requests.Response:
    last: Exception | None = None
    for attempt in range(config.base.server.retries + 1):
        try:
            response = requests.get(
                url, headers=headers, timeout=config.base.server.timeout_s,
                verify=config.base.server.verify_tls, **kwargs,
            )
            if response.status_code < 500 or attempt == config.base.server.retries:
                return response
        except requests.RequestException as exc:
            last = exc
            if attempt == config.base.server.retries:
                raise
        time.sleep(min(config.base.server.poll_seconds, 1.0) * (attempt + 1))
    assert last is not None
    raise last


def _safe_extract(bundle: Path, destination: Path) -> None:
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("Stirring result ZIP is empty.")
        for member in members:
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"Unsafe stirring ZIP member: {member.filename!r}")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"Stirring ZIP symlink is not permitted: {member.filename!r}")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)


def submit_stirring(
    config: ThreeStageConfig, capture_dir: Path, output_dir: Path, token: str,
) -> tuple[Path, StirringGeometry]:
    """Submit once, poll safely, download the complete debug bundle and validate it."""
    from complex_capture_client import validate_capture

    paths, depth_scale, capture_id = validate_capture(capture_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    jobs_url = config.base.server.url.rstrip("/") + "/v1/stirring/jobs"
    perception = config.stirring_perception
    data = {
        "capture_id": capture_id,
        "depth_scale": repr(depth_scale),
        "rod_radius_m": repr(perception.rod_radius_m),
        "grasp_inset_m": repr(perception.grasp_inset_m),
        "min_length_m": repr(perception.min_length_m),
        "max_length_m": repr(perception.max_length_m),
    }
    print(f"Validated stirring capture {capture_id}; uploading to {jobs_url}", flush=True)
    try:
        with ExitStack() as stack:
            files = {
                "rgb": ("rgb.png", stack.enter_context(paths["rgb"].open("rb")), "image/png"),
                "depth": ("depth.png", stack.enter_context(paths["depth"].open("rb")), "image/png"),
                "camera": ("camera.json", stack.enter_context(paths["camera"].open("rb")), "application/json"),
                "world_ee": (
                    "T_world_ee.json", stack.enter_context(paths["world_ee"].open("rb")), "application/json"
                ),
            }
            response = requests.post(
                jobs_url, headers=headers, files=files, data=data,
                timeout=config.base.server.timeout_s, verify=config.base.server.verify_tls,
            )
        job = _response_json(response)
    except requests.RequestException as exc:
        raise RuntimeError("Stirring submission failed ambiguously; inspect the server queue before retrying.") from exc
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Stirring server returned an invalid job_id.")
    _write_json(output_dir / "job.json", job)
    print(f"Submitted stirring job: {job_id}", flush=True)
    deadline = time.monotonic() + config.base.server.job_timeout_s
    previous_state = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Stirring job {job_id} did not finish before the local deadline.")
        status = _response_json(_get_with_retries(f"{jobs_url}/{job_id}", headers=headers, config=config))
        status = {**status, "job_id": job_id}
        _write_json(output_dir / "job.json", status)
        state = status.get("state")
        if state != previous_state:
            print(f"stirring job {job_id}: {state}", flush=True)
            previous_state = state
        if state in TERMINAL_STATES:
            if state != "completed":
                _write_json(output_dir / "failure.json", status)
                raise RuntimeError(f"Stirring job {job_id} ended as {state}.")
            break
        if state not in {"uploading", "queued", "running"}:
            raise ValueError(f"Unknown stirring server state: {state!r}")
        time.sleep(min(config.base.server.poll_seconds, remaining))
    result = _response_json(_get_with_retries(f"{jobs_url}/{job_id}/result", headers=headers, config=config))
    _write_json(output_dir / "result.json", result)
    geometry = validate_stirring_result(result, config, capture_id, _expected_world_camera(capture_dir))

    bundle = output_dir / f"stirring_{job_id}_all.zip"
    partial = bundle.with_suffix(".zip.part")
    with _get_with_retries(
        f"{jobs_url}/{job_id}/bundle", headers=headers, config=config, stream=True,
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    partial.replace(bundle)
    result_dir = output_dir / bundle.stem
    _safe_extract(bundle, result_dir)
    for name in ("job.json", "result.json"):
        target = result_dir / name
        if not target.is_file():
            target.write_bytes((output_dir / name).read_bytes())
    missing = [artifact for artifact in geometry.artifact_paths if not (result_dir / artifact).is_file()]
    if missing:
        raise ValueError(f"Stirring bundle is missing declared artifacts: {missing}")
    print(f"Stirring result: {result_dir}", flush=True)
    return result_dir, geometry


def load_stirring_result(config: ThreeStageConfig, capture_dir: Path, result_dir: Path) -> StirringGeometry:
    result = _read_json(result_dir / "result.json")
    geometry = validate_stirring_result(
        result, config, capture_dir.name, _expected_world_camera(capture_dir),
    )
    missing = [artifact for artifact in geometry.artifact_paths if not (result_dir / artifact).is_file()]
    if missing:
        raise ValueError(f"Saved stirring result is missing declared artifacts: {missing}")
    return geometry


def _expected_world_camera(capture_dir: Path) -> np.ndarray:
    t_world_ee = validate_transform(
        _read_json(capture_dir / "T_world_ee.json").get("T_world_ee"), "T_world_ee",
    )
    hand_eye_path = Path(__file__).resolve().parents[1] / "config" / "T_ee_cam.json"
    t_ee_camera = validate_transform(_read_json(hand_eye_path).get("T_ee_cam"), "T_ee_cam")
    return t_world_ee @ t_ee_camera


@dataclass(frozen=True)
class StirringObservationResult:
    capture_dir: Path
    result_dir: Path
    geometry: StirringGeometry


class StirringObservation:
    """Move to the final observation pose, capture, submit and require overlay review."""

    def __init__(
        self, config: ThreeStageConfig, robot: URRobot | None = None,
        camera: RealSenseWristCamera | None = None,
    ):
        self.config = config
        self.robot = robot or URRobot(config.base.robot)
        self.camera = camera or RealSenseWristCamera(config.base.camera)

    def run(
        self, scene: SceneGeometry, confirm: Callable[[str, str], None],
    ) -> StirringObservationResult:
        if not self.config.physical_execution_validated:
            raise RuntimeError("Final stirring observation moves the robot; physical_execution_validated must be true.")
        if self.config.base.server.ee_frame != "flange":
            raise ValueError("Stirring observation upload requires server.ee_frame=flange/tool0.")
        token_name = self.config.base.server.bearer_token_env
        token = os.getenv(token_name or "")
        if not token_name or not token:
            raise ValueError(f"Set {token_name or 'the configured bearer token'} before stirring observation.")
        helper = ThreeStageObservation(self.config, robot=self.robot, camera=self.camera)
        helper._preflight_server(token)
        target = observation_pose_for_current_tcp(self.config)
        capture_dir = self.config.base.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_dir.mkdir(parents=True, exist_ok=False)
        output_dir = capture_dir / "stirring_server_result"
        robot_connected = False
        camera_connected = False
        try:
            self.robot.connect()
            robot_connected = True
            start = self.robot.actual_tcp_pose()
            waypoints = helper._validate_observation_target(start, target)
            _write_json(capture_dir / "stirring_observation_plan.json", {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "start_tcp_pose": start,
                "target_tcp_pose": target,
                "waypoints": waypoints,
                "beaker_source_job_id": scene.source_job_id,
                "warning": "Cartesian fence only; no complete arm/camera/cable collision model.",
            })
            print(
                "Stirring observation will move now: dropper must be returned/released, gripper empty, "
                "glass rod flat and fully visible, beaker unmoved, and the route clear. "
                "The explicit --stir-observe 1 option authorizes this observation move.",
                flush=True,
            )
            for waypoint in waypoints:
                self.robot.move_linear(waypoint)
            time.sleep(self.config.observation.settle_s)
            reached = self.robot.actual_tcp_pose()
            if (
                np.linalg.norm(np.asarray(reached[:3]) - np.asarray(target[:3]))
                > self.config.base.robot.position_tolerance_m
                or _rotation_distance(reached, target) > self.config.orientation_tolerance_rad
            ):
                raise RuntimeError("Robot did not reach the requested final observation pose.")
            self.camera.connect()
            camera_connected = True
            before = self.robot.actual_tcp_pose()
            frame = self.camera.capture()
            after = self.robot.actual_tcp_pose()
            if (
                np.linalg.norm(np.asarray(after[:3]) - np.asarray(before[:3]))
                > self.config.base.robot.position_tolerance_m
                or _rotation_distance(before, after) > self.config.orientation_tolerance_rad
            ):
                raise RuntimeError("Robot moved during final stirring RGB-D capture.")
            helper._save_capture(capture_dir, frame, after)
            self.camera.disconnect()
            camera_connected = False
            self.robot.disconnect()
            robot_connected = False
            result_dir, geometry = submit_stirring(self.config, capture_dir, output_dir, token)
            overlay = result_dir / geometry.geometry_overlay_artifact
            print(
                f"Stirring geometry overlay (non-blocking): {overlay}. "
                "Inspect endpoints, centerline and grasp candidates before accepting either of the two motion checkpoints.",
                flush=True,
            )
            manifest = self.config.stirring_perception.latest_result_manifest
            manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = manifest.with_suffix(manifest.suffix + ".tmp")
            _write_json(temporary, {
                "capture_dir": str(capture_dir),
                "result_dir": str(result_dir),
                "job_id": geometry.job_id,
                "beaker_source_job_id": scene.source_job_id,
                "beaker_source_capture_id": scene.source_capture_id,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            })
            temporary.replace(manifest)
            return StirringObservationResult(capture_dir, result_dir, geometry)
        except BaseException as exc:
            job_record = _read_json(output_dir / "job.json") if (output_dir / "job.json").is_file() else None
            result_record = _read_json(output_dir / "result.json") if (output_dir / "result.json").is_file() else None
            _write_json(capture_dir / "failure.json", {
                "capture_id": capture_dir.name,
                "job_id": job_record.get("job_id") if job_record else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "motion_parameters": self.config.stirring_motion.__dict__,
                "server_result": result_record,
            })
            if robot_connected:
                self.robot.stop()
            raise
        finally:
            if camera_connected:
                self.camera.disconnect()
            if robot_connected:
                self.robot.disconnect()


def reuse_latest_stirring(config: ThreeStageConfig, scene: SceneGeometry) -> StirringObservationResult:
    manifest = config.stirring_perception.latest_result_manifest
    record = _read_json(manifest)
    if record.get("beaker_source_job_id") != scene.source_job_id:
        raise ValueError("Saved stirring geometry belongs to a different beaker/complex-demo result.")
    capture_dir = Path(str(record.get("capture_dir", ""))).expanduser().resolve()
    result_dir = Path(str(record.get("result_dir", ""))).expanduser().resolve()
    if not capture_dir.is_dir() or not result_dir.is_dir():
        raise ValueError("Saved stirring capture or result directory is missing.")
    geometry = load_stirring_result(config, capture_dir, result_dir)
    if geometry.job_id != record.get("job_id"):
        raise ValueError("Saved stirring manifest job_id does not match result.json.")
    return StirringObservationResult(capture_dir, result_dir, geometry)


@dataclass(frozen=True)
class StirringStep:
    kind: str
    name: str
    pose: tuple[float, ...] | None = None
    gripper_position: int | None = None
    gripper_force: int | None = None
    speed_mps: float | None = None
    seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "name": self.name}
        for key in ("pose", "gripper_position", "gripper_force", "speed_mps", "seconds"):
            value = getattr(self, key)
            if value is not None:
                result[key] = list(value) if key == "pose" else value
        return result


@dataclass(frozen=True)
class StirringPlan:
    candidate: RodGraspCandidate
    steps: tuple[StirringStep, ...]
    workspace_margin_m: float
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": "stirring",
            "selected_candidate": self.candidate.as_dict(),
            "workspace_margin_m": self.workspace_margin_m,
            "steps": [step.as_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }


def _rotvec(rotation: np.ndarray) -> np.ndarray:
    value, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    return value.reshape(3)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    value, _ = cv2.Rodrigues(axis * angle)
    return value


def _rotation_mapping(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.clip(first @ second, -1.0, 1.0))
    angle = math.acos(dot)
    axis = np.cross(first, second)
    if np.linalg.norm(axis) < 1e-9:
        if dot > 0:
            axis = np.array([1.0, 0.0, 0.0])
            return np.eye(3), axis, 0.0
        basis = np.array([1.0, 0.0, 0.0]) if abs(first[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(first, basis)
    axis /= np.linalg.norm(axis)
    return _axis_angle(axis, angle), axis, angle


class StirringWorkflow:
    """Plan and execute glass-rod grasp, vertical alignment and circular stirring."""

    def __init__(
        self, config: ThreeStageConfig, scene: SceneGeometry, geometry: StirringGeometry,
        result_dir: Path, robot: URRobot | None = None, gripper: DHGripper | None = None,
    ):
        self.config = config
        self.scene = scene
        self.geometry = geometry
        self.result_dir = result_dir
        self.robot = robot or URRobot(config.base.robot)
        self.gripper = gripper or DHGripper(config.base.gripper)

    def _measured_motion(self) -> dict[str, float]:
        motion = self.config.stirring_motion
        if not motion.validated:
            raise RuntimeError(
                "stirring_motion.validated is false; measure every stirring dimension before planning motion."
            )
        names = (
            "pregrasp_height_m", "lift_height_m", "preinsert_clearance_m",
            "target_depth_below_mouth_m", "stir_radius_m", "stir_speed_mps",
            "beaker_safe_inner_radius_m", "bottom_clearance_m",
        )
        values = {name: getattr(motion, name) for name in names}
        if any(value is None for value in values.values()) or motion.rod_grasp_center_tool_m is None:
            raise RuntimeError("Stirring motion has unmeasured null parameters.")
        return {name: float(value) for name, value in values.items()}

    def _grasp_rotation(self, direction: np.ndarray) -> np.ndarray:
        rod_world = np.asarray(direction, dtype=float)
        rod_world[2] = 0.0
        if np.linalg.norm(rod_world) < 0.95:
            raise ValueError("Glass rod is not sufficiently horizontal for top-down grasp planning.")
        rod_world /= np.linalg.norm(rod_world)
        approach_world = np.array([0.0, 0.0, -1.0])
        closing_world = np.cross(rod_world, approach_world)
        closing_world /= np.linalg.norm(closing_world)
        closing_tool = np.asarray(self.config.stirring_motion.closing_axis_tool, dtype=float)
        approach_tool = np.array([0.0, 0.0, 1.0])
        rod_tool = np.cross(approach_tool, closing_tool)
        rod_tool /= np.linalg.norm(rod_tool)
        tool_basis = np.column_stack((closing_tool, rod_tool, approach_tool))
        world_basis = np.column_stack((closing_world, rod_world, approach_world))
        rotation = world_basis @ tool_basis.T
        if np.linalg.det(rotation) < 0.999:
            raise ValueError("Cannot construct a proper top-down rod grasp orientation.")
        return rotation

    def _pose(self, xyz: np.ndarray, rotation: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in (*xyz, *_rotvec(rotation)))

    def _move(self, name: str, xyz: np.ndarray, rotation: np.ndarray, speed: float | None = None) -> StirringStep:
        return StirringStep("moveL", name, pose=self._pose(xyz, rotation), speed_mps=speed)

    def _candidate_plan(self, candidate: RodGraspCandidate, current_pose: Sequence[float]) -> StirringPlan:
        measured = self._measured_motion()
        motion = self.config.stirring_motion
        up = np.array([0.0, 0.0, 1.0])
        down = -up
        rod_grasp = candidate.grasp_point_world
        tip_vector_world = candidate.opposite_tip_world - rod_grasp
        tip_length = float(np.linalg.norm(tip_vector_world))
        grasp_rotation = self._grasp_rotation(candidate.direction_grasp_to_tip_world)
        rod_tool = grasp_rotation.T @ (tip_vector_world / tip_length)
        rod_center_tool = np.asarray(motion.rod_grasp_center_tool_m, dtype=float)
        tip_tool = rod_center_tool + rod_tool * tip_length
        grasp_tcp = rod_grasp - grasp_rotation @ rod_center_tool
        alignment, alignment_axis, alignment_angle = _rotation_mapping(
            candidate.direction_grasp_to_tip_world, down,
        )
        vertical_rotation = alignment @ grasp_rotation
        if float((vertical_rotation @ rod_tool) @ down) < 0.999:
            raise ValueError("Computed tool orientation does not point the free rod end world-down.")

        pregrasp = grasp_tcp + up * measured["pregrasp_height_m"]
        lift = grasp_tcp + up * measured["lift_height_m"]
        table_z = min(self.geometry.endpoint_a_world[2], self.geometry.endpoint_b_world[2]) - self.geometry.rod_radius_m
        sweep_clearance = self.geometry.rod_radius_m + self.geometry.endpoint_uncertainty_m
        rotation_steps: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(1, motion.orientation_steps + 1):
            rotation = _axis_angle(alignment_axis, alignment_angle * index / motion.orientation_steps) @ grasp_rotation
            tip = lift + rotation @ tip_tool
            if tip[2] <= table_z + sweep_clearance:
                raise ValueError(f"Candidate {candidate.id} rod tip may hit the table during vertical adjustment.")
            rotation_steps.append((rotation, tip))

        mouth = self.scene.beaker_mouth_center
        preinsert_tip = mouth + up * measured["preinsert_clearance_m"]
        stir_center_tip = mouth - up * measured["target_depth_below_mouth_m"]
        beaker_bottom_proxy_z = float(np.min(
            _read_ply_xyz(self.config.result_dir / "models" / "beaker_world.ply")[:, 2]
        ))
        tip_surface_bottom_clearance = (
            stir_center_tip[2] - self.geometry.rod_radius_m - beaker_bottom_proxy_z
        )
        if tip_surface_bottom_clearance < measured["bottom_clearance_m"]:
            raise ValueError(
                f"Requested insertion leaves only {tip_surface_bottom_clearance:.4f} m above the fitted "
                f"beaker-bottom proxy, below bottom_clearance_m={measured['bottom_clearance_m']:.4f}."
            )
        preinsert_tcp = preinsert_tip - vertical_rotation @ tip_tool
        stir_center_tcp = stir_center_tip - vertical_rotation @ tip_tool
        safe_tcp_z = max(
            lift[2], preinsert_tcp[2], self.config.transit_z_m - (vertical_rotation @ tip_tool)[2],
        )
        vertical_safe_over_rod = np.array([lift[0], lift[1], safe_tcp_z])
        vertical_safe_over_cup = np.array([preinsert_tcp[0], preinsert_tcp[1], safe_tcp_z])

        if measured["beaker_safe_inner_radius_m"] > self.scene.beaker_inner_radius_estimate_m:
            raise ValueError("Measured safe beaker inner radius exceeds the fitted mouth radius.")
        required_radius = (
            measured["stir_radius_m"] + self.geometry.endpoint_uncertainty_m + self.geometry.rod_radius_m
        )
        if required_radius >= measured["beaker_safe_inner_radius_m"]:
            raise ValueError("Stirring circle violates rod-radius and endpoint-uncertainty cup margin.")

        current = np.asarray(current_pose, dtype=float)
        current_rotation = rotvec_to_matrix(current[3:])
        initial_transit = current[:3].copy()
        initial_transit[2] = max(initial_transit[2], self.config.transit_z_m)
        steps: list[StirringStep] = [
            StirringStep("gripper", "open_before_rod", gripper_position=self.config.base.gripper.open_position),
            self._move("raise_for_rod", initial_transit, current_rotation),
            self._move("over_rod_transit", np.array([pregrasp[0], pregrasp[1], max(pregrasp[2], self.config.transit_z_m)]), grasp_rotation),
            self._move("rod_pregrasp", pregrasp, grasp_rotation),
            self._move("rod_grasp", grasp_tcp, grasp_rotation),
            StirringStep("checkpoint", "grasp"),
            StirringStep("gripper_force", "set_rod_grasp_force", gripper_force=motion.rod_grasp_force),
            StirringStep("gripper", "close_on_rod", gripper_position=motion.rod_grasp_position),
            StirringStep("wait", "settle_rod_grasp", seconds=motion.settle_s),
            self._move("lift_rod_horizontal", lift, grasp_rotation),
        ]
        for index, (rotation, _) in enumerate(rotation_steps, 1):
            steps.append(self._move(f"align_rod_vertical_{index:02d}", lift, rotation))
        steps.extend((
            self._move("raise_vertical_rod_safe", vertical_safe_over_rod, vertical_rotation),
            self._move("rod_over_beaker_safe", vertical_safe_over_cup, vertical_rotation),
            self._move("rod_tip_preinsert", preinsert_tcp, vertical_rotation),
            StirringStep("checkpoint", "insert"),
            self._move("insert_rod_shallow", stir_center_tcp, vertical_rotation, measured["stir_speed_mps"]),
        ))
        points = motion.stir_cycles * motion.stir_points_per_cycle
        for index in range(1, points + 1):
            theta = 2 * math.pi * index / motion.stir_points_per_cycle
            point = stir_center_tcp + np.array([
                measured["stir_radius_m"] * math.cos(theta),
                measured["stir_radius_m"] * math.sin(theta),
                0.0,
            ])
            steps.append(self._move(f"stir_{index:03d}", point, vertical_rotation, measured["stir_speed_mps"]))
        steps.extend((
            self._move("return_stir_center", stir_center_tcp, vertical_rotation, measured["stir_speed_mps"]),
            self._move("withdraw_rod_vertical", preinsert_tcp, vertical_rotation, measured["stir_speed_mps"]),
            self._move("rod_leave_beaker_safe", vertical_safe_over_cup, vertical_rotation),
            self._move("vertical_rod_over_return", vertical_safe_over_rod, vertical_rotation),
            self._move("lower_vertical_rod_for_restore", lift, vertical_rotation),
        ))
        for index in range(motion.orientation_steps - 1, -1, -1):
            rotation = _axis_angle(alignment_axis, alignment_angle * index / motion.orientation_steps) @ grasp_rotation
            steps.append(self._move(f"restore_rod_horizontal_{index:02d}", lift, rotation))
        steps.extend((
            self._move("return_rod_pregrasp", pregrasp, grasp_rotation),
            self._move("replace_rod", grasp_tcp, grasp_rotation),
            StirringStep("gripper", "release_rod", gripper_position=self.config.base.gripper.open_position),
            self._move("retreat_from_rod", pregrasp, grasp_rotation),
        ))

        lower = np.asarray(self.config.base.motion.workspace_min)
        upper = np.asarray(self.config.base.motion.workspace_max)
        poses = [np.asarray(step.pose, dtype=float) for step in steps if step.pose is not None]
        for step, pose in ((step, np.asarray(step.pose, dtype=float)) for step in steps if step.pose is not None):
            if not np.all(np.isfinite(pose)) or not np.all((pose[:3] >= lower) & (pose[:3] <= upper)):
                raise ValueError(f"Candidate {candidate.id} step {step.name} is outside the Cartesian workspace.")
        margin = min(float(np.min(pose[:3] - lower)) for pose in poses)
        margin = min(margin, min(float(np.min(upper - pose[:3])) for pose in poses))
        warnings = (
            "No complete arm, wrist-camera, cable or environment collision model is available.",
            f"Configured bottom_clearance_m={measured['bottom_clearance_m']:.4f} must be verified against the real beaker bottom; the server does not locate it.",
            "Rod placement and grasp success are open-loop and require operator checkpoints.",
        )
        return StirringPlan(candidate, tuple(steps), margin, warnings)

    def plan_candidates(self, current_pose: Sequence[float]) -> tuple[StirringPlan, ...]:
        plans: list[StirringPlan] = []
        errors: list[str] = []
        for candidate in self.geometry.candidates:
            try:
                plans.append(self._candidate_plan(candidate, current_pose))
            except ValueError as exc:
                errors.append(f"{candidate.id}: {exc}")
        if not plans:
            raise ValueError(f"Neither glass-rod candidate passes geometric planning: {errors}")
        return tuple(sorted(plans, key=lambda plan: plan.workspace_margin_m, reverse=True))

    def _ik_safe(self, plan: StirringPlan) -> tuple[bool, str]:
        for step in plan.steps:
            if step.pose is None:
                continue
            if not self.robot.has_inverse_kinematics(step.pose):
                return False, f"no IK for {step.name}"
            joints = self.robot.inverse_kinematics(step.pose)
            if not self.robot.joints_within_safety_limits(joints):
                return False, f"joint safety limits fail at {step.name}"
        return True, "all sampled moveL poses have IK and pass controller joint safety limits"

    def save_plan(self, plan: StirringPlan, capture_dir: Path) -> Path:
        path = self.result_dir / "local_stirring_plan.json"
        _write_json(path, {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execute": False,
            "stirring_capture_dir": str(capture_dir),
            "stirring_result_dir": str(self.result_dir),
            "beaker_source_job_id": self.scene.source_job_id,
            "beaker_mouth_center_base_m": self.scene.beaker_mouth_center.tolist(),
            "perception": self.geometry.as_dict(),
            "motion_parameters": self.config.stirring_motion.__dict__,
            "plan": plan.as_dict(),
        })
        return path

    def execute(
        self, capture_dir: Path, confirm: Callable[[str, str], None],
    ) -> Path:
        if not self.config.physical_execution_validated or not self.config.base.calibration.validated:
            raise RuntimeError("Physical execution and hand-eye calibration must both be validated.")
        self._measured_motion()
        run_dir = self.result_dir / "local_execution_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        event_path = run_dir / "events.jsonl"
        gripper_connected = False
        selected: StirringPlan | None = None
        try:
            self.robot.connect()
            current = self.robot.actual_tcp_pose()
            geometric_plans = self.plan_candidates(current)
            rejected: list[dict[str, str]] = []
            for plan in geometric_plans:
                ok, reason = self._ik_safe(plan)
                if ok:
                    if selected is None:
                        selected = plan
                    else:
                        rejected.append({
                            "candidate": plan.candidate.id,
                            "reason": "IK passed but workspace margin is smaller than the selected candidate",
                        })
                else:
                    rejected.append({"candidate": plan.candidate.id, "reason": reason})
            if selected is None:
                raise RuntimeError(f"Neither glass-rod candidate passes controller IK/joint-limit checks: {rejected}")
            _write_json(run_dir / "plan_stirring.json", {
                "capture_id": self.geometry.capture_id,
                "job_id": self.geometry.job_id,
                "beaker_source_job_id": self.scene.source_job_id,
                "selected": selected.as_dict(),
                "rejected_candidates": rejected,
                "motion_parameters": self.config.stirring_motion.__dict__,
                "complete_server_result": self.geometry.raw_result,
            })
            print(
                f"Stirring plan: {run_dir / 'plan_stirring.json'}; selected {selected.candidate.id}. "
                "Execution has two motion checkpoints: lowest grasp point and beaker pre-insert alignment.",
                flush=True,
            )
            self.gripper.connect()
            gripper_connected = True
            for step in selected.steps:
                record = {
                    "time": datetime.now(timezone.utc).isoformat(), "stage": "stirring",
                    "step": step.name, "status": "started",
                }
                try:
                    if step.kind == "moveL":
                        assert step.pose is not None
                        self.robot.move_linear(step.pose, speed=step.speed_mps)
                    elif step.kind == "gripper":
                        assert step.gripper_position is not None
                        self.gripper.set_position(step.gripper_position)
                    elif step.kind == "gripper_force":
                        assert step.gripper_force is not None
                        self.gripper.set_force(step.gripper_force)
                    elif step.kind == "wait":
                        time.sleep(float(step.seconds))
                    elif step.kind == "checkpoint":
                        before = self.robot.actual_tcp_pose()
                        confirm(f"stirring.{step.name}", f"Inspect {step.name}; continue only if safe")
                        after = self.robot.actual_tcp_pose()
                        if (
                            np.linalg.norm(np.asarray(before[:3]) - np.asarray(after[:3]))
                            > self.config.base.robot.position_tolerance_m
                            or _rotation_distance(before, after) > self.config.orientation_tolerance_rad
                        ):
                            raise RuntimeError("Robot moved during stirring checkpoint confirmation.")
                    else:
                        raise ValueError(f"Unknown stirring step kind: {step.kind}")
                    record["status"] = "completed"
                except BaseException as exc:
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    raise
                finally:
                    record["time_finished"] = datetime.now(timezone.utc).isoformat()
                    with event_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _write_json(run_dir / "completed.json", {
                "stage": "stirring", "completed_at": datetime.now(timezone.utc).isoformat(),
                "capture_dir": str(capture_dir), "job_id": self.geometry.job_id,
                "selected_candidate": selected.candidate.id,
                "final_tcp_pose": self.robot.actual_tcp_pose(),
            })
            return run_dir
        except BaseException as exc:
            self.robot.stop()
            _write_json(run_dir / "failure.json", {
                "capture_id": self.geometry.capture_id,
                "job_id": self.geometry.job_id,
                "selected_candidate": None if selected is None else selected.candidate.id,
                "motion_parameters": self.config.stirring_motion.__dict__,
                "complete_server_result": self.geometry.raw_result,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise
        finally:
            if gripper_connected:
                self.gripper.disconnect()
            self.robot.disconnect()
