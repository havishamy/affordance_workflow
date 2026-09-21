from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from .camera import RealSenseWristCamera
from .complex_config import ComplexDemoConfig
from .complex_server import ComplexLocalizationClient, ComplexScene
from .geometry import pose_to_matrix, rotvec_to_matrix
from .hardware import DHGripper, URRobot
from .server import build_upload_files

LOG = logging.getLogger(__name__)

STAGE_NAMES = ("open_bottle", "pour", "dropper", "stir")


@dataclass(frozen=True)
class Action:
    kind: str
    name: str
    pose: tuple[float, ...] | None = None
    gripper_position: int | None = None
    seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.pose is not None:
            result["pose"] = list(self.pose)
        if self.gripper_position is not None:
            result["gripper_position"] = self.gripper_position
        if self.seconds is not None:
            result["seconds"] = self.seconds
        return result


@dataclass(frozen=True)
class StagePlan:
    name: str
    actions: tuple[Action, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "actions": [action.as_dict() for action in self.actions]}


def _normalize(vector: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.all(np.isfinite(value)) or norm < 1e-9:
        raise ValueError(f"{name} must be a finite nonzero 3D vector.")
    return value / norm


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = _normalize(axis, "rotation axis")
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    return rotvec.reshape(3)


def _plane_basis(normal: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    n = _normalize(normal, "plane normal")
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first = _normalize(np.cross(n, helper), "plane basis")
    second = np.cross(n, first)
    return first, second


class ComplexDemoWorkflow:
    """Independent four-stage chemistry demo; the original workflow is untouched."""

    def __init__(
        self,
        config: ComplexDemoConfig,
        robot: URRobot | None = None,
        gripper: DHGripper | None = None,
        camera: RealSenseWristCamera | None = None,
        client: ComplexLocalizationClient | None = None,
    ):
        self.config = config
        self.robot = robot or URRobot(config.base.robot)
        self.gripper = gripper or DHGripper(config.base.gripper)
        self.camera = camera or RealSenseWristCamera(config.base.camera)
        self.client = client or ComplexLocalizationClient(
            config.base.server, config.api_path, config.geometry,
            require_glass_rod=config.stir.enabled,
        )

    def run(
        self,
        execute: bool,
        stages: Sequence[str] = STAGE_NAMES,
        scene_json: str | Path | None = None,
        before_stage: Callable[[str, StagePlan], None] | None = None,
    ) -> dict[str, Any]:
        selected = self._validate_stages(stages)
        if execute:
            if not self.config.base.calibration.validated:
                raise RuntimeError("The base hand-eye calibration gate is not validated.")
            if not self.config.physical_accuracy_validated:
                raise RuntimeError(
                    "Complex demo physical geometry is not validated. Set physical_accuracy_validated only after dry-run checks."
                )
        run_id = "complex_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.config.base.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        event_log = run_dir / "execution_events.jsonl"
        camera_connected = False
        gripper_connected = False
        try:
            self.robot.connect()
            initial_pose = self.robot.actual_tcp_pose()
            if scene_json is None:
                self.camera.connect()
                camera_connected = True
                time.sleep(self.config.base.motion.settle_s)
                pose_before = self.robot.actual_tcp_pose()
                frame = self.camera.capture()
                capture_pose = self.robot.actual_tcp_pose()
                self._assert_same_pose(pose_before, capture_pose, "during complex capture")
                t_world_ee = self._world_ee_matrix(capture_pose)
                capture_dir = run_dir / "capture"
                capture_dir.mkdir()
                for filename, content, _ in build_upload_files(frame, t_world_ee).values():
                    (capture_dir / filename).write_bytes(content)
                capture_info = {
                    "capture_id": run_id,
                    "actual_tcp_pose": capture_pose,
                    "tcp_offset": list(self.config.base.robot.tcp_offset),
                    "depth_scale_m_per_unit": frame.depth_scale_m,
                    "world_frame": "base",
                    "ee_frame": self.config.base.server.ee_frame,
                    "camera_serial": frame.camera_serial,
                }
                self._write_json(capture_dir / "capture_info.json", capture_info)
                self.camera.disconnect()
                camera_connected = False
                scene = self.client.locate(frame, t_world_ee, run_id, run_dir / "server_result")
            else:
                payload = self._load_json(Path(scene_json).expanduser().resolve())
                scene = ComplexScene.from_response(
                    payload, self.config.geometry, require_glass_rod="stir" in selected,
                )
                self._write_json(run_dir / "scene_result.json", payload)

            plans = self.build_plan(scene, initial_pose, selected)
            record = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "execute": execute,
                "selected_stages": list(selected),
                "scene": scene.raw_response,
                "derived_geometry_local": {
                    "source": "affordance-region medians plus locally calibrated canonical-model priors",
                    "not_server_predictions": True,
                    "stopper_grasp_world": scene.stopper_grasp.tolist(),
                    "bottle_body_grasp_world": scene.bottle_body_grasp.tolist(),
                    "bottle_axis_out_world": scene.bottle_axis_out.tolist(),
                    "bottle_mouth_approx_world": scene.bottle_mouth.tolist(),
                    "beaker_mouth_prior_world": scene.beaker_mouth.tolist(),
                    "beaker_axis_prior_world": scene.beaker_normal_out.tolist(),
                    "beaker_radius_prior_scaled_m": scene.beaker_radius_m,
                    "dropper_bulb_grasp_world": scene.dropper_bulb_grasp.tolist(),
                    "dropper_tip_prior_world": scene.dropper_tip.tolist(),
                    "glass_rod_grasp_world": None if scene.glass_rod_grasp is None else scene.glass_rod_grasp.tolist(),
                    "glass_rod_lower_tip_world": None if scene.glass_rod_lower_tip is None else scene.glass_rod_lower_tip.tolist(),
                },
                "plans": [plan.as_dict() for plan in plans],
                "warnings": [
                    "All grasp/tool orientations are configured, not inferred from a collision-aware planner.",
                    "Pour quantity, drop count and titration endpoint are open-loop commands.",
                ],
            }
            self._write_json(run_dir / "complex_plan.json", record)
            self._log_plan(plans, execute)

            if execute:
                capture_reference = self.robot.actual_tcp_pose()
                self._assert_same_pose(initial_pose, capture_reference, "before complex execution")
                for stage in plans:
                    pose_before_confirmation = self.robot.actual_tcp_pose()
                    if before_stage is not None:
                        before_stage(stage.name, stage)
                    self._assert_same_pose(
                        pose_before_confirmation,
                        self.robot.actual_tcp_pose(),
                        f"while confirming stage {stage.name}",
                    )
                    if not gripper_connected:
                        self.gripper.connect()
                        gripper_connected = True
                    self._execute_stage(stage, event_log)
                record["status"] = "completed"
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._write_json(run_dir / "execution_result.json", record)
            return {**record, "run_dir": str(run_dir)}
        except BaseException as exc:
            self.robot.stop()
            self._write_json(
                run_dir / "failure.json",
                {"error_type": type(exc).__name__, "error": str(exc), "time": datetime.now(timezone.utc).isoformat()},
            )
            raise
        finally:
            if camera_connected:
                self.camera.disconnect()
            if gripper_connected:
                self.gripper.disconnect()
            self.robot.disconnect()

    def build_plan(
        self,
        scene: ComplexScene,
        current_tcp_pose: Sequence[float],
        stages: Sequence[str] = STAGE_NAMES,
    ) -> tuple[StagePlan, ...]:
        selected = self._validate_stages(stages)
        plans: list[StagePlan] = []
        current = np.asarray(current_tcp_pose[:3], dtype=float)
        for stage_name in selected:
            if stage_name == "open_bottle":
                stage = self._plan_open_bottle(scene, current)
            elif stage_name == "pour":
                stage = self._plan_pour(scene, current)
            elif stage_name == "dropper":
                stage = self._plan_dropper(scene, current)
            elif stage_name == "stir":
                if not self.config.stir.enabled:
                    raise ValueError("The stir stage was requested but stir.enabled is false.")
                stage = self._plan_stir(scene, current)
            else:  # protected by _validate_stages
                raise AssertionError(stage_name)
            for action in stage.actions:
                if action.kind == "move_linear" and action.pose is not None:
                    self._validate_pose(action.pose, f"{stage.name}.{action.name}")
                    current = np.asarray(action.pose[:3], dtype=float)
            plans.append(stage)
        return tuple(plans)

    def _plan_open_bottle(self, scene: ComplexScene, current: np.ndarray) -> StagePlan:
        cfg = self.config.stopper
        axis = scene.stopper_axis_out
        grasp = scene.stopper_grasp + np.asarray(cfg.grasp_offset_m)
        approach = grasp + axis * cfg.approach_m
        pull = grasp + axis * cfg.pull_distance_m
        park = np.asarray(cfg.parking_pose[:3], dtype=float)
        park_orientation = cfg.parking_pose[3:]
        actions = [
            self._move("raise_for_stopper", self._transit(current), cfg.orientation_rotvec),
            self._move("above_stopper_transit", self._transit(approach), cfg.orientation_rotvec),
            self._move("stopper_axis_approach", approach, cfg.orientation_rotvec),
            self._move("stopper_grasp", grasp, cfg.orientation_rotvec),
            self._gripper("grasp_stopper", cfg.gripper_position),
            self._wait("settle_stopper_grasp", self.config.base.motion.grasp_settle_s),
            self._move("pull_stopper_along_axis", pull, cfg.orientation_rotvec),
            self._move("lift_stopper_to_transit", self._transit(pull), cfg.orientation_rotvec),
            self._move("stopper_parking_transit", self._transit(park), park_orientation),
            self._move("stopper_parking_approach", park + np.array([0, 0, cfg.parking_approach_m]), park_orientation),
            self._move("place_stopper", park, park_orientation),
            self._gripper("release_stopper", self.config.base.gripper.open_position),
            self._move("retreat_from_stopper", park + np.array([0, 0, cfg.parking_approach_m]), park_orientation),
        ]
        return StagePlan("open_bottle", tuple(actions))

    def _plan_pour(self, scene: ComplexScene, current: np.ndarray) -> StagePlan:
        cfg = self.config.bottle
        grasp = scene.bottle_body_grasp + np.asarray(cfg.grasp_offset_m)
        approach = grasp + scene.bottle_axis_out * cfg.approach_m
        upright_r = rotvec_to_matrix(cfg.orientation_rotvec)
        mouth_from_tcp = upright_r.T @ (scene.bottle_mouth - grasp)
        desired_mouth = (
            scene.beaker_mouth
            + scene.beaker_normal_out * cfg.mouth_clearance_m
            + np.asarray(cfg.pour_target_offset_m)
        )
        actions = [
            self._move("raise_for_bottle", self._transit(current), cfg.orientation_rotvec),
            self._move("above_bottle_transit", self._transit(approach), cfg.orientation_rotvec),
            self._move("bottle_grasp_approach", approach, cfg.orientation_rotvec),
            self._move("bottle_body_grasp", grasp, cfg.orientation_rotvec),
            self._gripper("grasp_bottle", cfg.gripper_position),
            self._wait("settle_bottle_grasp", self.config.base.motion.grasp_settle_s),
            self._move("lift_bottle", grasp + scene.bottle_axis_out * cfg.lift_m, cfg.orientation_rotvec),
            self._move("bottle_transit", self._transit(grasp), cfg.orientation_rotvec),
        ]
        tilt_axis = _normalize(cfg.tilt_axis_tool, "bottle.tilt_axis_tool")
        for index in range(cfg.tilt_steps + 1):
            fraction = index / cfg.tilt_steps
            rotation = upright_r @ _axis_rotation(tilt_axis, math.radians(cfg.tilt_angle_deg) * fraction)
            tcp_xyz = desired_mouth - rotation @ mouth_from_tcp
            actions.append(self._move_matrix(f"pour_tilt_{index:02d}", tcp_xyz, rotation))
        actions.append(self._wait("hold_pour", cfg.hold_s))
        for index in range(cfg.tilt_steps - 1, -1, -1):
            fraction = index / cfg.tilt_steps
            rotation = upright_r @ _axis_rotation(tilt_axis, math.radians(cfg.tilt_angle_deg) * fraction)
            tcp_xyz = desired_mouth - rotation @ mouth_from_tcp
            actions.append(self._move_matrix(f"pour_restore_{index:02d}", tcp_xyz, rotation))
        actions.extend([
            self._move("return_bottle_transit", self._transit(grasp), cfg.orientation_rotvec),
            self._move("return_bottle_approach", approach, cfg.orientation_rotvec),
            self._move("replace_bottle", grasp, cfg.orientation_rotvec),
            self._gripper("release_bottle", self.config.base.gripper.open_position),
            self._move("retreat_from_bottle", approach, cfg.orientation_rotvec),
        ])
        return StagePlan("pour", tuple(actions))

    def _plan_dropper(self, scene: ComplexScene, current: np.ndarray) -> StagePlan:
        cfg = self.config.dropper
        orientation = rotvec_to_matrix(cfg.orientation_rotvec)
        bulb = scene.dropper_bulb_grasp + np.asarray(cfg.grasp_offset_m)
        axis_out = _normalize(scene.dropper_bulb_grasp - scene.dropper_tip, "dropper outward axis")
        approach = bulb + axis_out * cfg.approach_m
        tip_from_tcp = orientation.T @ (scene.dropper_tip - bulb)
        desired_tip = (
            scene.beaker_mouth
            + scene.beaker_normal_out * cfg.tip_clearance_m
            + np.asarray(cfg.target_offset_m)
        )
        dispense_tcp = desired_tip - orientation @ tip_from_tcp
        retreat_tip = desired_tip + scene.beaker_normal_out * cfg.approach_m
        retreat_tcp = retreat_tip - orientation @ tip_from_tcp
        actions = [
            self._move("raise_for_dropper", self._transit(current), cfg.orientation_rotvec),
            self._move("above_dropper_transit", self._transit(approach), cfg.orientation_rotvec),
            self._move("dropper_axis_approach", approach, cfg.orientation_rotvec),
            self._move("dropper_bulb_grasp", bulb, cfg.orientation_rotvec),
            self._gripper("grasp_dropper", cfg.hold_position),
            self._wait("settle_dropper_grasp", self.config.base.motion.grasp_settle_s),
            self._move("extract_dropper", bulb + axis_out * cfg.lift_m, cfg.orientation_rotvec),
            self._move("dropper_transit", self._transit(bulb), cfg.orientation_rotvec),
            self._move("align_tip_above_beaker", dispense_tcp, cfg.orientation_rotvec),
        ]
        for index in range(cfg.squeeze_cycles):
            actions.extend([
                self._gripper(f"squeeze_drop_{index + 1}", cfg.squeeze_position),
                self._wait(f"hold_squeeze_{index + 1}", cfg.squeeze_hold_s),
                self._gripper(f"release_squeeze_{index + 1}", cfg.hold_position),
                self._wait(f"wait_between_drops_{index + 1}", cfg.squeeze_hold_s),
            ])
        actions.extend([
            self._move("retract_dropper_from_beaker", retreat_tcp, cfg.orientation_rotvec),
            self._move("return_dropper_transit", self._transit(bulb), cfg.orientation_rotvec),
            self._move("return_dropper_approach", approach, cfg.orientation_rotvec),
            self._move("replace_dropper", bulb, cfg.orientation_rotvec),
            self._gripper("release_dropper", self.config.base.gripper.open_position),
            self._move("retreat_from_dropper", approach, cfg.orientation_rotvec),
        ])
        return StagePlan("dropper", tuple(actions))

    def _plan_stir(self, scene: ComplexScene, current: np.ndarray) -> StagePlan:
        cfg = self.config.stir
        if scene.glass_rod_grasp is None or scene.glass_rod_lower_tip is None:
            raise ValueError("Glass rod targets are required for the stir stage.")
        orientation = rotvec_to_matrix(cfg.orientation_rotvec)
        grasp = scene.glass_rod_grasp + np.asarray(cfg.grasp_offset_m)
        rod_axis_out = _normalize(scene.glass_rod_grasp - scene.glass_rod_lower_tip, "glass rod outward axis")
        approach = grasp + rod_axis_out * cfg.approach_m
        lower_from_tcp = orientation.T @ (scene.glass_rod_lower_tip - grasp)
        preinsert_tip = scene.beaker_mouth + scene.beaker_normal_out * cfg.preinsert_clearance_m
        preinsert_tcp = preinsert_tip - orientation @ lower_from_tcp
        # No liquid-surface perception is claimed. This is a configured open-loop
        # target below the locally calibrated beaker-mouth model anchor.
        stir_center = scene.stir_reference_mouth - scene.beaker_normal_out * cfg.target_depth_below_mouth_m
        basis_u, basis_v = _plane_basis(scene.beaker_normal_out)
        if cfg.radius_m >= 0.8 * scene.beaker_radius_m:
            raise RuntimeError(
                f"Stir radius {cfg.radius_m:.4f} m is too close to beaker radius {scene.beaker_radius_m:.4f} m."
            )
        actions = [
            self._move("raise_for_glass_rod", self._transit(current), cfg.orientation_rotvec),
            self._move("above_glass_rod_transit", self._transit(approach), cfg.orientation_rotvec),
            self._move("glass_rod_approach", approach, cfg.orientation_rotvec),
            self._move("glass_rod_grasp", grasp, cfg.orientation_rotvec),
            self._gripper("grasp_glass_rod", cfg.gripper_position),
            self._wait("settle_glass_rod_grasp", self.config.base.motion.grasp_settle_s),
            self._move("lift_glass_rod", grasp + rod_axis_out * cfg.lift_m, cfg.orientation_rotvec),
            self._move("glass_rod_transit", self._transit(grasp), cfg.orientation_rotvec),
            self._move("rod_tip_above_beaker", preinsert_tcp, cfg.orientation_rotvec),
        ]
        total_points = cfg.turns * cfg.points_per_turn
        for index in range(total_points + 1):
            angle = 2.0 * math.pi * index / cfg.points_per_turn
            desired_lower_tip = stir_center + cfg.radius_m * (math.cos(angle) * basis_u + math.sin(angle) * basis_v)
            tcp_xyz = desired_lower_tip - orientation @ lower_from_tcp
            actions.append(self._move(f"stir_{index:03d}", tcp_xyz, cfg.orientation_rotvec))
            if cfg.point_hold_s > 0:
                actions.append(self._wait(f"stir_dwell_{index:03d}", cfg.point_hold_s))
        actions.extend([
            self._move("extract_rod_above_beaker", preinsert_tcp, cfg.orientation_rotvec),
            self._move("return_rod_transit", self._transit(grasp), cfg.orientation_rotvec),
            self._move("return_rod_approach", approach, cfg.orientation_rotvec),
            self._move("replace_glass_rod", grasp, cfg.orientation_rotvec),
            self._gripper("release_glass_rod", self.config.base.gripper.open_position),
            self._move("retreat_from_glass_rod", approach, cfg.orientation_rotvec),
        ])
        return StagePlan("stir", tuple(actions))

    def _execute_stage(self, stage: StagePlan, event_log: Path) -> None:
        self._event(event_log, stage.name, "stage", "started")
        try:
            for action in stage.actions:
                self._event(event_log, stage.name, action.name, "started")
                if action.kind == "move_linear":
                    assert action.pose is not None
                    self.robot.move_linear(action.pose)
                elif action.kind == "gripper":
                    assert action.gripper_position is not None
                    self.gripper.set_position(action.gripper_position)
                elif action.kind == "wait":
                    time.sleep(float(action.seconds or 0.0))
                else:
                    raise ValueError(f"Unknown complex action kind: {action.kind}")
                self._event(event_log, stage.name, action.name, "completed")
            self._event(event_log, stage.name, "stage", "completed")
        except BaseException as exc:
            self._event(event_log, stage.name, "stage", "failed", str(exc))
            raise

    def _world_ee_matrix(self, tcp_pose: Sequence[float]) -> np.ndarray:
        t_base_tcp = pose_to_matrix(tcp_pose)
        frame = self.config.base.server.ee_frame
        if frame == "tcp":
            return t_base_tcp
        if frame == "flange":
            return t_base_tcp @ np.linalg.inv(pose_to_matrix(self.config.base.robot.tcp_offset))
        raise ValueError("server.ee_frame must be tcp or flange.")

    def _assert_same_pose(self, before: Sequence[float], after: Sequence[float], stage: str) -> None:
        first, second = pose_to_matrix(before), pose_to_matrix(after)
        distance = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        relative = first[:3, :3].T @ second[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        if distance > self.config.base.robot.position_tolerance_m or angle > 0.02:
            raise RuntimeError(f"Robot pose changed {stage}: {distance:.4f} m, {angle:.4f} rad.")

    def _transit(self, xyz: Sequence[float]) -> np.ndarray:
        result = np.asarray(xyz, dtype=float).copy()
        result[2] = max(result[2], self.config.base.motion.transit_z_m)
        return result

    @staticmethod
    def _move(name: str, xyz: Sequence[float], rotvec: Sequence[float]) -> Action:
        return Action("move_linear", name, tuple(float(v) for v in (*xyz, *rotvec)))

    @staticmethod
    def _move_matrix(name: str, xyz: Sequence[float], rotation: np.ndarray) -> Action:
        return ComplexDemoWorkflow._move(name, xyz, _matrix_to_rotvec(rotation))

    @staticmethod
    def _gripper(name: str, position: int) -> Action:
        return Action("gripper", name, gripper_position=int(position))

    @staticmethod
    def _wait(name: str, seconds: float) -> Action:
        return Action("wait", name, seconds=float(seconds))

    def _validate_pose(self, pose: Sequence[float], label: str) -> None:
        value = np.asarray(pose, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} is not a finite six-value pose.")
        lower = np.asarray(self.config.base.motion.workspace_min)
        upper = np.asarray(self.config.base.motion.workspace_max)
        if not np.all((value[:3] >= lower) & (value[:3] <= upper)):
            raise RuntimeError(f"{label} position {value[:3].tolist()} is outside workspace {lower.tolist()}..{upper.tolist()}.")

    def _validate_stages(self, stages: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(stages)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Select one or more unique complex-demo stages.")
        unknown = set(selected) - set(STAGE_NAMES)
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}")
        return selected

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        return payload

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _event(path: Path, stage: str, action: str, status: str, error: str | None = None) -> None:
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "action": action,
            "status": status,
        }
        if error is not None:
            payload["error"] = error
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _log_plan(plans: Sequence[StagePlan], execute: bool) -> None:
        LOG.info("Complex demo mode: %s", "EXECUTE" if execute else "PLAN ONLY")
        for stage in plans:
            move_count = sum(action.kind == "move_linear" for action in stage.actions)
            LOG.info("Stage %-12s actions=%d moveL=%d", stage.name, len(stage.actions), move_count)
