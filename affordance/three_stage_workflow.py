from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .geometry import rotvec_to_matrix
from .hardware import DHGripper, URRobot
from .three_stage_config import ThreeStageConfig
from .three_stage_result import SceneGeometry, _read_ply_xyz, load_scene

LOG = logging.getLogger(__name__)
CHEMISTRY_STAGES = ("stopper", "bottle", "dropper")
STAGES = (*CHEMISTRY_STAGES, "stirring")


@dataclass(frozen=True)
class Step:
    kind: str
    name: str
    pose: tuple[float, ...] | None = None
    gripper_position: int | None = None
    seconds: float | None = None

    def as_dict(self) -> dict:
        result = {"kind": self.kind, "name": self.name}
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
    steps: tuple[Step, ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [step.as_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }

    @property
    def last_pose(self) -> tuple[float, ...]:
        return next(step.pose for step in reversed(self.steps) if step.pose is not None)


def _pose(xyz: Sequence[float], rotvec: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in (*xyz, *rotvec))


def _move(name: str, xyz: Sequence[float], rotvec: Sequence[float]) -> Step:
    return Step("moveL", name, pose=_pose(xyz, rotvec))


def _grip(name: str, position: int) -> Step:
    return Step("gripper", name, gripper_position=int(position))


def _wait(name: str, seconds: float) -> Step:
    return Step("wait", name, seconds=float(seconds))


def _checkpoint(name: str) -> Step:
    return Step("checkpoint", name)


def _axis_angle_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _rotvec(rotation: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    return vector.reshape(3)


class ThreeStageWorkflow:
    """Offline-first client-side planner and guarded UR executor for one saved scene."""

    def __init__(self, config: ThreeStageConfig, robot: URRobot | None = None, gripper: DHGripper | None = None):
        self.config = config
        self.robot = robot or URRobot(config.base.robot)
        self.gripper = gripper or DHGripper(config.base.gripper)

    def load_scene(self) -> SceneGeometry:
        return load_scene(self.config)

    def captured_tcp_pose(self) -> list[float]:
        """Return the historical preview pose; execution replans from live TCP.

        Scene points are already in UR base. A changed TCP setting does not
        change those world points, provided the *current* configured TCP is
        physically calibrated to the intended gripper contact center.
        """
        path = self.config.capture_dir / "capture_info.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        pose = np.asarray(info.get("actual_tcp_pose"), dtype=float)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ValueError("capture_info.json does not contain a valid actual_tcp_pose.")
        offset = np.asarray(info.get("tcp_offset"), dtype=float)
        if offset.shape != (6,) or not np.all(np.isfinite(offset)):
            raise ValueError("capture_info.json does not contain a valid TCP offset.")
        current_offset = np.asarray(self.config.base.robot.tcp_offset, dtype=float)
        if not np.allclose(offset, current_offset, atol=1e-9):
            LOG.warning(
                "TCP offset changed since capture: saved=%s, current=%s. "
                "Saved object points remain in UR base; live motion assumes the current TCP is the "
                "physically measured gripper grasp center. Offline preview starts from the historical pose, "
                "but execution replans from live TCP.",
                offset.tolist(), current_offset.tolist(),
            )
        return pose.tolist()

    def plan_stage(self, name: str, scene: SceneGeometry, current_pose: Sequence[float]) -> StagePlan:
        if name == "stopper":
            plan = self._plan_stopper(scene, current_pose)
        elif name == "bottle":
            plan = self._plan_bottle(scene, current_pose)
        elif name == "dropper":
            plan = self._plan_dropper(scene, current_pose)
        else:
            raise ValueError(f"Unknown stage: {name}")
        warnings = self._validate_plan(plan, current_pose, scene)
        plan = replace(plan, warnings=(*plan.warnings, *warnings))
        for warning in plan.warnings:
            LOG.warning("%s planning advisory: %s", name, warning)
        return plan

    def plan_all(self, scene: SceneGeometry, initial_pose: Sequence[float]) -> tuple[StagePlan, ...]:
        pose = tuple(initial_pose)
        stages: list[StagePlan] = []
        for name in CHEMISTRY_STAGES:
            plan = self.plan_stage(name, scene, pose)
            stages.append(plan)
            pose = plan.last_pose
        return tuple(stages)

    def _transit(self, xyz: Sequence[float]) -> np.ndarray:
        result = np.asarray(xyz, dtype=float).copy()
        result[2] = max(result[2], self.config.transit_z_m)
        return result

    def _plan_stopper(self, scene: SceneGeometry, current_pose: Sequence[float]) -> StagePlan:
        cfg = self.config.stopper
        center = scene.stopper_grasp
        axis = scene.bottle_axis_out
        approach = center + axis * cfg.approach_m
        pulled = center + axis * cfg.pull_distance_m
        place = np.asarray(cfg.place_xyz_base_m)
        place_above = place + np.array([0, 0, cfg.place_approach_m])
        steps = (
            _grip("open_before_stopper", self.config.base.gripper.open_position),
            _move("raise_to_transit", self._transit(current_pose[:3]), current_pose[3:]),
            _move("over_stopper", self._transit(approach), cfg.orientation_rotvec),
            _move("stopper_approach", approach, cfg.orientation_rotvec),
            _move("stopper_grasp", center, cfg.orientation_rotvec),
            _grip("close_on_stopper", cfg.gripper_position),
            _wait("settle_stopper", cfg.settle_s),
            _checkpoint("check_stopper_grip_before_pull"),
            _move("pull_stopper_vertically", pulled, cfg.orientation_rotvec),
            _checkpoint("check_stopper_removed"),
            _move("lift_stopper_to_transit", self._transit(pulled), cfg.orientation_rotvec),
            _move("over_stopper_place", self._transit(place_above), cfg.orientation_rotvec),
            _move("stopper_place_approach", place_above, cfg.orientation_rotvec),
            _move("place_stopper", place, cfg.orientation_rotvec),
            _grip("release_stopper", self.config.base.gripper.open_position),
            _move("retreat_from_stopper", place_above, cfg.orientation_rotvec),
        )
        return StagePlan("stopper", steps)

    def _plan_bottle(self, scene: SceneGeometry, current_pose: Sequence[float]) -> StagePlan:
        cfg = self.config.bottle
        warnings: list[str] = []
        if cfg.pour_side_offset_m <= 0 or cfg.pour_side_offset_m > scene.beaker_inner_radius_estimate_m - 0.003:
            warnings.append(
                f"Pour side offset {cfg.pour_side_offset_m:.4f} m is outside the fitted rim's "
                f"advisory 3 mm margin (fitted radius {scene.beaker_inner_radius_estimate_m:.4f} m). "
                "Check the actual cup and liquid landing point."
            )
        up = scene.bottle_axis_out
        grasp = scene.bottle_body_grasp + up * cfg.grasp_offset_along_axis_m
        approach = grasp + up * cfg.approach_m
        mouth = scene.stopper_grasp + up * cfg.stopper_center_to_mouth_m
        upright_rotation = rotvec_to_matrix(cfg.orientation_rotvec)
        mouth_tool = upright_rotation.T @ (mouth - grasp)

        from_beaker_to_bottle = grasp[:2] - scene.beaker_mouth_center[:2]
        from_beaker_to_bottle /= np.linalg.norm(from_beaker_to_bottle)
        toward_beaker = -from_beaker_to_bottle
        side_xy = scene.beaker_mouth_center[:2] + from_beaker_to_bottle * cfg.pour_side_offset_m
        high_mouth = np.array([*side_xy, scene.beaker_mouth_center[2] + cfg.pretilt_mouth_height_m])
        low_mouth = np.array([*side_xy, scene.beaker_mouth_center[2] + cfg.pour_mouth_clearance_m])
        tilt_axis = np.cross(up, np.array([*toward_beaker, 0.0]))
        tilt_axis /= np.linalg.norm(tilt_axis)

        def tcp_for_mouth(desired_mouth: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            return desired_mouth - rotation @ mouth_tool

        high_upright_tcp = tcp_for_mouth(high_mouth, upright_rotation)
        steps: list[Step] = [
            _grip("open_before_bottle", self.config.base.gripper.open_position),
            _move("raise_for_bottle", self._transit(current_pose[:3]), current_pose[3:]),
            _move("over_bottle", self._transit(approach), cfg.orientation_rotvec),
            _move("bottle_approach", approach, cfg.orientation_rotvec),
            _move("bottle_body_grasp", grasp, cfg.orientation_rotvec),
            _grip("close_on_bottle", cfg.gripper_position),
            _checkpoint("check_bottle_grip_before_lift"),
            _move("lift_bottle", grasp + up * cfg.lift_m, cfg.orientation_rotvec),
            _checkpoint("check_bottle_clear_of_table"),
            _move("bottle_to_transit", self._transit(grasp), cfg.orientation_rotvec),
            _move("bottle_over_beaker_transit", self._transit(high_upright_tcp), cfg.orientation_rotvec),
            _move("bottle_high_upright", high_upright_tcp, cfg.orientation_rotvec),
            _checkpoint("check_high_pour_clearance"),
        ]

        high_tilt: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(1, cfg.tilt_steps + 1):
            fraction = index / cfg.tilt_steps
            rotation = _axis_angle_rotation(tilt_axis, math.radians(cfg.tilt_angle_deg) * fraction) @ upright_rotation
            high_tilt.append((rotation, tcp_for_mouth(high_mouth, rotation)))
            steps.append(_move(f"tilt_high_{index:02d}", high_tilt[-1][1], _rotvec(rotation)))
        steps.append(_checkpoint("check_tilt_direction_and_clearance"))
        final_rotation, final_high_tcp = high_tilt[-1]
        for index in range(1, cfg.lower_steps + 1):
            fraction = index / cfg.lower_steps
            desired = high_mouth * (1 - fraction) + low_mouth * fraction
            steps.append(_move(f"lower_tilted_bottle_{index:02d}", tcp_for_mouth(desired, final_rotation), _rotvec(final_rotation)))
        steps.extend((
            _checkpoint("check_liquid_enters_beaker_before_hold"),
            _wait("hold_pour", cfg.hold_s),
        ))
        for index in range(cfg.lower_steps - 1, -1, -1):
            fraction = index / cfg.lower_steps
            desired = high_mouth * (1 - fraction) + low_mouth * fraction
            steps.append(_move(f"raise_tilted_bottle_{index:02d}", tcp_for_mouth(desired, final_rotation), _rotvec(final_rotation)))
        for index in range(cfg.tilt_steps - 1, -1, -1):
            fraction = index / cfg.tilt_steps
            rotation = _axis_angle_rotation(tilt_axis, math.radians(cfg.tilt_angle_deg) * fraction) @ upright_rotation
            steps.append(_move(f"restore_bottle_{index:02d}", tcp_for_mouth(high_mouth, rotation), _rotvec(rotation)))
        steps.extend((
            _wait("settle_after_restore", cfg.restore_settle_s),
            _move("bottle_return_transit", self._transit(high_upright_tcp), cfg.orientation_rotvec),
            _move("over_bottle_original", self._transit(approach), cfg.orientation_rotvec),
            _move("return_bottle_approach", approach, cfg.orientation_rotvec),
            _move("replace_bottle", grasp, cfg.orientation_rotvec),
            _grip("release_bottle", self.config.base.gripper.open_position),
            _move("retreat_from_bottle", approach, cfg.orientation_rotvec),
        ))
        return StagePlan("bottle", tuple(steps), tuple(warnings))

    def _plan_dropper(self, scene: SceneGeometry, current_pose: Sequence[float]) -> StagePlan:
        cfg = self.config.dropper
        bulb = scene.dropper_bulb_grasp + np.array([0.0, 0.0, cfg.grasp_offset_z_m])
        approach = bulb + np.array([0, 0, cfg.approach_m])
        extracted = bulb + np.array([0, 0, cfg.extraction_m])
        # tip_below_bulb_m is measured from the nominal affordance center.
        # Moving the contact point down shortens the grasp-center-to-tip length.
        tip_below_shifted_grasp_m = cfg.tip_below_bulb_m + cfg.grasp_offset_z_m
        target = np.array([
            scene.beaker_mouth_center[0], scene.beaker_mouth_center[1],
            scene.beaker_mouth_center[2] + tip_below_shifted_grasp_m + cfg.tip_clearance_above_beaker_m,
        ])
        above_target = target + np.array([0, 0, cfg.approach_m])
        steps: list[Step] = [
            _grip("open_before_dropper", self.config.base.gripper.open_position),
            _move("raise_for_dropper", self._transit(current_pose[:3]), current_pose[3:]),
            _move("over_dropper", self._transit(approach), cfg.orientation_rotvec),
            _move("dropper_approach", approach, cfg.orientation_rotvec),
            _move("dropper_bulb_grasp", bulb, cfg.orientation_rotvec),
            _grip("close_on_bulb", cfg.hold_position),
            _wait("settle_dropper_grasp", cfg.settle_s),
            _checkpoint("check_dropper_grip_before_extract"),
            _move("extract_dropper", extracted, cfg.orientation_rotvec),
            _checkpoint("check_pipette_is_free"),
            _move("dropper_to_transit", self._transit(extracted), cfg.orientation_rotvec),
            _move("dropper_over_beaker_transit", self._transit(above_target), cfg.orientation_rotvec),
            _move("dropper_above_target", above_target, cfg.orientation_rotvec),
            _move("dropper_dose_position", target, cfg.orientation_rotvec),
            _checkpoint("check_tip_clearance_before_dosing"),
        ]
        for number in range(1, cfg.squeeze_cycles + 1):
            steps.extend((
                _grip(f"squeeze_{number}", cfg.squeeze_position),
                _wait(f"hold_squeeze_{number}", cfg.squeeze_hold_s),
                _grip(f"relax_{number}", cfg.hold_position),
                _wait(f"wait_drop_{number}", cfg.squeeze_hold_s),
            ))
        steps.extend((
            _move("retract_dropper_from_beaker", above_target, cfg.orientation_rotvec),
            _move("dropper_return_transit", self._transit(extracted), cfg.orientation_rotvec),
            _move("over_dropper_original", self._transit(approach), cfg.orientation_rotvec),
            _move("return_dropper_approach", approach, cfg.orientation_rotvec),
            _move("replace_dropper", bulb, cfg.orientation_rotvec),
            _grip("release_dropper", self.config.base.gripper.open_position),
            _move("retreat_from_dropper", approach, cfg.orientation_rotvec),
        ))
        return StagePlan("dropper", tuple(steps))

    def _validate_pose(self, pose: Sequence[float], label: str) -> None:
        values = np.asarray(pose, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} is not a finite UR pose.")
        lower = np.asarray(self.config.base.motion.workspace_min)
        upper = np.asarray(self.config.base.motion.workspace_max)
        if not np.all((values[:3] >= lower) & (values[:3] <= upper)):
            raise ValueError(f"{label} xyz {values[:3].tolist()} is outside workspace {lower.tolist()}..{upper.tolist()}.")

    def _validate_plan(
        self, plan: StagePlan, initial_pose: Sequence[float], scene: SceneGeometry,
    ) -> list[str]:
        # Validate every interpolated Cartesian segment, not just endpoints.
        previous = np.asarray(initial_pose, dtype=float)
        self._validate_pose(previous, f"{plan.name}.initial")
        for step in plan.steps:
            if step.pose is None:
                continue
            target = np.asarray(step.pose, dtype=float)
            self._validate_pose(target, f"{plan.name}.{step.name}")
            segment_length = np.linalg.norm(target[:3] - previous[:3])
            n = max(2, int(math.ceil(segment_length / 0.01)))
            for index in range(1, n + 1):
                fraction = index / n
                position = previous[:3] * (1 - fraction) + target[:3] * fraction
                if not np.all((position >= self.config.base.motion.workspace_min) & (position <= self.config.base.motion.workspace_max)):
                    raise ValueError(f"{plan.name}.{step.name} leaves the workspace between waypoints.")
            previous = target
        if plan.name == "bottle":
            return self._bottle_model_clearance_advisories(plan, scene)
        return []

    def _bottle_model_clearance_advisories(
        self, plan: StagePlan, scene: SceneGeometry,
    ) -> list[str]:
        """Report uncertain model clearances; never claim full collision safety."""
        cloud = _read_ply_xyz(self.config.result_dir / "models/wide_mouth_bottle_world.ply")
        grasp = scene.bottle_body_grasp + scene.bottle_axis_out * self.config.bottle.grasp_offset_along_axis_m
        rotation_at_grasp = rotvec_to_matrix(self.config.bottle.orientation_rotvec)
        minimum_table_z = float(np.min(cloud[:, 2])) + 0.003
        risks: list[str] = []
        check = False
        for step in plan.steps:
            if step.pose is None:
                continue
            if step.name == "lift_bottle":
                check = True
            if not check:
                continue
            if step.name == "return_bottle_approach":
                break
            target = np.asarray(step.pose, dtype=float)
            rotated = (rotvec_to_matrix(target[3:]) @ rotation_at_grasp.T @ (cloud - grasp).T).T
            placed = rotated + target[:3]
            if np.min(placed[:, 2]) < minimum_table_z:
                risks.append(f"Bottle model may intersect the table at {step.name}.")
            if step.name.startswith(("tilt_high_", "lower_tilted_bottle_", "raise_tilted_bottle_", "restore_bottle_")):
                # This excludes the whole reconstructed bottle from the cup's
                # cylindrical opening; liquid is not represented by this model.
                xy = np.linalg.norm(placed[:, :2] - scene.beaker_mouth_center[:2], axis=1)
                inside = (xy < scene.beaker_inner_radius_estimate_m) & (
                    placed[:, 2] < scene.beaker_mouth_center[2] + 0.010
                )
                if np.any(inside):
                    risks.append(
                        f"Bottle model may enter the fitted beaker rim at {step.name} "
                        f"({int(np.count_nonzero(inside))} sampled points); inspect real clearance."
                    )
        if len(risks) > 5:
            return [*risks[:5], f"... plus {len(risks) - 5} additional model-clearance advisories."]
        return risks

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_offline_plan(self, scene: SceneGeometry, plans: Sequence[StagePlan]) -> Path:
        output = self.config.result_dir / "local_three_stage_plan.json"
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execute": False,
            "source_capture": str(self.config.capture_dir),
            "source_result": str(self.config.result_dir / "result.json"),
            "grasp_strategy": "top-down approach; lateral jaw closure on bottle body",
            "derived_geometry": scene.as_dict(),
            "stages": [stage.as_dict() for stage in plans],
            "warnings": [
                "No continuous collision model for arm/gripper/objects.",
                "Stopper parking and gripper apertures require physical calibration.",
                "Bottle mouth offset and pipette tip length are physical placeholders.",
                "Pour volume and true drop count are not measured; actions are open-loop.",
                "The saved scene must be unchanged before real execution.",
                *scene.quality_report.get("warnings", []),
                *(warning for stage in plans for warning in stage.warnings),
            ],
        }
        self._write_json(output, record)
        return output

    def execute(
        self,
        scene: SceneGeometry,
        stage_names: Sequence[str],
        confirm: Callable[[str, str], None],
    ) -> Path:
        if not self.config.physical_execution_validated:
            raise RuntimeError("physical_execution_validated is false; refusing to connect to robot.")
        if not self.config.base.calibration.validated:
            raise RuntimeError("Hand-eye calibration is not validated; refusing to connect to robot.")
        if tuple(stage_names) != CHEMISTRY_STAGES and not (
            len(stage_names) == 1 and stage_names[0] in CHEMISTRY_STAGES
        ):
            raise ValueError("Select all three stages in order, or exactly one stage.")
        # A low transit height can turn the first horizontal move from an
        # arbitrary start pose into a sweep through the saved scene. This is
        # only an object-height proxy; it is not full collision checking.
        tallest_model_z = max(
            float(np.max(_read_ply_xyz(self.config.result_dir / "models" / f"{name}_world.ply")[:, 2]))
            for name in ("wide_mouth_bottle", "beaker", "dropper")
        )
        minimum_transit_z = tallest_model_z + 0.080
        transit_warning = None
        if self.config.transit_z_m < minimum_transit_z:
            transit_warning = (
                f"Configured transit Z {self.config.transit_z_m:.3f} m is below the saved scene's "
                f"model-top+80 mm advisory {minimum_transit_z:.3f} m. "
                "This is not full collision checking: inspect the real lift and horizontal route."
            )
            LOG.warning(transit_warning)
        run_dir = self.config.result_dir / "local_execution_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        event_path = run_dir / "events.jsonl"
        gripper_connected = False
        try:
            self.robot.connect()
            current = self.robot.actual_tcp_pose()
            # The current TCP is the actual start of the new stage plan. It need
            # not match the camera pose or the end of an earlier execution.
            self._validate_pose(current, "current_robot_pose")
            if stage_names[0] == "stopper":
                scene_message = (
                    "Confirm the gripper is empty; bottle, stopper, dropper and beaker still match the saved capture; "
                    "and lifting vertically from the current TCP to transit height is clear"
                )
            elif stage_names[0] == "bottle":
                scene_message = (
                    "Confirm the stopper has actually been removed and parked, the gripper is empty, "
                    "bottle/dropper/beaker remain at their saved positions, and vertical lift from the current TCP is clear"
                )
            else:
                scene_message = (
                    "Confirm the bottle is back at its saved position, the gripper is empty, "
                    "dropper/beaker have not moved, and vertical lift from the current TCP is clear"
                )
            if transit_warning:
                scene_message += "\nWARNING: " + transit_warning
            if scene.quality_report.get("warnings"):
                scene_message += "\nScene quality warnings: " + "; ".join(scene.quality_report["warnings"])
            confirm("scene", scene_message)
            self._assert_pose_unchanged(current, self.robot.actual_tcp_pose(), "during scene confirmation")
            for name in stage_names:
                current = self.robot.actual_tcp_pose()
                plan = self.plan_stage(name, scene, current)
                plan_path = run_dir / f"plan_{name}.json"
                self._write_json(plan_path, plan.as_dict())
                stage_message = f"Review live-start {name} plan at {plan_path} and physical clearances"
                if plan.warnings:
                    stage_message += "\nWARNING: " + "; ".join(plan.warnings)
                confirm(name, stage_message)
                self._assert_pose_unchanged(current, self.robot.actual_tcp_pose(), f"during {name} confirmation")
                if not gripper_connected:
                    self.gripper.connect()
                    gripper_connected = True
                self._run_stage(plan, confirm, event_path)
            self._write_json(run_dir / "completed.json", {
                "stages": list(stage_names),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "final_tcp_pose": self.robot.actual_tcp_pose(),
                "source_job_id": scene.source_job_id,
            })
            return run_dir
        except BaseException as exc:
            self.robot.stop()
            self._write_json(run_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
            raise
        finally:
            if gripper_connected:
                self.gripper.disconnect()
            self.robot.disconnect()

    def _run_stage(self, plan: StagePlan, confirm: Callable[[str, str], None], event_path: Path) -> None:
        for step in plan.steps:
            record = {"time": datetime.now(timezone.utc).isoformat(), "stage": plan.name, "step": step.name, "status": "started"}
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            try:
                if step.kind == "moveL":
                    assert step.pose is not None
                    self.robot.move_linear(step.pose)
                elif step.kind == "gripper":
                    assert step.gripper_position is not None
                    self.gripper.set_position(step.gripper_position)
                elif step.kind == "wait":
                    time.sleep(float(step.seconds))
                elif step.kind == "checkpoint":
                    before = self.robot.actual_tcp_pose()
                    confirm(f"{plan.name}.{step.name}", f"Inspect {step.name}; continue only if safe")
                    self._assert_pose_unchanged(before, self.robot.actual_tcp_pose(), f"during {step.name}")
                else:
                    raise ValueError(f"Unknown step kind: {step.kind}")
                record["status"] = "completed"
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                raise
            finally:
                record["time_finished"] = datetime.now(timezone.utc).isoformat()
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

    def _assert_pose_unchanged(self, before: Sequence[float], after: Sequence[float], context: str) -> None:
        distance = float(np.linalg.norm(np.asarray(before[:3]) - np.asarray(after[:3])))
        relative = rotvec_to_matrix(before[3:]).T @ rotvec_to_matrix(after[3:])
        angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        if distance > self.config.base.robot.position_tolerance_m or angle > self.config.orientation_tolerance_rad:
            raise RuntimeError(f"Robot pose changed {context}: {distance:.3f} m, {angle:.3f} rad.")
