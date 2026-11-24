from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .geometry import rotvec_to_matrix
from .hardware import DHGripper, URRobot
from .three_stage_config import ThreeStageConfig
from .three_stage_result import SceneGeometry, _read_ply_xyz, load_scene
from .target_geometry import sample_opening_disk, unit_axis
from .dropper_instances import LEGACY_CHEMISTRY_STAGES

LOG = logging.getLogger(__name__)
CHEMISTRY_STAGES = LEGACY_CHEMISTRY_STAGES  # historical import; runtime order is config.chemistry_stages
STAGES = (*CHEMISTRY_STAGES, "dropper_small", "dropper_large", "dropper_1", "dropper_2", "stirring")


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
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [step.as_dict() for step in self.steps],
            "warnings": list(self.warnings),
            "metadata": self.metadata,
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


def _bottle_upright_rotation(
    bottle_up: Sequence[float], pour_direction: Sequence[float], closing_axis_tool: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align jaw closing with the tilt axis and keep the pour direction between the fingers."""
    up = np.asarray(bottle_up, dtype=float)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-9:
        raise ValueError("Bottle axis cannot be zero when choosing pour yaw.")
    up /= up_norm
    pour = np.asarray(pour_direction, dtype=float)
    pour -= up * float(pour @ up)
    pour_norm = float(np.linalg.norm(pour))
    if pour_norm < 1e-9:
        raise ValueError("Bottle-to-beaker direction is parallel to the bottle axis; cannot choose pour yaw.")
    pour /= pour_norm

    closing_world = np.cross(up, pour)
    closing_world /= np.linalg.norm(closing_world)
    closing_tool = np.asarray(closing_axis_tool, dtype=float)
    closing_norm = float(np.linalg.norm(closing_tool))
    if closing_norm < 1e-9:
        raise ValueError("Gripper closing axis cannot be zero when choosing pour yaw.")
    closing_tool /= closing_norm
    tool_z = np.array([0.0, 0.0, 1.0])
    pour_tool = np.cross(tool_z, closing_tool)
    pour_tool /= np.linalg.norm(pour_tool)

    # Both bases are right-handed: closing x pour = tool +Z / world down.
    tool_basis = np.column_stack((closing_tool, pour_tool, tool_z))
    world_basis = np.column_stack((closing_world, pour, -up))
    rotation = world_basis @ tool_basis.T
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6,
    ):
        raise ValueError("Dynamic bottle orientation is not a valid rotation.")
    return rotation, pour, closing_world


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
            if self.config.dropper_count == 2:
                raise ValueError("In dual mode 'dropper' selects both operations; use plan_sequence or --stage dropper.")
            plan = self._plan_dropper(scene, current_pose)
        elif name in ("dropper_small", "dropper_large"):
            plan = self._plan_dropper(scene, current_pose, role=name.removeprefix("dropper_"))
        else:
            raise ValueError(f"Unknown stage: {name}")
        warnings = self._validate_plan(plan, current_pose, scene)
        plan = replace(plan, warnings=(*plan.warnings, *warnings))
        for warning in plan.warnings:
            LOG.warning("%s planning advisory: %s", name, warning)
        return plan

    def plan_all(self, scene: SceneGeometry, initial_pose: Sequence[float]) -> tuple[StagePlan, ...]:
        return self.plan_sequence(self.config.chemistry_stages, scene, initial_pose)

    def plan_sequence(self, names: Sequence[str], scene: SceneGeometry, initial_pose: Sequence[float]) -> tuple[StagePlan, ...]:
        pose = tuple(initial_pose)
        stages: list[StagePlan] = []
        for name in names:
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

    def _plan_bottle(
        self, scene: SceneGeometry, current_pose: Sequence[float], orientation_branch: int | None = None,
    ) -> StagePlan:
        cfg = self.config.bottle
        warnings: list[str] = []
        opening = scene.quality_report.get("beaker_opening", {})
        receiving_radius = float(opening.get("receiving_radius_m", scene.beaker_inner_radius_estimate_m - 0.003))
        if receiving_radius <= 0:
            raise ValueError("Beaker has no positive receiving radius after edge margin.")
        side_offset = min(cfg.pour_side_offset_m, receiving_radius)
        if cfg.pour_side_offset_m > receiving_radius:
            warnings.append(
                f"Pour side offset {cfg.pour_side_offset_m:.4f} m was limited to "
                f"the inset receiving opening radius {receiving_radius:.4f} m."
            )
        up = scene.bottle_axis_out
        grasp = scene.bottle_body_grasp + up * cfg.grasp_offset_along_axis_m
        approach = grasp + up * cfg.approach_m
        mouth = scene.stopper_grasp + up * cfg.stopper_center_to_mouth_m

        from_beaker_to_bottle = grasp[:2] - scene.beaker_mouth_center[:2]
        bottle_beaker_distance = float(np.linalg.norm(from_beaker_to_bottle))
        if bottle_beaker_distance < 1e-9:
            raise ValueError("Bottle and beaker have the same XY center; cannot choose a pour direction.")
        from_beaker_to_bottle /= bottle_beaker_distance
        toward_beaker_xy = -from_beaker_to_bottle
        upright_rotation, toward_beaker, tilt_axis = _bottle_upright_rotation(
            up, np.array([*toward_beaker_xy, 0.0]), cfg.closing_axis_tool,
        )
        grasp_rotvec = cfg.grasp_orientation_rotvec
        grasp_rotation = rotvec_to_matrix(grasp_rotvec)
        # Parallel jaws define an unoriented line. Both yaw branches preserve
        # that line and world-down tool Z; the signed pour axis stays unchanged.
        candidates = [upright_rotation, _axis_angle_rotation(up, math.pi) @ upright_rotation]
        angles = [float(np.linalg.norm(_rotvec(grasp_rotation.T @ rotation))) for rotation in candidates]
        branch = int(np.argmin(angles)) if orientation_branch is None else orientation_branch
        if branch not in (0, 1):
            raise ValueError("Bottle orientation branch must be 0 or 1.")
        upright_rotation = candidates[branch]
        upright_rotvec = _rotvec(upright_rotation)
        if cfg.keep_grasp_orientation_until_tilt:
            # Keep the proven grasp orientation through approach, grasp, lift,
            # and transfer. Start the actual pour as a gradual tilt from this
            # orientation instead of issuing a large standalone wrist yaw.
            upright_rotation = rotvec_to_matrix(grasp_rotvec)
            upright_rotvec = grasp_rotvec
            tilt_axis = np.cross(up / np.linalg.norm(up), np.array([*toward_beaker_xy, 0.0]))
            tilt_axis = tilt_axis / np.linalg.norm(tilt_axis)
            branch = 0
        mouth_tool = upright_rotation.T @ (mouth - grasp)
        normal = unit_axis(opening.get("normal_world", [0, 0, 1]))
        side_direction = np.array([*from_beaker_to_bottle, 0.0])
        side_direction -= normal * float(side_direction @ normal)
        side_direction = unit_axis(side_direction)
        receiving_point = scene.beaker_mouth_center + side_direction * side_offset
        # Liquid falls vertically: retain the receiving point's XY while
        # adding world-Z clearance above its (possibly tilted) opening plane.
        high_mouth = receiving_point + np.array([0, 0, cfg.pretilt_mouth_height_m])
        low_mouth = receiving_point + np.array([0, 0, cfg.pour_mouth_clearance_m])

        def tcp_for_mouth(desired_mouth: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            return desired_mouth - rotation @ mouth_tool

        high_upright_tcp = tcp_for_mouth(high_mouth, upright_rotation)
        steps: list[Step] = [
            _grip("open_before_bottle", self.config.base.gripper.open_position),
            _move("raise_for_bottle", self._transit(current_pose[:3]), current_pose[3:]),
            _move("over_bottle", self._transit(approach), grasp_rotvec),
            _move("bottle_approach", approach, grasp_rotvec),
            _move("bottle_body_grasp", grasp, grasp_rotvec),
            _grip("close_on_bottle", cfg.gripper_position),
            _checkpoint("check_bottle_grip_before_lift"),
            _move("lift_bottle", grasp + up * cfg.lift_m, grasp_rotvec),
            _checkpoint("check_bottle_clear_of_table"),
            _move("bottle_to_transit", self._transit(grasp), grasp_rotvec),
        ]
        if not cfg.keep_grasp_orientation_until_tilt:
            grasp_rotation = rotvec_to_matrix(grasp_rotvec)
            relative_rotvec = _rotvec(grasp_rotation.T @ upright_rotation)
            high_grasp = self._transit(grasp)
            for index in range(1, cfg.orientation_transition_steps + 1):
                fraction = index / cfg.orientation_transition_steps
                transition_rotation = grasp_rotation @ rotvec_to_matrix(relative_rotvec * fraction)
                steps.append(_move(
                    f"rotate_bottle_at_transit_{index:02d}", high_grasp, _rotvec(transition_rotation),
                ))
        steps.extend((
            _move("bottle_over_beaker_transit", self._transit(high_upright_tcp), upright_rotvec),
            _move("bottle_high_upright", high_upright_tcp, upright_rotvec),
            _checkpoint("check_high_pour_clearance"),
        ))

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
            _move("bottle_return_transit", self._transit(high_upright_tcp), upright_rotvec),
            _move("over_bottle_original", self._transit(approach), upright_rotvec),
            _move("return_bottle_approach", approach, upright_rotvec),
            _move("replace_bottle", grasp, upright_rotvec),
            _grip("release_bottle", self.config.base.gripper.open_position),
            _move("retreat_from_bottle", approach, upright_rotvec),
        ))
        return StagePlan("bottle", tuple(steps), tuple(warnings), metadata={
            "orientation_branch": branch,
            "orientation_transition_deg": 0.0 if cfg.keep_grasp_orientation_until_tilt else math.degrees(angles[branch]),
            "equivalent_branch_transition_deg": [math.degrees(angle) for angle in angles],
            "orientation_selection": "nearest_rotation_offline; joint-path selection before live bottle motion",
        })

    def _bottle_joint_preflight(self, plan: StagePlan, current_pose, start_joints) -> dict:
        """Check the whole Cartesian route with continuous, seeded controller IK.

        Each segment is sampled at <=10 mm and <=5 degrees. A large raw joint
        jump is rejected rather than hidden by wrapping angles modulo 2*pi.
        """
        previous_pose = np.asarray(current_pose, dtype=float)
        joints = np.asarray(start_joints, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise RuntimeError("Bottle preflight requires six finite live joint positions.")
        if not self.robot.joints_within_safety_limits(joints.tolist()):
            raise RuntimeError("Current joints fail controller safety limits.")
        travel, largest_jump, samples = 0.0, 0.0, 0
        for step in plan.steps:
            if step.pose is None:
                continue
            target = np.asarray(step.pose, dtype=float)
            start_rotation = rotvec_to_matrix(previous_pose[3:])
            relative = _rotvec(start_rotation.T @ rotvec_to_matrix(target[3:]))
            count = max(1, math.ceil(np.linalg.norm(target[:3]-previous_pose[:3])/.01),
                        math.ceil(np.linalg.norm(relative)/math.radians(5)))
            for index in range(1, count+1):
                fraction = index/count
                xyz = previous_pose[:3]*(1-fraction)+target[:3]*fraction
                rotation = start_rotation @ rotvec_to_matrix(relative*fraction)
                pose = [*xyz.tolist(), *_rotvec(rotation).tolist()]
                try:
                    following = np.asarray(self.robot.inverse_kinematics(pose, qnear=joints.tolist()), dtype=float)
                except Exception as exc:
                    raise RuntimeError(f"{step.name}: controller IK failed at sample {index}/{count}: {exc}") from exc
                if following.shape != (6,) or not np.all(np.isfinite(following)):
                    raise RuntimeError(f"{step.name}: controller returned invalid IK.")
                if not self.robot.joints_within_safety_limits(following.tolist()):
                    raise RuntimeError(f"{step.name}: IK violates controller joint/safety limits at sample {index}/{count}.")
                delta = np.abs(following-joints)
                jump = float(delta.max())
                if jump > math.radians(45):
                    raise RuntimeError(f"{step.name}: discontinuous IK joint jump {math.degrees(jump):.1f} deg.")
                largest_jump = max(largest_jump, jump)
                travel += float(delta.sum())
                joints = following
                samples += 1
            previous_pose = target
        return {"samples": samples, "total_joint_travel_rad": travel,
                "largest_sample_joint_change_deg": math.degrees(largest_jump),
                "start_joints_rad": list(start_joints), "final_joints_rad": joints.tolist()}

    def _select_live_bottle_plan(self, scene, current_pose) -> StagePlan:
        """Compare both valid jaw orientations before approaching/gripping."""
        start_joints = self.robot.actual_joints()
        checked, rejected = [], []
        for branch in ((0,) if self.config.bottle.keep_grasp_orientation_until_tilt else (0, 1)):
            try:
                plan = self._plan_bottle(scene, current_pose, orientation_branch=branch)
                warnings = self._validate_plan(plan, current_pose, scene)
                plan = replace(plan, warnings=(*plan.warnings, *warnings))
                audit = self._bottle_joint_preflight(plan, current_pose, start_joints)
                checked.append(replace(plan, metadata={**plan.metadata, "joint_preflight": audit}))
            except (RuntimeError, ValueError) as exc:
                rejected.append({"branch": branch, "reason": str(exc)})
        if not checked:
            raise RuntimeError(f"No continuous bottle route passes controller IK and joint limits; no bottle motion started. {rejected}")
        best = min(checked, key=lambda plan: (
            plan.metadata["joint_preflight"]["total_joint_travel_rad"], plan.metadata["orientation_transition_deg"],
        ))
        metadata = {**best.metadata, "orientation_selection": "minimum_joint_travel_among_controller_valid_routes",
                    "rejected_orientation_branches": rejected,
                    "valid_orientation_branches": [{"branch": plan.metadata["orientation_branch"],
                        "transition_deg": plan.metadata["orientation_transition_deg"],
                        "joint_travel_rad": plan.metadata["joint_preflight"]["total_joint_travel_rad"]} for plan in checked]}
        LOG.info("Bottle orientation: branch=%s, transition=%.1f deg; controller-valid alternatives=%s",
                 metadata["orientation_branch"], metadata["orientation_transition_deg"], metadata["valid_orientation_branches"])
        return replace(best, metadata=metadata)

    def _plan_dropper(self, scene: SceneGeometry, current_pose: Sequence[float], role: str = "single") -> StagePlan:
        cfg = self.config.dropper_motion(role)
        bulb = scene.dropper_grasp(role) + np.array([0.0, 0.0, cfg.grasp_offset_z_m])
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
        name = "dropper" if role == "single" else f"dropper_{role}"
        instance_id = scene.dropper_instance_ids.get(role, "dropper")
        return StagePlan(name, tuple(steps), metadata={
            "dropper_role": role, "source_object_id": instance_id,
            "reagent_role": {"small": "phenolphthalein", "large": "acid"}.get(role, "legacy"),
            "size_metric": scene.dropper_size_ranking.get("models", {}).get(instance_id),
            "motion_parameters": asdict(cfg),
            "dose_control": "fixed_squeeze_cycles_no_color_feedback",
            "return_grasp_base_m": bulb.tolist(),
        })

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
        artifact = scene.model_artifacts.get("wide_mouth_bottle", "models/wide_mouth_bottle_world.ply")
        cloud = _read_ply_xyz(self.config.result_dir / artifact)
        grasp = scene.bottle_body_grasp + scene.bottle_axis_out * self.config.bottle.grasp_offset_along_axis_m
        grasp_step = next(step for step in plan.steps if step.name == "bottle_body_grasp")
        assert grasp_step.pose is not None
        rotation_at_grasp = rotvec_to_matrix(grasp_step.pose[3:])
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
        filename = "local_three_stage_plan_original_coordinates.json" if self.config.table_constraint_enabled else "local_three_stage_plan.json"
        output = self.config.result_dir / filename
        target_artifacts = {}
        opening = scene.quality_report.get("beaker_opening")
        if opening:
            disk_path = self.config.result_dir / "local_beaker_opening_world.ply"
            disk = sample_opening_disk(opening).astype("<f4")
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                "comment SYNTHETIC receiving free-space disk; not measured surface\n"
                f"element vertex {len(disk)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
            )
            disk_path.write_bytes(header.encode("ascii") + disk.tobytes())
            opening_path = self.config.result_dir / "local_beaker_opening.json"
            self._write_json(opening_path, opening)
            target_artifacts = {"synthetic_beaker_opening_ply": str(disk_path), "beaker_opening_json": str(opening_path)}
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execute": False,
            "source_capture": str(self.config.capture_dir),
            "source_result": str(self.config.result_dir / "result.json"),
            "grasp_strategy": "shared axial-density-mode local weighted XYZ centers; top-down lateral closure",
            "target_artifacts": target_artifacts,
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
        stage_names = tuple(name for stage in stage_names for name in self.config.select_chemistry_stages(stage))
        if stage_names != self.config.chemistry_stages and stage_names != ("dropper_small", "dropper_large") and not (
            len(stage_names) == 1 and stage_names[0] in self.config.chemistry_stages
        ):
            raise ValueError("Select all chemistry stages in order, the ordered pair of droppers, or one explicit stage.")
        # A low transit height can turn the first horizontal move from an
        # arbitrary start pose into a sweep through the saved scene. This is
        # only an object-height proxy; it is not full collision checking.
        tallest_model_z = max(
            float(np.max(_read_ply_xyz(self.config.result_dir / artifact)[:, 2]))
            for artifact in (scene.model_artifacts or {name: f"models/{name}_world.ply" for name in self.config.object_names}).values()
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
                    "Confirm the gripper is empty; bottle, stopper, all droppers and beaker still match the saved capture; "
                    "and lifting vertically from the current TCP to transit height is clear"
                )
            elif stage_names[0] == "bottle":
                scene_message = (
                    "Confirm the stopper has actually been removed and parked, the gripper is empty, "
                    "bottle/all droppers/beaker remain at their saved positions, and vertical lift from the current TCP is clear"
                )
            else:
                scene_message = (
                    "Confirm the bottle is back at its saved position, the gripper is empty, "
                    "all droppers/beaker have not moved, the previous pipette is returned, and vertical lift from the current TCP is clear"
                )
            if transit_warning:
                scene_message += "\nWARNING: " + transit_warning
            if scene.quality_report.get("warnings"):
                scene_message += "\nScene quality warnings: " + "; ".join(scene.quality_report["warnings"])
            confirm("scene", scene_message)
            self._assert_pose_unchanged(current, self.robot.actual_tcp_pose(), "during scene confirmation")
            for name in stage_names:
                current = self.robot.actual_tcp_pose()
                plan = self._select_live_bottle_plan(scene, current) if name == "bottle" else self.plan_stage(name, scene, current)
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
                "dropper_instance_ids": scene.dropper_instance_ids,
                "dropper_size_ranking": scene.dropper_size_ranking,
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
            if plan.metadata.get("source_object_id"):
                record["source_object_id"] = plan.metadata["source_object_id"]
                record["dropper_role"] = plan.metadata["dropper_role"]
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            try:
                if step.kind == "moveL":
                    assert step.pose is not None
                    if plan.name == "bottle":
                        record["joints_before_rad"] = self.robot.actual_joints()
                    self.robot.move_linear(step.pose)
                    if plan.name == "bottle":
                        record["joints_after_rad"] = self.robot.actual_joints()
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
