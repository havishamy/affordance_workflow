from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import AppConfig


def _values(raw: Any, count: int, name: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw)
    if len(values) != count or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain {count} finite values.")
    return values


def _optional_float(raw: Any, name: str) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite when configured.")
    return value


def _optional_values(raw: Any, count: int, name: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return _values(raw, count, name)


@dataclass(frozen=True)
class ResultQualityConfig:
    minimum_region_points: int
    minimum_silhouette_iou: float
    minimum_bottle_axis_world_z: float
    minimum_stopper_body_separation_m: float
    maximum_stopper_rms_m: float
    maximum_body_rms_m: float
    maximum_beaker_mouth_rms_m: float
    maximum_bulb_rms_m: float
    minimum_beaker_radius_m: float
    maximum_beaker_radius_m: float
    maximum_beaker_circle_residual_m: float
    minimum_beaker_arc_rad: float


@dataclass(frozen=True)
class StopperStageConfig:
    orientation_rotvec: tuple[float, float, float]
    approach_m: float
    pull_distance_m: float
    gripper_position: int
    place_xyz_base_m: tuple[float, float, float]
    place_approach_m: float
    settle_s: float


@dataclass(frozen=True)
class BottleStageConfig:
    orientation_rotvec: tuple[float, float, float]
    grasp_offset_along_axis_m: float
    approach_m: float
    lift_m: float
    gripper_position: int
    stopper_center_to_mouth_m: float
    pour_side_offset_m: float
    pour_mouth_clearance_m: float
    pretilt_mouth_height_m: float
    tilt_angle_deg: float
    tilt_steps: int
    lower_steps: int
    hold_s: float
    restore_settle_s: float


@dataclass(frozen=True)
class DropperStageConfig:
    orientation_rotvec: tuple[float, float, float]
    grasp_offset_z_m: float
    approach_m: float
    extraction_m: float
    hold_position: int
    squeeze_position: int
    tip_below_bulb_m: float
    tip_clearance_above_beaker_m: float
    squeeze_cycles: int
    squeeze_hold_s: float
    settle_s: float


@dataclass(frozen=True)
class ObservationConfig:
    enabled: int
    reference_capture_info: Path
    settle_s: float
    reuse_latest_result: bool
    latest_result_manifest: Path


@dataclass(frozen=True)
class StirringPerceptionConfig:
    rod_radius_m: float
    grasp_inset_m: float
    min_length_m: float
    max_length_m: float
    min_segmentation_confidence: float
    min_mask_elongation: float
    min_table_plane_inlier_fraction: float
    max_table_plane_rmse_m: float
    max_endpoint_uncertainty_m: float
    expected_rod_length_m: float
    rod_length_tolerance_m: float
    latest_result_manifest: Path


@dataclass(frozen=True)
class StirringMotionConfig:
    validated: bool
    pregrasp_height_m: float | None
    lift_height_m: float | None
    preinsert_clearance_m: float | None
    target_depth_below_mouth_m: float | None
    stir_radius_m: float | None
    stir_cycles: int
    stir_points_per_cycle: int
    stir_speed_mps: float | None
    beaker_safe_inner_radius_m: float | None
    bottom_clearance_m: float | None
    rod_grasp_position: int
    rod_grasp_force: int
    rod_grasp_center_tool_m: tuple[float, float, float] | None
    closing_axis_tool: tuple[float, float, float]
    orientation_steps: int
    settle_s: float


@dataclass(frozen=True)
class ThreeStageConfig:
    base: AppConfig
    capture_dir: Path
    result_dir: Path
    physical_execution_validated: bool
    transit_z_m: float
    orientation_tolerance_rad: float
    observation: ObservationConfig
    quality: ResultQualityConfig
    stopper: StopperStageConfig
    bottle: BottleStageConfig
    dropper: DropperStageConfig
    stirring_perception: StirringPerceptionConfig
    stirring_motion: StirringMotionConfig

    @classmethod
    def load(cls, path: str | Path) -> "ThreeStageConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Four-stage configuration must contain a mapping (legacy three_stage filename is supported).")
        base_path = Path(raw.get("base_config", "config.yaml")).expanduser()
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = AppConfig.load(base_path)
        robot_overrides = raw.get("robot", {})
        base = replace(
            base,
            robot=replace(
                base.robot,
                speed=float(robot_overrides.get("speed", base.robot.speed)),
                acceleration=float(robot_overrides.get("acceleration", base.robot.acceleration)),
            ),
        )

        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            return candidate.resolve()

        quality = raw["quality"]
        observation = raw.get("observation", {})
        stopper = raw["stopper"]
        bottle = raw["bottle"]
        dropper = raw["dropper"]
        stirring_perception = raw["stirring_perception"]
        stirring_motion = raw["stirring_motion"]
        result = cls(
            base=base,
            capture_dir=resolve(raw["capture_dir"]),
            result_dir=resolve(raw["result_dir"]),
            physical_execution_validated=bool(raw.get("physical_execution_validated", False)),
            transit_z_m=float(raw.get("transit_z_m", base.motion.transit_z_m)),
            orientation_tolerance_rad=float(raw.get("orientation_tolerance_rad", 0.05)),
            observation=ObservationConfig(
                enabled=int(observation.get("enabled", 0)),
                reference_capture_info=resolve(
                    observation.get("reference_capture_info", str(resolve(raw["capture_dir"]) / "capture_info.json"))
                ),
                settle_s=float(observation.get("settle_s", 0.8)),
                reuse_latest_result=bool(observation.get("reuse_latest_result", True)),
                latest_result_manifest=resolve(
                    observation.get("latest_result_manifest", str(base.output_dir / "three_stage_latest_result.json"))
                ),
            ),
            quality=ResultQualityConfig(
                minimum_region_points=int(quality.get("minimum_region_points", 100)),
                minimum_silhouette_iou=float(quality.get("minimum_silhouette_iou", 0.70)),
                minimum_bottle_axis_world_z=float(quality.get("minimum_bottle_axis_world_z", 0.85)),
                minimum_stopper_body_separation_m=float(quality.get("minimum_stopper_body_separation_m", 0.025)),
                maximum_stopper_rms_m=float(quality.get("maximum_stopper_rms_m", 0.025)),
                maximum_body_rms_m=float(quality.get("maximum_body_rms_m", 0.045)),
                maximum_beaker_mouth_rms_m=float(quality.get("maximum_beaker_mouth_rms_m", 0.025)),
                maximum_bulb_rms_m=float(quality.get("maximum_bulb_rms_m", 0.020)),
                minimum_beaker_radius_m=float(quality.get("minimum_beaker_radius_m", 0.015)),
                maximum_beaker_radius_m=float(quality.get("maximum_beaker_radius_m", 0.060)),
                maximum_beaker_circle_residual_m=float(quality.get("maximum_beaker_circle_residual_m", 0.003)),
                minimum_beaker_arc_rad=float(quality.get("minimum_beaker_arc_rad", 1.5)),
            ),
            stopper=StopperStageConfig(
                orientation_rotvec=_values(stopper["orientation_rotvec"], 3, "stopper.orientation_rotvec"),
                approach_m=float(stopper.get("approach_m", 0.08)),
                pull_distance_m=float(stopper.get("pull_distance_m", 0.10)),
                gripper_position=int(stopper["gripper_position"]),
                place_xyz_base_m=_values(stopper["place_xyz_base_m"], 3, "stopper.place_xyz_base_m"),
                place_approach_m=float(stopper.get("place_approach_m", 0.10)),
                settle_s=float(stopper.get("settle_s", 0.8)),
            ),
            bottle=BottleStageConfig(
                orientation_rotvec=_values(bottle["orientation_rotvec"], 3, "bottle.orientation_rotvec"),
                grasp_offset_along_axis_m=float(bottle.get("grasp_offset_along_axis_m", 0.01)),
                approach_m=float(bottle.get("approach_m", 0.10)),
                lift_m=float(bottle.get("lift_m", 0.15)),
                gripper_position=int(bottle["gripper_position"]),
                stopper_center_to_mouth_m=float(bottle.get("stopper_center_to_mouth_m", -0.01)),
                pour_side_offset_m=float(bottle.get("pour_side_offset_m", 0.025)),
                pour_mouth_clearance_m=float(bottle.get("pour_mouth_clearance_m", 0.06)),
                pretilt_mouth_height_m=float(bottle.get("pretilt_mouth_height_m", 0.18)),
                tilt_angle_deg=float(bottle.get("tilt_angle_deg", 100.0)),
                tilt_steps=int(bottle.get("tilt_steps", 12)),
                lower_steps=int(bottle.get("lower_steps", 4)),
                hold_s=float(bottle.get("hold_s", 2.0)),
                restore_settle_s=float(bottle.get("restore_settle_s", 1.0)),
            ),
            dropper=DropperStageConfig(
                orientation_rotvec=_values(dropper["orientation_rotvec"], 3, "dropper.orientation_rotvec"),
                grasp_offset_z_m=float(dropper.get("grasp_offset_z_m", 0.0)),
                approach_m=float(dropper.get("approach_m", 0.08)),
                extraction_m=float(dropper.get("extraction_m", 0.12)),
                hold_position=int(dropper["hold_position"]),
                squeeze_position=int(dropper["squeeze_position"]),
                tip_below_bulb_m=float(dropper["tip_below_bulb_m"]),
                tip_clearance_above_beaker_m=float(dropper["tip_clearance_above_beaker_m"]),
                squeeze_cycles=int(dropper.get("squeeze_cycles", 1)),
                squeeze_hold_s=float(dropper.get("squeeze_hold_s", 0.5)),
                settle_s=float(dropper.get("settle_s", 0.8)),
            ),
            stirring_perception=StirringPerceptionConfig(
                rod_radius_m=float(stirring_perception.get("rod_radius_m", 0.0025)),
                grasp_inset_m=float(stirring_perception.get("grasp_inset_m", 0.025)),
                min_length_m=float(stirring_perception.get("min_length_m", 0.08)),
                max_length_m=float(stirring_perception.get("max_length_m", 0.40)),
                min_segmentation_confidence=float(stirring_perception.get("min_segmentation_confidence", 0.50)),
                min_mask_elongation=float(stirring_perception.get("min_mask_elongation", 6.0)),
                min_table_plane_inlier_fraction=float(
                    stirring_perception.get("min_table_plane_inlier_fraction", 0.75)
                ),
                max_table_plane_rmse_m=float(stirring_perception.get("max_table_plane_rmse_m", 0.004)),
                max_endpoint_uncertainty_m=float(
                    stirring_perception.get("max_endpoint_uncertainty_m", 0.010)
                ),
                expected_rod_length_m=float(stirring_perception["expected_rod_length_m"]),
                rod_length_tolerance_m=float(stirring_perception["rod_length_tolerance_m"]),
                latest_result_manifest=resolve(
                    stirring_perception.get(
                        "latest_result_manifest", str(base.output_dir / "stirring_latest_result.json")
                    )
                ),
            ),
            stirring_motion=StirringMotionConfig(
                validated=bool(stirring_motion.get("validated", False)),
                pregrasp_height_m=_optional_float(
                    stirring_motion.get("pregrasp_height_m"), "stirring_motion.pregrasp_height_m"
                ),
                lift_height_m=_optional_float(
                    stirring_motion.get("lift_height_m"), "stirring_motion.lift_height_m"
                ),
                preinsert_clearance_m=_optional_float(
                    stirring_motion.get("preinsert_clearance_m"),
                    "stirring_motion.preinsert_clearance_m",
                ),
                target_depth_below_mouth_m=_optional_float(
                    stirring_motion.get("target_depth_below_mouth_m"),
                    "stirring_motion.target_depth_below_mouth_m",
                ),
                stir_radius_m=_optional_float(
                    stirring_motion.get("stir_radius_m"), "stirring_motion.stir_radius_m"
                ),
                stir_cycles=int(stirring_motion.get("stir_cycles", 1)),
                stir_points_per_cycle=int(stirring_motion.get("stir_points_per_cycle", 24)),
                stir_speed_mps=_optional_float(
                    stirring_motion.get("stir_speed_mps"), "stirring_motion.stir_speed_mps"
                ),
                beaker_safe_inner_radius_m=_optional_float(
                    stirring_motion.get("beaker_safe_inner_radius_m"),
                    "stirring_motion.beaker_safe_inner_radius_m",
                ),
                bottom_clearance_m=_optional_float(
                    stirring_motion.get("bottom_clearance_m"), "stirring_motion.bottom_clearance_m"
                ),
                rod_grasp_position=int(stirring_motion.get("rod_grasp_position", 50)),
                rod_grasp_force=int(stirring_motion.get("rod_grasp_force", base.gripper.force)),
                rod_grasp_center_tool_m=_optional_values(
                    stirring_motion.get("rod_grasp_center_tool_m"),
                    3,
                    "stirring_motion.rod_grasp_center_tool_m",
                ),
                closing_axis_tool=_values(
                    stirring_motion.get("closing_axis_tool", [1, 0, 0]),
                    3,
                    "stirring_motion.closing_axis_tool",
                ),
                orientation_steps=int(stirring_motion.get("orientation_steps", 8)),
                settle_s=float(stirring_motion.get("settle_s", 0.8)),
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.capture_dir.is_dir() or not self.result_dir.is_dir():
            raise ValueError("Configured capture_dir and result_dir must exist.")
        if not self.base.motion.workspace_min[2] <= self.transit_z_m <= self.base.motion.workspace_max[2]:
            raise ValueError("transit_z_m must be inside the base workspace.")
        if self.orientation_tolerance_rad <= 0:
            raise ValueError("orientation_tolerance_rad must be positive.")
        if self.observation.enabled not in (0, 1):
            raise ValueError("observation.enabled must be 0 or 1.")
        if not self.observation.reference_capture_info.is_file():
            raise ValueError("observation.reference_capture_info does not exist.")
        if not math.isfinite(self.observation.settle_s) or self.observation.settle_s < 0:
            raise ValueError("observation.settle_s must be finite and nonnegative.")
        if not 0 < self.bottle.tilt_angle_deg < 150:
            raise ValueError("bottle.tilt_angle_deg must be between 0 and 150 degrees.")
        if self.bottle.tilt_steps < 3 or self.bottle.lower_steps < 2:
            raise ValueError("Bottle tilt/lower trajectories need at least 3/2 steps.")
        if self.bottle.pretilt_mouth_height_m <= self.bottle.pour_mouth_clearance_m:
            raise ValueError("Pre-tilt height must exceed final pour-mouth clearance.")
        if self.dropper.squeeze_cycles < 1:
            raise ValueError("dropper.squeeze_cycles must be at least one.")
        if not math.isfinite(self.dropper.grasp_offset_z_m):
            raise ValueError("dropper.grasp_offset_z_m must be finite.")
        if self.dropper.tip_below_bulb_m + self.dropper.grasp_offset_z_m <= 0:
            raise ValueError("Dropper tip must remain below the shifted grasp center.")
        if self.base.robot.speed > 0.2 or self.base.robot.acceleration > 0.2:
            raise ValueError("Four-stage experiment caps RTDE speed/acceleration at 0.2.")
        for name, value in (
            ("stopper.gripper_position", self.stopper.gripper_position),
            ("bottle.gripper_position", self.bottle.gripper_position),
            ("dropper.hold_position", self.dropper.hold_position),
            ("dropper.squeeze_position", self.dropper.squeeze_position),
        ):
            if not 0 <= value <= 1000:
                raise ValueError(f"{name} must be within [0, 1000].")
        opening = self.base.gripper.open_position
        if not all(position < opening for position in (
            self.stopper.gripper_position, self.bottle.gripper_position, self.dropper.hold_position,
        )):
            raise ValueError("All grasp positions must be more closed than the configured open position.")
        if self.dropper.squeeze_position >= self.dropper.hold_position:
            raise ValueError("dropper.squeeze_position must be more closed than hold_position.")
        for name, value in (
            ("robot.speed", self.base.robot.speed),
            ("robot.acceleration", self.base.robot.acceleration),
            ("stopper.approach_m", self.stopper.approach_m),
            ("stopper.pull_distance_m", self.stopper.pull_distance_m),
            ("bottle.approach_m", self.bottle.approach_m),
            ("bottle.lift_m", self.bottle.lift_m),
            ("dropper.approach_m", self.dropper.approach_m),
            ("dropper.extraction_m", self.dropper.extraction_m),
            ("dropper.tip_below_bulb_m", self.dropper.tip_below_bulb_m),
            ("dropper.tip_clearance_above_beaker_m", self.dropper.tip_clearance_above_beaker_m),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        perception = self.stirring_perception
        positive_perception = (
            perception.rod_radius_m, perception.grasp_inset_m, perception.min_length_m,
            perception.max_length_m, perception.min_mask_elongation,
            perception.max_table_plane_rmse_m, perception.max_endpoint_uncertainty_m,
            perception.expected_rod_length_m, perception.rod_length_tolerance_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_perception):
            raise ValueError("Stirring perception distances and quality thresholds must be finite and positive.")
        if perception.min_length_m >= perception.max_length_m:
            raise ValueError("stirring_perception min_length_m must be below max_length_m.")
        for name, value in (
            ("min_segmentation_confidence", perception.min_segmentation_confidence),
            ("min_table_plane_inlier_fraction", perception.min_table_plane_inlier_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"stirring_perception.{name} must be in [0, 1].")

        motion = self.stirring_motion
        if not 0 <= motion.rod_grasp_position <= 1000:
            raise ValueError("stirring_motion.rod_grasp_position must be within [0, 1000].")
        if not 0 <= motion.rod_grasp_force <= 100:
            raise ValueError("stirring_motion.rod_grasp_force must be within [0, 100].")
        if motion.rod_grasp_position >= self.base.gripper.open_position:
            raise ValueError("Stirring grasp position must be more closed than open_position.")
        closing_axis = np.asarray(motion.closing_axis_tool)
        if not math.isclose(float(np.linalg.norm(closing_axis)), 1.0, abs_tol=1e-6):
            raise ValueError("stirring_motion.closing_axis_tool must be a unit vector.")
        if abs(float(closing_axis[2])) > 1e-6:
            raise ValueError("stirring_motion.closing_axis_tool must be perpendicular to tool Z.")
        if motion.orientation_steps < 2 or motion.stir_cycles < 1 or motion.stir_points_per_cycle < 8:
            raise ValueError("Stirring orientation/cycle sampling is too small.")
        if not math.isfinite(motion.settle_s) or motion.settle_s < 0:
            raise ValueError("stirring_motion.settle_s must be finite and nonnegative.")
        measured = {
            "pregrasp_height_m": motion.pregrasp_height_m,
            "lift_height_m": motion.lift_height_m,
            "preinsert_clearance_m": motion.preinsert_clearance_m,
            "target_depth_below_mouth_m": motion.target_depth_below_mouth_m,
            "stir_radius_m": motion.stir_radius_m,
            "stir_speed_mps": motion.stir_speed_mps,
            "beaker_safe_inner_radius_m": motion.beaker_safe_inner_radius_m,
            "bottom_clearance_m": motion.bottom_clearance_m,
        }
        if motion.validated:
            missing = [name for name, value in measured.items() if value is None]
            if motion.rod_grasp_center_tool_m is None:
                missing.append("rod_grasp_center_tool_m")
            if missing:
                raise ValueError(f"Validated stirring motion is missing measured values: {missing}")
        for name, value in measured.items():
            if value is not None and value <= 0:
                raise ValueError(f"stirring_motion.{name} must be positive when configured.")
        if all(value is not None for value in (
            motion.stir_radius_m, motion.beaker_safe_inner_radius_m,
        )) and (
            motion.stir_radius_m + perception.max_endpoint_uncertainty_m + perception.rod_radius_m
            >= motion.beaker_safe_inner_radius_m
        ):
            raise ValueError("Configured stirring radius violates the beaker inner-radius uncertainty margin.")
