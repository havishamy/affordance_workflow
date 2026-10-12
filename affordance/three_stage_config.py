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

    @classmethod
    def load(cls, path: str | Path) -> "ThreeStageConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Three-stage configuration must contain a mapping.")
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
            raise ValueError("Three-stage experiment caps RTDE speed/acceleration at 0.2.")
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
