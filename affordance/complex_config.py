from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import AppConfig


def _values(raw: Any, count: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in raw)
    if len(result) != count or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {count} finite values.")
    return result


@dataclass(frozen=True)
class StopperMotion:
    orientation_rotvec: tuple[float, float, float]
    grasp_offset_m: tuple[float, float, float]
    approach_m: float
    pull_distance_m: float
    parking_pose: tuple[float, ...]
    parking_approach_m: float
    gripper_position: int


@dataclass(frozen=True)
class BottleMotion:
    orientation_rotvec: tuple[float, float, float]
    grasp_offset_m: tuple[float, float, float]
    approach_m: float
    lift_m: float
    pour_target_offset_m: tuple[float, float, float]
    mouth_clearance_m: float
    tilt_axis_tool: tuple[float, float, float]
    tilt_angle_deg: float
    tilt_steps: int
    hold_s: float
    gripper_position: int


@dataclass(frozen=True)
class DropperMotion:
    orientation_rotvec: tuple[float, float, float]
    grasp_offset_m: tuple[float, float, float]
    approach_m: float
    lift_m: float
    target_offset_m: tuple[float, float, float]
    tip_clearance_m: float
    squeeze_cycles: int
    squeeze_hold_s: float
    hold_position: int
    squeeze_position: int


@dataclass(frozen=True)
class StirMotion:
    enabled: bool
    orientation_rotvec: tuple[float, float, float]
    grasp_offset_m: tuple[float, float, float]
    approach_m: float
    lift_m: float
    preinsert_clearance_m: float
    target_depth_below_mouth_m: float
    radius_m: float
    turns: int
    points_per_turn: int
    point_hold_s: float
    gripper_position: int


@dataclass(frozen=True)
class GeometryPriors:
    """Locally calibrated canonical-model geometry; not server predictions."""

    minimum_affordance_points: int
    bottle_model_id: str
    beaker_model_id: str
    dropper_model_id: str
    glass_rod_model_id: str
    bottle_axis_out_object: tuple[float, float, float]
    bottle_mouth_from_stopper_along_axis_m: float
    beaker_axis_out_world: tuple[float, float, float]
    beaker_inner_radius_m: float
    dropper_tip_anchor_object: tuple[float, float, float]
    glass_rod_grasp_end: str


@dataclass(frozen=True)
class ComplexDemoConfig:
    base: AppConfig
    api_path: str
    physical_accuracy_validated: bool
    stopper: StopperMotion
    bottle: BottleMotion
    dropper: DropperMotion
    stir: StirMotion
    geometry: GeometryPriors

    @classmethod
    def load(cls, path: str | Path) -> "ComplexDemoConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Complex demo configuration must contain a mapping.")

        base_path = Path(raw.get("base_config", "config.yaml")).expanduser()
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = AppConfig.load(base_path)
        stopper = raw["stopper"]
        bottle = raw["bottle"]
        dropper = raw["dropper"]
        stir = raw["stir"]
        geometry = raw["geometry_priors"]

        result = cls(
            base=base,
            api_path=str(raw.get("api_path", "/v1/complex-demo/jobs")),
            physical_accuracy_validated=bool(raw.get("physical_accuracy_validated", False)),
            stopper=StopperMotion(
                orientation_rotvec=_values(stopper["orientation_rotvec"], 3, "stopper.orientation_rotvec"),
                grasp_offset_m=_values(stopper.get("grasp_offset_m", [0, 0, 0]), 3, "stopper.grasp_offset_m"),
                approach_m=float(stopper.get("approach_m", 0.08)),
                pull_distance_m=float(stopper.get("pull_distance_m", 0.10)),
                parking_pose=_values(stopper["parking_pose"], 6, "stopper.parking_pose"),
                parking_approach_m=float(stopper.get("parking_approach_m", 0.08)),
                gripper_position=int(stopper.get("gripper_position", base.gripper.grasp_position)),
            ),
            bottle=BottleMotion(
                orientation_rotvec=_values(bottle["orientation_rotvec"], 3, "bottle.orientation_rotvec"),
                grasp_offset_m=_values(bottle.get("grasp_offset_m", [0, 0, 0]), 3, "bottle.grasp_offset_m"),
                approach_m=float(bottle.get("approach_m", 0.10)),
                lift_m=float(bottle.get("lift_m", 0.12)),
                pour_target_offset_m=_values(bottle.get("pour_target_offset_m", [0, 0, 0]), 3, "bottle.pour_target_offset_m"),
                mouth_clearance_m=float(bottle.get("mouth_clearance_m", 0.08)),
                tilt_axis_tool=_values(bottle.get("tilt_axis_tool", [0, 1, 0]), 3, "bottle.tilt_axis_tool"),
                tilt_angle_deg=float(bottle.get("tilt_angle_deg", 100.0)),
                tilt_steps=int(bottle.get("tilt_steps", 10)),
                hold_s=float(bottle.get("hold_s", 2.0)),
                gripper_position=int(bottle.get("gripper_position", base.gripper.grasp_position)),
            ),
            dropper=DropperMotion(
                orientation_rotvec=_values(dropper["orientation_rotvec"], 3, "dropper.orientation_rotvec"),
                grasp_offset_m=_values(dropper.get("grasp_offset_m", [0, 0, 0]), 3, "dropper.grasp_offset_m"),
                approach_m=float(dropper.get("approach_m", 0.08)),
                lift_m=float(dropper.get("lift_m", 0.10)),
                target_offset_m=_values(dropper.get("target_offset_m", [0, 0, 0]), 3, "dropper.target_offset_m"),
                tip_clearance_m=float(dropper.get("tip_clearance_m", 0.03)),
                squeeze_cycles=int(dropper.get("squeeze_cycles", 3)),
                squeeze_hold_s=float(dropper.get("squeeze_hold_s", 0.35)),
                hold_position=int(dropper.get("hold_position", base.gripper.grasp_position)),
                squeeze_position=int(dropper.get("squeeze_position", base.gripper.squeeze_position)),
            ),
            stir=StirMotion(
                enabled=bool(stir.get("enabled", True)),
                orientation_rotvec=_values(stir["orientation_rotvec"], 3, "stir.orientation_rotvec"),
                grasp_offset_m=_values(stir.get("grasp_offset_m", [0, 0, 0]), 3, "stir.grasp_offset_m"),
                approach_m=float(stir.get("approach_m", 0.08)),
                lift_m=float(stir.get("lift_m", 0.10)),
                preinsert_clearance_m=float(stir.get("preinsert_clearance_m", 0.05)),
                target_depth_below_mouth_m=float(stir.get("target_depth_below_mouth_m", 0.03)),
                radius_m=float(stir.get("radius_m", 0.015)),
                turns=int(stir.get("turns", 2)),
                points_per_turn=int(stir.get("points_per_turn", 24)),
                point_hold_s=float(stir.get("point_hold_s", 0.02)),
                gripper_position=int(stir.get("gripper_position", base.gripper.grasp_position)),
            ),
            geometry=GeometryPriors(
                minimum_affordance_points=int(geometry.get("minimum_affordance_points", 20)),
                bottle_model_id=str(geometry["bottle_model_id"]),
                beaker_model_id=str(geometry["beaker_model_id"]),
                dropper_model_id=str(geometry["dropper_model_id"]),
                glass_rod_model_id=str(geometry["glass_rod_model_id"]),
                bottle_axis_out_object=_values(
                    geometry["bottle_axis_out_object"], 3, "geometry_priors.bottle_axis_out_object"
                ),
                bottle_mouth_from_stopper_along_axis_m=float(
                    geometry.get("bottle_mouth_from_stopper_along_axis_m", 0.0)
                ),
                beaker_axis_out_world=_values(
                    geometry.get("beaker_axis_out_world", [0, 0, 1]), 3,
                    "geometry_priors.beaker_axis_out_world",
                ),
                beaker_inner_radius_m=float(geometry["beaker_inner_radius_m"]),
                dropper_tip_anchor_object=_values(
                    geometry["dropper_tip_anchor_object"], 3, "geometry_priors.dropper_tip_anchor_object"
                ),
                glass_rod_grasp_end=str(geometry.get("glass_rod_grasp_end", "higher_world_z")),
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.api_path.startswith("/") or not self.api_path.endswith("/jobs"):
            raise ValueError("api_path must be an absolute API path ending in /jobs.")
        positive = {
            "stopper.approach_m": self.stopper.approach_m,
            "stopper.pull_distance_m": self.stopper.pull_distance_m,
            "stopper.parking_approach_m": self.stopper.parking_approach_m,
            "bottle.approach_m": self.bottle.approach_m,
            "bottle.lift_m": self.bottle.lift_m,
            "bottle.mouth_clearance_m": self.bottle.mouth_clearance_m,
            "dropper.approach_m": self.dropper.approach_m,
            "dropper.lift_m": self.dropper.lift_m,
            "dropper.tip_clearance_m": self.dropper.tip_clearance_m,
            "stir.approach_m": self.stir.approach_m,
            "stir.lift_m": self.stir.lift_m,
            "stir.preinsert_clearance_m": self.stir.preinsert_clearance_m,
            "stir.target_depth_below_mouth_m": self.stir.target_depth_below_mouth_m,
            "stir.radius_m": self.stir.radius_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 < self.bottle.tilt_angle_deg < 170 or self.bottle.tilt_steps < 2:
            raise ValueError("Bottle tilt must be between 0 and 170 degrees with at least two steps.")
        if self.dropper.squeeze_cycles < 1:
            raise ValueError("dropper.squeeze_cycles must be at least one.")
        if self.stir.turns < 1 or self.stir.points_per_turn < 8:
            raise ValueError("Stirring requires at least one turn and eight points per turn.")
        for name, position in (
            ("stopper.gripper_position", self.stopper.gripper_position),
            ("bottle.gripper_position", self.bottle.gripper_position),
            ("dropper.hold_position", self.dropper.hold_position),
            ("dropper.squeeze_position", self.dropper.squeeze_position),
            ("stir.gripper_position", self.stir.gripper_position),
        ):
            if not 0 <= position <= 1000:
                raise ValueError(f"{name} must be within [0, 1000].")
        for name, axis in (("bottle.tilt_axis_tool", self.bottle.tilt_axis_tool),):
            if np.linalg.norm(axis) < 1e-9:
                raise ValueError(f"{name} cannot be zero.")
        if self.geometry.minimum_affordance_points < 3:
            raise ValueError("geometry_priors.minimum_affordance_points must be at least three.")
        for name, model_id in (
            ("bottle_model_id", self.geometry.bottle_model_id),
            ("beaker_model_id", self.geometry.beaker_model_id),
            ("dropper_model_id", self.geometry.dropper_model_id),
            ("glass_rod_model_id", self.geometry.glass_rod_model_id),
        ):
            if not model_id:
                raise ValueError(f"geometry_priors.{name} cannot be empty.")
        if np.linalg.norm(self.geometry.bottle_axis_out_object) < 1e-9:
            raise ValueError("geometry_priors.bottle_axis_out_object cannot be zero.")
        if np.linalg.norm(self.geometry.beaker_axis_out_world) < 1e-9:
            raise ValueError("geometry_priors.beaker_axis_out_world cannot be zero.")
        if not math.isfinite(self.geometry.beaker_inner_radius_m) or self.geometry.beaker_inner_radius_m <= 0:
            raise ValueError("geometry_priors.beaker_inner_radius_m must be finite and positive.")
        if self.geometry.glass_rod_grasp_end not in {"higher_world_z", "end_a", "end_b"}:
            raise ValueError("geometry_priors.glass_rod_grasp_end must be higher_world_z, end_a or end_b.")
