from __future__ import annotations

import json
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .camera import CapturedFrame
from .complex_config import GeometryPriors
from .config import ServerConfig
from .geometry import validate_transform
from .server import build_upload_files

LOG = logging.getLogger(__name__)


def _normalize(raw: Any, name: str) -> np.ndarray:
    value = np.asarray(raw, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite values.")
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError(f"{name} cannot be zero.")
    return value / norm


def _scale(raw: Any, name: str) -> np.ndarray:
    value = np.asarray(raw, dtype=float)
    if value.shape == ():
        value = np.repeat(value, 3)
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"{name} must be a positive scalar or three positive finite values.")
    return value


@dataclass(frozen=True)
class ReconstructedObject:
    name: str
    model_id: str
    transform_world_object: np.ndarray
    scale: np.ndarray
    model_points_world_artifact: str
    regions_world: dict[str, np.ndarray]

    @classmethod
    def parse(
        cls,
        objects: dict[str, Any],
        name: str,
        expected_model_id: str,
        required_regions: tuple[str, ...],
        minimum_points: int,
    ) -> "ReconstructedObject":
        raw = objects.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"objects.{name} must be a JSON object.")
        model_id = raw.get("model_id")
        if model_id != expected_model_id:
            raise ValueError(
                f"objects.{name}.model_id={model_id!r} does not match local geometry prior {expected_model_id!r}."
            )
        transform = validate_transform(raw.get("T_world_object"), f"objects.{name}.T_world_object")
        scale = _scale(raw.get("scale"), f"objects.{name}.scale")
        artifact = raw.get("model_points_world_artifact")
        if not isinstance(artifact, str) or not artifact or artifact.startswith("/") or ".." in Path(artifact).parts:
            raise ValueError(f"objects.{name}.model_points_world_artifact must be a safe relative bundle path.")
        raw_regions = raw.get("affordance_regions", {})
        if not isinstance(raw_regions, dict):
            raise ValueError(f"objects.{name}.affordance_regions must be a JSON object.")
        regions: dict[str, np.ndarray] = {}
        for region_name in required_regions:
            region = raw_regions.get(region_name)
            if not isinstance(region, dict):
                raise ValueError(f"objects.{name}.affordance_regions.{region_name} is required.")
            points = np.asarray(region.get("points_world"), dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < minimum_points:
                raise ValueError(
                    f"objects.{name}.{region_name}.points_world needs at least {minimum_points} 3D points."
                )
            if not np.all(np.isfinite(points)):
                raise ValueError(f"objects.{name}.{region_name}.points_world contains non-finite values.")
            regions[region_name] = points
        return cls(name, model_id, transform, scale, artifact, regions)

    def anchor_world(self, anchor_object: Any) -> np.ndarray:
        anchor = np.asarray(anchor_object, dtype=float)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            raise ValueError(f"Invalid local anchor for {self.name}.")
        return self.transform_world_object[:3, :3] @ (self.scale * anchor) + self.transform_world_object[:3, 3]

    def direction_world(self, direction_object: Any) -> np.ndarray:
        direction = _normalize(direction_object, f"{self.name} canonical direction")
        return _normalize(
            self.transform_world_object[:3, :3] @ (self.scale * direction),
            f"{self.name} world direction",
        )

    def region_center(self, name: str) -> np.ndarray:
        # Median is deliberately computed locally from the actual affordance point region.
        return np.median(self.regions_world[name], axis=0)

    def planar_radius_world(self, radius_object: float, axis_object: Any) -> float:
        axis = _normalize(axis_object, f"{self.name} radius axis")
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        first = _normalize(np.cross(axis, helper), "canonical radius basis")
        second = np.cross(axis, first)
        scale_factor = 0.5 * (np.linalg.norm(self.scale * first) + np.linalg.norm(self.scale * second))
        return float(radius_object * scale_factor)


@dataclass(frozen=True)
class ComplexScene:
    stopper_grasp: np.ndarray
    stopper_axis_out: np.ndarray
    bottle_body_grasp: np.ndarray
    bottle_mouth: np.ndarray
    bottle_axis_out: np.ndarray
    beaker_mouth: np.ndarray
    beaker_normal_out: np.ndarray
    beaker_radius_m: float
    stir_reference_mouth: np.ndarray
    dropper_bulb_grasp: np.ndarray
    dropper_tip: np.ndarray
    glass_rod_grasp: np.ndarray | None
    glass_rod_lower_tip: np.ndarray | None
    model_artifacts: tuple[str, ...]
    raw_response: dict[str, Any]

    @classmethod
    def from_response(
        cls,
        payload: Any,
        geometry: GeometryPriors,
        require_glass_rod: bool = True,
    ) -> "ComplexScene":
        if not isinstance(payload, dict):
            raise ValueError("Complex localization response must be a JSON object.")
        if payload.get("frame") != "world" or payload.get("units") != "meters":
            raise ValueError("Complex API must return frame=world and units=meters (UR base).")
        objects = payload.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("Complex API response is missing objects.")
        minimum = geometry.minimum_affordance_points
        bottle = ReconstructedObject.parse(
            objects, "wide_mouth_bottle", geometry.bottle_model_id,
            ("stopper", "body_grasp"), minimum,
        )
        beaker = ReconstructedObject.parse(
            objects, "beaker", geometry.beaker_model_id, ("mouth",), minimum,
        )
        dropper = ReconstructedObject.parse(
            objects, "dropper", geometry.dropper_model_id, ("bulb",), minimum,
        )
        rod_grasp = rod_tip = None
        if "glass_rod" in objects:
            rod = ReconstructedObject.parse(
                objects, "glass_rod", geometry.glass_rod_model_id,
                ("end_a", "end_b"), minimum,
            )
            end_a = rod.region_center("end_a")
            end_b = rod.region_center("end_b")
            if geometry.glass_rod_grasp_end == "end_a":
                rod_grasp, rod_tip = end_a, end_b
            elif geometry.glass_rod_grasp_end == "end_b":
                rod_grasp, rod_tip = end_b, end_a
            else:
                rod_grasp, rod_tip = (end_a, end_b) if end_a[2] >= end_b[2] else (end_b, end_a)
        elif require_glass_rod:
            raise ValueError("Stirring is enabled but the response has no glass_rod reconstruction.")
        stopper_center = bottle.region_center("stopper")
        body_center = bottle.region_center("body_grasp")
        bottle_axis = bottle.direction_world(geometry.bottle_axis_out_object)
        # Resolve the canonical sign using the observed affordance ordering.
        if float(np.dot(stopper_center - body_center, bottle_axis)) < 0:
            bottle_axis = -bottle_axis
        bottle_mouth = stopper_center + bottle_axis * geometry.bottle_mouth_from_stopper_along_axis_m
        # The beaker receiving target is an existing server affordance region,
        # reused from the old pour/beaker-mouth pipeline. Normal and radius are
        # still local physical-cell configuration, not server predictions.
        beaker_mouth = beaker.region_center("mouth")
        beaker_axis = _normalize(geometry.beaker_axis_out_world, "configured beaker world axis")
        dropper_tip = dropper.anchor_world(geometry.dropper_tip_anchor_object)
        reconstructions = [bottle, beaker, dropper]
        if "glass_rod" in objects:
            reconstructions.append(rod)
        return cls(
            stopper_grasp=stopper_center,
            stopper_axis_out=bottle_axis,
            bottle_body_grasp=body_center,
            bottle_mouth=bottle_mouth,
            bottle_axis_out=bottle_axis,
            beaker_mouth=beaker_mouth,
            beaker_normal_out=beaker_axis,
            beaker_radius_m=geometry.beaker_inner_radius_m,
            stir_reference_mouth=beaker_mouth.copy(),
            dropper_bulb_grasp=dropper.region_center("bulb"),
            dropper_tip=dropper_tip,
            glass_rod_grasp=rod_grasp,
            glass_rod_lower_tip=rod_tip,
            model_artifacts=tuple(item.model_points_world_artifact for item in reconstructions),
            raw_response=payload,
        )


class ComplexLocalizationClient:
    """Async HTTP client for the separate multi-object affordance endpoint."""

    def __init__(
        self,
        config: ServerConfig,
        api_path: str,
        geometry: GeometryPriors,
        require_glass_rod: bool,
    ):
        self.config = config
        self.api_path = api_path
        self.geometry = geometry
        self.require_glass_rod = require_glass_rod

    def locate(
        self,
        frame: CapturedFrame,
        t_world_ee: np.ndarray,
        capture_id: str,
        output_dir: Path,
    ) -> ComplexScene:
        output_dir.mkdir(parents=True, exist_ok=True)
        files = build_upload_files(frame, t_world_ee)
        headers = {"Accept": "application/json"}
        if self.config.bearer_token_env:
            token = os.getenv(self.config.bearer_token_env)
            if not token:
                raise ValueError(f"Set environment variable {self.config.bearer_token_env}.")
            headers["Authorization"] = f"Bearer {token}"
        jobs_url = self.config.url.rstrip("/") + self.api_path
        data = {"depth_scale": str(frame.depth_scale_m), "capture_id": capture_id}
        if self.config.gpu is not None:
            data["gpu"] = str(self.config.gpu)
        try:
            with requests.post(
                jobs_url,
                headers=headers,
                files=files,
                data=data,
                timeout=self.config.timeout_s,
                verify=self.config.verify_tls,
            ) as response:
                submitted = self._checked_json(response)
        except requests.RequestException as exc:
            raise RuntimeError("Complex job submission is ambiguous; inspect the queue before resubmitting.") from exc
        job_id = submitted.get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise ValueError("Server returned an invalid complex-demo job_id.")
        self._write(output_dir / "job.json", submitted)

        deadline = time.monotonic() + self.config.job_timeout_s
        job_url = f"{jobs_url}/{job_id}"
        while True:
            job = self._get_json(job_url, headers, deadline)
            self._write(output_dir / "job.json", {**job, "job_id": job_id})
            state = job.get("state")
            LOG.info("Complex localization job %s: %s", job_id, state)
            if state == "completed":
                break
            if state in {"failed", "interrupted"}:
                raise RuntimeError(f"Complex localization job {job_id} {state}: {json.dumps(job, ensure_ascii=False)}")
            if state not in {"uploading", "queued", "running"}:
                raise ValueError(f"Unknown complex job state: {state!r}.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for complex job {job_id}.")
            time.sleep(min(self.config.poll_seconds, remaining))

        payload = self._get_json(job_url + "/result", headers, deadline)
        self._write(output_dir / "result.json", payload)
        scene = ComplexScene.from_response(payload, self.geometry, self.require_glass_rod)
        if self.config.download_bundle:
            bundle = self._download(job_url, job_id, headers, output_dir)
            self._validate_bundle_artifacts(bundle, scene.model_artifacts)
        return scene

    @staticmethod
    def _checked_json(response: requests.Response) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Server response must be a JSON object.")
        return payload

    def _get_json(self, url: str, headers: dict[str, str], deadline: float) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {url}.")
            try:
                with requests.get(
                    url,
                    headers=headers,
                    timeout=min(self.config.timeout_s, remaining),
                    verify=self.config.verify_tls,
                ) as response:
                    return self._checked_json(response)
            except requests.RequestException as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if (status is not None and status < 500 and status != 429) or attempt == self.config.retries:
                    raise
                time.sleep(min(0.5 * 2**attempt, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(f"GET retries exhausted: {last_error}")

    def _download(self, job_url: str, job_id: str, headers: dict[str, str], output_dir: Path) -> Path:
        final = output_dir / f"complex_{job_id}_{self.config.bundle_scope}.zip"
        partial = final.with_suffix(".zip.part")
        with requests.get(
            job_url + "/bundle",
            params={"scope": self.config.bundle_scope},
            headers=headers,
            stream=True,
            timeout=self.config.timeout_s,
            verify=self.config.verify_tls,
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        handle.write(block)
        partial.replace(final)
        return final

    @staticmethod
    def _validate_bundle_artifacts(bundle: Path, required: tuple[str, ...]) -> None:
        try:
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Downloaded complex result bundle is not a valid ZIP: {exc}") from exc
        missing = sorted(set(required) - names)
        if missing:
            raise ValueError(f"Complex result bundle is missing declared model artifacts: {missing}")

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
