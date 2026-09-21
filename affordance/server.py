from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from .camera import CapturedFrame
from .config import ServerConfig
from .geometry import validate_transform

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocatedPoint:
    position: np.ndarray
    frame: str
    confidence: float | None


@dataclass(frozen=True)
class SceneLocations:
    rubber_bulb: LocatedPoint
    beaker_mouth: LocatedPoint
    raw_response: dict[str, Any]


def _encode_png(image: np.ndarray, name: str) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {name} as PNG.")
    return encoded.tobytes()


def build_upload_files(
    frame: CapturedFrame, t_world_ee: np.ndarray,
) -> dict[str, tuple[str, bytes, str]]:
    """Use the simple intrinsics and matrix JSON formats in the capture API guide."""
    if frame.depth_u16.dtype != np.uint16 or frame.depth_u16.ndim != 2:
        raise ValueError("Aligned depth must be a two-dimensional uint16 image.")
    if frame.color_bgr.dtype != np.uint8 or frame.color_bgr.shape != (*frame.depth_u16.shape, 3):
        raise ValueError("Color must be uint8 BGR and have the same size as aligned depth.")
    height, width = frame.depth_u16.shape
    if frame.intrinsics.get("width") != width or frame.intrinsics.get("height") != height:
        raise ValueError("Color intrinsics dimensions must match the captured images.")
    for key in ("fx", "fy", "cx", "cy"):
        if not np.isfinite(frame.intrinsics.get(key, float("nan"))):
            raise ValueError(f"Missing or invalid camera intrinsic: {key}.")
    if frame.intrinsics["fx"] <= 0 or frame.intrinsics["fy"] <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    matrix = validate_transform(t_world_ee, "T_world_ee")
    camera = {**frame.intrinsics, "depth_scale_m_per_unit": frame.depth_scale_m}
    return {
        "rgb": ("rgb.png", _encode_png(frame.color_bgr, "color image"), "image/png"),
        "depth": ("depth.png", _encode_png(frame.depth_u16, "depth image"), "image/png"),
        "camera": ("camera.json", json.dumps(camera, allow_nan=False).encode(), "application/json"),
        "world_ee": ("T_world_ee.json", json.dumps({"T_world_ee": matrix.tolist()}).encode(), "application/json"),
    }


def _parse_point(result: dict[str, Any], key: str, default_frame: str, unit_scale: float) -> LocatedPoint:
    item = result.get(key)
    if item is None:
        legacy_key = f"{key}_position"
        item = result.get(legacy_key)
    if isinstance(item, dict):
        position = item.get("position")
        frame = str(item.get("coordinate_frame", default_frame))
        confidence = float(item.get("confidence", 1.0))
    else:
        position = item
        frame = default_frame
        confidence = 1.0
    xyz = np.asarray(position, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"Server field {key!r} must contain a finite 3D position.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Server confidence for {key!r} must be in [0, 1].")
    return LocatedPoint(xyz * unit_scale, frame, confidence)


class LocalizationClient:
    def __init__(self, config: ServerConfig):
        self.config = config

    def locate_files(
        self,
        files: dict[str, tuple[str, bytes, str]],
        depth_scale: float,
        capture_id: str,
        output_dir: Path | None = None,
    ) -> SceneLocations:
        """Submit once, wait for completion, persist results and download artifacts."""
        if self.config.download_bundle and output_dir is None:
            raise ValueError("An output directory is required to download the result bundle.")
        if set(files) != {"rgb", "depth", "camera", "world_ee"}:
            raise ValueError("Expected rgb, depth, camera and world_ee upload files.")
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("Depth scale must be finite and positive.")
        headers = {"Accept": "application/json"}
        if self.config.bearer_token_env:
            token = os.getenv(self.config.bearer_token_env)
            if not token:
                raise ValueError(f"Set environment variable {self.config.bearer_token_env} for server authentication.")
            headers["Authorization"] = f"Bearer {token}"
        url = self.config.url.rstrip("/") + "/v1/titration/jobs"
        data = {"depth_scale": str(depth_scale), "capture_id": capture_id}
        if self.config.gpu is not None:
            data["gpu"] = str(self.config.gpu)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        # An ambiguous POST failure may already have queued GPU work. Never resubmit automatically.
        try:
            with requests.post(
                url, headers=headers, files=files, data=data,
                timeout=self.config.timeout_s, verify=self.config.verify_tls,
            ) as response:
                submitted = self._checked_json(response)
        except requests.RequestException as exc:
            raise RuntimeError("Job submission failed; check the server queue before resubmitting.") from exc
        job_id = submitted.get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise ValueError("Server returned an invalid job_id.")
        LOG.info("Submitted localization job %s", job_id)
        self._save_json(output_dir, "job.json", submitted)
        deadline = time.monotonic() + self.config.job_timeout_s
        job_url = f"{url}/{job_id}"
        previous = None
        while True:
            job = self._get_json(job_url, headers, deadline)
            self._save_json(output_dir, "job.json", {**job, "job_id": job_id})
            state = job.get("state")
            if state != previous:
                LOG.info("Localization job %s: %s", job_id, state)
                previous = state
            if state == "completed":
                break
            if state in {"failed", "interrupted"}:
                raise RuntimeError(f"Localization job {job_id} {state}: {json.dumps(job, ensure_ascii=False)}")
            if state not in {"uploading", "queued", "running"}:
                raise ValueError(f"Unknown job state: {state!r}.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for job {job_id}; it may still be running on the server.")
            time.sleep(min(self.config.poll_seconds, remaining))
        payload = self._get_json(job_url + "/result", headers, deadline)
        self._save_json(output_dir, "result.json", payload)
        locations = self._parse_world_response(payload)
        if self.config.download_bundle:
            self._download_bundle(job_url, job_id, headers, output_dir)
        return locations

    @staticmethod
    def _save_json(directory: Path | None, name: str, payload: dict[str, Any]) -> None:
        if directory is not None:
            (directory / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _checked_json(response: requests.Response) -> dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Server response must be a JSON object.")
        return payload

    def _get_json(self, url: str, headers: dict[str, str], deadline: float) -> dict[str, Any]:
        for attempt in range(self.config.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {url}; the server task is not cancelled.")
            try:
                with requests.get(
                    url, headers=headers, timeout=min(self.config.timeout_s, remaining),
                    verify=self.config.verify_tls,
                ) as response:
                    return self._checked_json(response)
            except requests.RequestException as exc:
                status = exc.response.status_code if exc.response is not None else None
                if (status is not None and status < 500 and status != 429) or attempt == self.config.retries:
                    raise
                time.sleep(min(0.5 * 2**attempt, max(0.0, deadline - time.monotonic())))
        raise RuntimeError("GET request exhausted its attempts.")

    def _download_bundle(self, job_url: str, job_id: str, headers: dict[str, str], directory: Path) -> None:
        output = directory / f"titration_{job_id}_{self.config.bundle_scope}.zip"
        partial = output.with_suffix(".zip.part")
        with requests.get(
            job_url + "/bundle", params={"scope": self.config.bundle_scope}, headers=headers,
            stream=True, timeout=self.config.timeout_s, verify=self.config.verify_tls,
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        partial.replace(output)
        LOG.info("Downloaded localization bundle: %s", output)

    @staticmethod
    def _parse_world_response(payload: Any) -> SceneLocations:
        if not isinstance(payload, dict):
            raise ValueError("Server result must be a JSON object.")
        if payload.get("frame") != "world" or payload.get("units") != "meters":
            raise ValueError("Titration API must return frame=world and units=meters (world is UR base).")
        points = []
        for name in ("dropper_bulb_center_world", "beaker_mouth_center_world"):
            point = np.asarray(payload.get(name), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError(f"Server field {name!r} must contain a finite 3D position.")
            # The documented API supplies no confidence: do not fabricate one.
            points.append(LocatedPoint(point, "base", None))
        return SceneLocations(points[0], points[1], payload)

    def locate(
        self,
        frame: CapturedFrame,
        t_world_ee: np.ndarray,
        capture_id: str,
        output_dir: Path | None = None,
    ) -> SceneLocations:
        return self.locate_files(
            build_upload_files(frame, t_world_ee), frame.depth_scale_m, capture_id, output_dir,
        )

    @staticmethod
    def _parse_response(payload: Any) -> SceneLocations:
        if not isinstance(payload, dict):
            raise ValueError("Server response must be a JSON object.")
        if payload.get("status", "ok") != "ok":
            raise ValueError(f"Server reported failure: {payload.get('message', 'unknown error')}")
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            raise ValueError("Server result must be a JSON object.")
        unit = str(result.get("unit", payload.get("unit", "m"))).lower()
        if unit in {"m", "meter", "meters"}:
            scale = 1.0
        elif unit in {"mm", "millimeter", "millimeters"}:
            scale = 0.001
        else:
            raise ValueError(f"Unsupported server position unit: {unit!r}.")
        default_frame = str(result.get("coordinate_frame", payload.get("coordinate_frame", "camera")))
        return SceneLocations(
            rubber_bulb=_parse_point(result, "rubber_bulb", default_frame, scale),
            beaker_mouth=_parse_point(result, "beaker_mouth", default_frame, scale),
            raw_response=payload,
        )
