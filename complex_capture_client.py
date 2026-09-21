#!/usr/bin/env python3
"""Upload an existing complex-demo capture without connecting any hardware."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from affordance.geometry import validate_transform


DEFAULT_OBJECTS = ("wide_mouth_bottle", "beaker", "dropper")
DEFAULT_REGIONS = {
    "wide_mouth_bottle": ["stopper", "body_grasp"],
    "beaker": ["mouth"],
    "dropper": ["bulb"],
}
TERMINAL_STATES = {"completed", "failed", "interrupted"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _headers(token: str) -> dict[str, str]:
    if not token:
        raise ValueError("Set LAG_API_TOKEN or pass --token.")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _checked_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        preview = response.text[:500]
        raise RuntimeError(f"Server returned non-JSON HTTP {response.status_code}: {preview}") from exc
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(payload, ensure_ascii=False)}")
    if not isinstance(payload, dict):
        raise ValueError("Server response must be a JSON object.")
    return payload


def validate_capture(capture_dir: Path) -> tuple[dict[str, Path], float, str]:
    if not capture_dir.is_dir():
        raise ValueError(f"Capture directory does not exist: {capture_dir}")
    paths = {
        "rgb": capture_dir / "rgb.png",
        "depth": capture_dir / "depth.png",
        "camera": capture_dir / "camera.json",
        "world_ee": capture_dir / "T_world_ee.json",
        "capture_info": capture_dir / "capture_info.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Capture is missing required files: {missing}")

    rgb = cv2.imread(str(paths["rgb"]), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if rgb is None or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb.png must be a uint8 three-channel image.")
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("depth.png must be a two-dimensional uint16 PNG.")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth shapes differ: {rgb.shape} vs {depth.shape}.")

    camera = _load_json(paths["camera"])
    height, width = depth.shape
    if camera.get("width") != width or camera.get("height") != height:
        raise ValueError("camera.json dimensions do not match RGB-D images.")
    for key in ("fx", "fy", "cx", "cy"):
        value = float(camera.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"camera.json has invalid {key}.")
    if float(camera["fx"]) <= 0 or float(camera["fy"]) <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    world_ee = _load_json(paths["world_ee"])
    validate_transform(world_ee.get("T_world_ee"), "T_world_ee")
    info = _load_json(paths["capture_info"])
    if info.get("world_frame") != "base" or info.get("ee_frame") != "flange":
        raise ValueError("This capture must declare world_frame=base and ee_frame=flange/tool0.")
    capture_id = str(info.get("capture_id", ""))
    if capture_id != capture_dir.name:
        raise ValueError(f"capture_info capture_id {capture_id!r} differs from directory {capture_dir.name!r}.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", capture_id):
        raise ValueError("capture_id contains unsafe characters.")

    info_scale = float(info.get("depth_scale_m_per_unit", float("nan")))
    camera_scale = float(camera.get("depth_scale_m_per_unit", float("nan")))
    if not math.isfinite(info_scale) or info_scale <= 0:
        raise ValueError("capture_info has an invalid depth scale.")
    if not math.isclose(info_scale, camera_scale, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Depth scale mismatch: capture_info={info_scale}, camera={camera_scale}.")
    return paths, info_scale, capture_id


def validate_complex_result(payload: dict[str, Any], requested_objects: tuple[str, ...]) -> tuple[str, ...]:
    if payload.get("frame") != "world" or payload.get("units") != "meters":
        raise ValueError("Result must declare frame=world and units=meters.")
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Result is missing objects.")
    missing = sorted(set(requested_objects) - set(objects))
    if missing:
        raise ValueError(f"Result is missing requested objects: {missing}")
    artifacts: list[str] = []
    for name in requested_objects:
        item = objects[name]
        if not isinstance(item, dict) or not isinstance(item.get("model_id"), str) or not item["model_id"]:
            raise ValueError(f"objects.{name} is missing a nonempty model_id.")
        validate_transform(item.get("T_world_object"), f"objects.{name}.T_world_object")
        scale = np.asarray(item.get("scale"), dtype=float)
        if scale.shape == ():
            scale = np.repeat(scale, 3)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"objects.{name}.scale must be a positive scalar or three positive values.")
        artifact = item.get("model_points_world_artifact")
        if not isinstance(artifact, str) or not artifact or artifact.startswith("/") or ".." in Path(artifact).parts:
            raise ValueError(f"objects.{name}.model_points_world_artifact is unsafe or missing.")
        artifacts.append(artifact)
        regions = item.get("affordance_regions", {})
        if not isinstance(regions, dict):
            raise ValueError(f"objects.{name}.affordance_regions must be an object.")

    for object_name, region_names in DEFAULT_REGIONS.items():
        if object_name not in requested_objects:
            continue
        regions = objects[object_name]["affordance_regions"]
        for region_name in region_names:
            region = regions.get(region_name)
            points = np.asarray(region.get("points_world") if isinstance(region, dict) else None, dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 3:
                raise ValueError(f"{object_name}.{region_name} must contain at least three world points.")
            if not np.all(np.isfinite(points)):
                raise ValueError(f"{object_name}.{region_name} contains non-finite points.")
    return tuple(artifacts)


def submit_and_wait(args: argparse.Namespace) -> int:
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    paths, depth_scale, capture_id = validate_capture(capture_dir)
    requested_objects = tuple(item.strip() for item in args.objects.split(",") if item.strip())
    if not requested_objects or "glass_rod" in requested_objects:
        raise ValueError("This capture requests wide_mouth_bottle,beaker,dropper and contains no glass_rod.")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else capture_dir / "complex_server_result"
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = _headers(args.token)
    jobs_url = args.server.rstrip("/") + "/v1/complex-demo/jobs"
    data = {
        "depth_scale": repr(depth_scale),
        "capture_id": capture_id,
        "requested_objects": json.dumps(requested_objects, separators=(",", ":")),
        "requested_affordance_regions": json.dumps(DEFAULT_REGIONS, separators=(",", ":")),
    }
    if args.gpu is not None:
        data["gpu"] = str(args.gpu)
    print(f"Validated capture {capture_id}; uploading to {jobs_url}", flush=True)
    try:
        with ExitStack() as stack:
            files = {
                "rgb": ("rgb.png", stack.enter_context(paths["rgb"].open("rb")), "image/png"),
                "depth": ("depth.png", stack.enter_context(paths["depth"].open("rb")), "image/png"),
                "camera": ("camera.json", stack.enter_context(paths["camera"].open("rb")), "application/json"),
                "world_ee": ("T_world_ee.json", stack.enter_context(paths["world_ee"].open("rb")), "application/json"),
            }
            response = requests.post(
                jobs_url,
                headers=headers,
                files=files,
                data=data,
                timeout=args.request_timeout,
                verify=not args.insecure,
            )
        submitted = _checked_json(response)
    except requests.RequestException as exc:
        raise RuntimeError("Submission failed ambiguously; inspect the server queue before retrying.") from exc
    job_id = submitted.get("job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Server returned an invalid job_id.")
    _write_json(output_dir / "job.json", submitted)
    print(f"Submitted complex job: {job_id}", flush=True)

    job_url = f"{jobs_url}/{job_id}"
    deadline = time.monotonic() + args.job_timeout
    previous = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Local wait timed out; server job {job_id} may still be running.")
        response = requests.get(
            job_url,
            headers=headers,
            timeout=min(args.request_timeout, remaining),
            verify=not args.insecure,
        )
        job = _checked_json(response)
        job = {**job, "job_id": job_id}
        _write_json(output_dir / "job.json", job)
        state = job.get("state")
        if state != previous:
            print(f"job {job_id}: {state}", flush=True)
            previous = state
        if state in TERMINAL_STATES:
            if state != "completed":
                _write_json(output_dir / "failure.json", job)
                raise RuntimeError(f"Server job {job_id} ended as {state}; see {output_dir / 'failure.json'}")
            break
        if state not in {"uploading", "queued", "running"}:
            raise ValueError(f"Unknown server state: {state!r}")
        time.sleep(min(args.poll_seconds, remaining))

    result = _checked_json(requests.get(
        job_url + "/result",
        headers=headers,
        timeout=args.request_timeout,
        verify=not args.insecure,
    ))
    _write_json(output_dir / "result.json", result)
    artifacts = validate_complex_result(result, requested_objects)

    final_zip = output_dir / f"complex_{job_id}_{args.scope}.zip"
    partial_zip = final_zip.with_suffix(".zip.part")
    with requests.get(
        job_url + "/bundle",
        params={"scope": args.scope},
        headers=headers,
        stream=True,
        timeout=args.bundle_timeout,
        verify=not args.insecure,
    ) as response:
        response.raise_for_status()
        with partial_zip.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    partial_zip.replace(final_zip)
    with zipfile.ZipFile(final_zip) as archive:
        names = set(archive.namelist())
    missing_artifacts = sorted(set(artifacts) - names)
    if missing_artifacts:
        raise ValueError(f"Downloaded ZIP is missing declared artifacts: {missing_artifacts}")
    print(f"Result JSON: {output_dir / 'result.json'}")
    print(f"Result ZIP:  {final_zip}")
    print("No robot, gripper, or camera connection was made by this client.")
    return 0


def check_health(args: argparse.Namespace) -> int:
    response = requests.get(
        args.server.rstrip("/") + "/health",
        headers=_headers(args.token),
        timeout=args.request_timeout,
        verify=not args.insecure,
    )
    print(json.dumps(_checked_json(response), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.getenv("LAG_SERVER_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("LAG_API_TOKEN", ""))
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    run = sub.add_parser("run")
    run.add_argument("--capture-dir", required=True)
    run.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    run.add_argument("--output-dir")
    run.add_argument("--gpu", type=int)
    run.add_argument("--scope", choices=["results", "all"], default="results")
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument("--job-timeout", type=float, default=1800.0)
    run.add_argument("--bundle-timeout", type=float, default=600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "health":
            return check_health(args)
        return submit_and_wait(args)
    except KeyboardInterrupt:
        print("Interrupted locally; the server job may still be running.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
