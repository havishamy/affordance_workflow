from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CameraConfig


@dataclass(frozen=True)
class CapturedFrame:
    color_bgr: np.ndarray
    depth_u16: np.ndarray
    intrinsics: dict[str, Any]
    depth_scale_m: float
    camera_serial: str
    captured_at_unix_s: float
    rgb_timestamp_ms: float | None = None
    depth_timestamp_ms: float | None = None
    rgb_timestamp_domain: str | None = None
    depth_timestamp_domain: str | None = None


class RealSenseWristCamera:
    """RealSense color + aligned depth acquisition without ROS."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._rs = None
        self._pipeline = None
        self._profile = None
        self._align = None

    def connect(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required to use the wrist camera.") from exc

        self._rs = rs
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(self.config.serial)
        stream_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )
        stream_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )
        try:
            profile = pipeline.start(stream_config)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise
        self._pipeline = pipeline
        self._profile = profile
        self._align = rs.align(rs.stream.color)
        for _ in range(self.config.warmup_frames):
            pipeline.wait_for_frames(self.config.timeout_ms)

    def capture(self) -> CapturedFrame:
        if self._pipeline is None or self._align is None or self._profile is None:
            raise RuntimeError("Camera is not connected.")
        frames = self._pipeline.wait_for_frames(self.config.timeout_ms)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete color/depth frameset.")

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_sensor = self._profile.get_device().first_depth_sensor()
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        if color.shape[:2] != depth.shape:
            raise RuntimeError(f"Aligned color/depth shape mismatch: {color.shape} vs {depth.shape}.")
        return CapturedFrame(
            color_bgr=color,
            depth_u16=depth,
            intrinsics={
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
                "distortion_model": str(intr.model).split(".")[-1],
                "distortion_coefficients": [float(v) for v in intr.coeffs],
            },
            depth_scale_m=float(depth_sensor.get_depth_scale()),
            camera_serial=self.config.serial,
            captured_at_unix_s=time.time(),
            rgb_timestamp_ms=float(color_frame.get_timestamp()),
            depth_timestamp_ms=float(depth_frame.get_timestamp()),
            rgb_timestamp_domain=str(color_frame.get_frame_timestamp_domain()),
            depth_timestamp_domain=str(depth_frame.get_frame_timestamp_domain()),
        )

    def disconnect(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._profile = None
        self._align = None
