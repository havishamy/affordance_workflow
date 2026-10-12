from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .config import GripperConfig, RobotConfig


class URRobot:
    """Minimal blocking UR RTDE wrapper, following ur_data_collection."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.receive = None
        self.control = None

    def connect(self) -> None:
        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise RuntimeError("ur_rtde is required to control the robot.") from exc
        self.receive = RTDEReceiveInterface(self.config.ip)
        try:
            self.control = RTDEControlInterface(self.config.ip)
            result = self.control.setTcp(list(self.config.tcp_offset))
            if isinstance(result, (bool, np.bool_)) and not bool(result):
                raise RuntimeError("RTDE setTcp returned False; the capture/calibration frame would be incorrect.")
            result = self.control.setPayload(self.config.payload_mass, list(self.config.payload_cog))
            if isinstance(result, (bool, np.bool_)) and not bool(result):
                raise RuntimeError("RTDE setPayload returned False.")
            pose = self.actual_tcp_pose()
            if len(pose) != 6:
                raise RuntimeError("Failed to read a valid TCP pose from UR.")
        except Exception:
            self.disconnect()
            raise

    def actual_tcp_pose(self) -> list[float]:
        if self.receive is None:
            raise RuntimeError("Robot is not connected.")
        return [float(v) for v in self.receive.getActualTCPPose()]

    def assert_motion_ready(self) -> None:
        if self.receive is None or self.control is None:
            raise RuntimeError("Robot is not connected.")
        if hasattr(self.receive, "isProtectiveStopped") and self.receive.isProtectiveStopped():
            raise RuntimeError("UR is protective-stopped; refusing motion.")
        if hasattr(self.receive, "isEmergencyStopped") and self.receive.isEmergencyStopped():
            raise RuntimeError("UR is emergency-stopped; refusing motion.")

    def move_linear(
        self, pose: Sequence[float], speed: float | None = None, acceleration: float | None = None,
    ) -> None:
        if self.control is None:
            raise RuntimeError("Robot is not connected.")
        target = [float(v) for v in pose]
        self.assert_motion_ready()
        selected_speed = self.config.speed if speed is None else float(speed)
        selected_acceleration = self.config.acceleration if acceleration is None else float(acceleration)
        if not 0 < selected_speed <= 0.2 or not 0 < selected_acceleration <= 0.2:
            raise ValueError("UR Cartesian speed and acceleration must be in (0, 0.2].")
        result = self.control.moveL(target, selected_speed, selected_acceleration)
        if isinstance(result, (bool, np.bool_)) and not bool(result):
            raise RuntimeError(f"UR moveL failed for target {target}.")
        actual = np.asarray(self.actual_tcp_pose()[:3])
        error = float(np.linalg.norm(actual - np.asarray(target[:3])))
        if error > self.config.position_tolerance_m:
            raise RuntimeError(
                f"UR stopped {error:.4f} m from target (limit {self.config.position_tolerance_m:.4f} m)."
            )

    def has_inverse_kinematics(self, pose: Sequence[float]) -> bool:
        if self.control is None:
            raise RuntimeError("Robot is not connected.")
        return bool(self.control.getInverseKinematicsHasSolution([float(value) for value in pose]))

    def inverse_kinematics(self, pose: Sequence[float]) -> list[float]:
        if self.control is None:
            raise RuntimeError("Robot is not connected.")
        joints = [float(value) for value in self.control.getInverseKinematics([float(value) for value in pose])]
        if len(joints) != 6 or not np.all(np.isfinite(joints)):
            raise RuntimeError("UR controller did not return a valid six-joint IK solution.")
        return joints

    def joints_within_safety_limits(self, joints: Sequence[float]) -> bool:
        if self.control is None:
            raise RuntimeError("Robot is not connected.")
        return bool(self.control.isJointsWithinSafetyLimits([float(value) for value in joints]))

    def stop(self) -> None:
        if self.control is not None:
            try:
                self.control.stopL(2.0)
            except Exception:
                pass

    def disconnect(self) -> None:
        if self.control is not None:
            try:
                self.control.disconnect()
            except Exception:
                pass
        if self.receive is not None:
            try:
                self.receive.disconnect()
            except Exception:
                pass
        self.control = None
        self.receive = None


class DHGripper:
    """DH PGE wrapper with the same UART initialization as ur_data_collection."""

    def __init__(self, config: GripperConfig):
        self.config = config
        self.device = None

    def connect(self) -> None:
        try:
            import crcmod
            import serial
            from pyDHgripper import PGE
        except ImportError as exc:
            raise RuntimeError("pyDHgripper, pyserial and crcmod are required for the gripper.") from exc

        if self.config.init_feedback:
            device = PGE(self.config.port)
            device.init_feedback()
        else:
            device = PGE.__new__(PGE)
            device.ser = serial.Serial(port=self.config.port, baudrate=115200)
            device.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
        device.set_force(self.config.force)
        device.set_vel(self.config.speed)
        self.device = device

    def set_position(self, position: int, wait: bool = True) -> None:
        if self.device is None:
            raise RuntimeError("Gripper is not connected.")
        if not 0 <= position <= 1000:
            raise ValueError("Gripper position must be within [0, 1000].")
        self.device.set_pos(val=int(position), blocking=False)
        if not wait:
            return
        deadline = time.monotonic() + self.config.command_timeout_s
        while time.monotonic() < deadline:
            try:
                if abs(int(self.device.read_pos()) - int(position)) <= 30:
                    return
            except Exception:
                # Some firmware does not expose reliable position feedback.
                break
            time.sleep(0.05)
        # Do not falsely claim exact contact: allow motion time, then continue.
        time.sleep(min(0.5, self.config.command_timeout_s))

    def set_force(self, force: int) -> None:
        if self.device is None:
            raise RuntimeError("Gripper is not connected.")
        if not 0 <= force <= 100:
            raise ValueError("Gripper force must be within [0, 100].")
        self.device.set_force(int(force))

    def open(self) -> None:
        self.set_position(self.config.open_position)

    def grasp(self) -> None:
        self.set_position(self.config.grasp_position)

    def squeeze(self) -> None:
        self.set_position(self.config.squeeze_position)

    def disconnect(self) -> None:
        if self.device is not None and getattr(self.device, "ser", None) is not None:
            try:
                self.device.ser.close()
            except Exception:
                pass
        self.device = None
