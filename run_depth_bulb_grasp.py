#!/usr/bin/env python3
"""Raw RGB-D bulb-center diagnostic; optionally close the gripper without extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from affordance.camera import RealSenseWristCamera
from affordance.depth_bulb_grasp import (
    capture_current_view,
    estimate_bulb_from_raw_depth,
    grasp_from_current_pose,
    load_saved_capture,
    save_debug,
)
from affordance.hardware import DHGripper, URRobot
from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_observation import reuse_latest_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/three_stage_demo.yaml"))
    parser.add_argument("--capture-dir", help="Saved capture for offline analysis; defaults to latest validated three-stage capture")
    parser.add_argument("--live", action="store_true", help="Capture at the current robot pose without moving it or contacting the server")
    parser.add_argument("--execute", action="store_true", help="Requires --live; move to the raw-depth target and close, but do not extract")
    parser.add_argument("--roi", type=int, nargs=4, default=[345, 110, 400, 185], metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--depth-min", type=float, default=0.25)
    parser.add_argument("--depth-max", type=float, default=0.40)
    parser.add_argument("--grasp-z-offset", type=float, default=-0.010, help="Meters relative to visible raw-depth surface median")
    return parser.parse_args()


def confirm(message: str) -> None:
    if input(f"{message}\n输入 DEPTH GRASP 继续，其他输入取消: ").strip().upper() != "DEPTH GRASP":
        raise RuntimeError("Operator cancelled raw-depth grasp diagnostic.")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.execute and not args.live:
        print("ERROR: --execute requires --live; saved captures may no longer match the scene.", file=sys.stderr)
        return 2
    robot = None
    try:
        config = ThreeStageConfig.load(args.config)
        if args.execute and (not config.physical_execution_validated or not config.base.calibration.validated):
            raise RuntimeError("Real grasp requires physical_execution_validated and validated hand-eye calibration.")
        model_xyz = None
        if args.live:
            robot = URRobot(config.base.robot)
            robot.connect()
            capture_dir = capture_current_view(config, robot, RealSenseWristCamera(config.base.camera))
        else:
            if args.capture_dir:
                capture_dir = Path(args.capture_dir).expanduser().resolve()
                result_candidates = sorted(
                    capture_dir.glob("complex_server_result_with_beaker/complex_*/result.json")
                )
                result_path = result_candidates[-1] if result_candidates else None
            else:
                config = reuse_latest_result(config)
                capture_dir = config.capture_dir
                result_path = config.result_dir / "result.json"
            if result_path is not None and result_path.is_file():
                server_result = json.loads(result_path.read_text(encoding="utf-8"))
                if server_result.get("capture_id") != capture_dir.name:
                    raise ValueError("Server result does not belong to the selected capture.")
                local_hand_eye_hash = hashlib.sha256(
                    (Path(__file__).parent / "config/T_ee_cam.json").read_bytes()
                ).hexdigest()
                server_hand_eye_id = server_result.get("calibration", {}).get("id")
                if server_hand_eye_id != f"sha256:{local_hand_eye_hash}":
                    raise ValueError("Local hand-eye file differs from the server calibration used for this result.")
                model_xyz = np.median(
                    np.asarray(server_result["objects"]["dropper"]["affordance_regions"]["bulb"]["points_world"], dtype=float),
                    axis=0,
                )
        color, depth, intrinsics, scale, t_base_camera = load_saved_capture(capture_dir)
        estimate = estimate_bulb_from_raw_depth(
            color, depth, intrinsics, scale, t_base_camera,
            args.roi, args.depth_min, args.depth_max,
        )
        target = estimate.surface_center_base_m + np.array([0.0, 0.0, args.grasp_z_offset])
        output = save_debug(capture_dir, color, estimate, target, model_xyz)
        print(f"采集目录: {capture_dir}")
        print(f"原始深度可见表面中心 base(m): {estimate.surface_center_base_m.round(5).tolist()}")
        print(f"本次测试抓取 TCP base(m): {target.round(5).tolist()}")
        print(f"颜色像素={estimate.color_pixels}, 有效深度={estimate.valid_depth_pixels}, 连续深度={estimate.depth_inlier_pixels}")
        if model_xyz is not None:
            print(f"服务器胶头 affordance 中心 base(m): {model_xyz.round(5).tolist()}")
            print(f"原始深度中心减服务器中心 mm: {((estimate.surface_center_base_m-model_xyz)*1000).round(2).tolist()}")
        print(f"诊断文件: {output}")
        if args.execute:
            grasp_from_current_pose(config, robot, DHGripper(config.base.gripper), target, confirm)
            print("夹爪已按原始深度目标闭合；未执行滴管抽出。")
        else:
            print("仅计算诊断：没有发送抓取运动。")
        return 0
    except KeyboardInterrupt:
        if robot is not None:
            robot.stop()
        print("操作者中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        if robot is not None:
            robot.stop()
        logging.exception("Raw-depth bulb diagnostic failed: %s", exc)
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    sys.exit(main())
