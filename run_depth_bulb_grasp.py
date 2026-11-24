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
    latest_complete_depth_capture,
    load_saved_capture,
    move_to_observation_pose,
    save_debug,
    validate_depth_calibration,
)
from affordance.hardware import DHGripper, URRobot
from affordance.three_stage_config import ThreeStageConfig
from affordance.target_geometry import weighted_modal_center
from affordance.three_stage_result import _read_ply_xyz, _region_scores
from affordance.dropper_instances import DROPPER_IDS, rank_dropper_models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/three_stage_demo.yaml"))
    parser.add_argument("--capture-dir", help="Saved capture for offline analysis; defaults to latest complete RGB-D capture, independent of server results")
    parser.add_argument("--live", action="store_true", help="Capture a fresh RGB-D frame without contacting the server")
    parser.add_argument(
        "--observe", action="store_true",
        help="Requires --live; first move through a guarded transit to the configured fixed observation pose",
    )
    parser.add_argument("--execute", action="store_true", help="Requires --live; move to the raw-depth target and close, but do not extract")
    parser.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"), help="Optional bulb search rectangle; default automatically searches the full image")
    parser.add_argument("--depth-min", type=float, default=0.20)
    parser.add_argument("--depth-max", type=float, default=0.50)
    parser.add_argument("--grasp-z-offset", type=float, default=0.0, help="Explicit base-Z offset in meters from the visible raw-depth center; default 0 for calibration trials")
    parser.add_argument("--compare-server", action="store_true", help="Offline only: additionally compare with a matching saved server bulb region")
    parser.add_argument("--dropper-size", choices=("small", "large"), help="With --compare-server on a dual result, explicitly select the metric-size role")
    parser.add_argument("--yes", action="store_true", help="Automatically accept observation and grasp checkpoints without reading stdin")
    return parser.parse_args(argv)


def confirm(message: str) -> None:
    if input(f"{message}\n输入 DEPTH GRASP 继续，其他输入取消: ").strip().upper() != "DEPTH GRASP":
        raise RuntimeError("Operator cancelled raw-depth grasp diagnostic.")


def auto_confirm(message: str) -> None:
    print(f"{message}\n--yes：自动通过确认点。", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.execute and not args.live:
        print("ERROR: --execute requires --live; saved captures may no longer match the scene.", file=sys.stderr)
        return 2
    if args.observe and not args.live:
        print("ERROR: --observe requires --live.", file=sys.stderr)
        return 2
    if args.live and (args.capture_dir or args.compare_server):
        print("ERROR: --capture-dir and --compare-server are offline-only options.", file=sys.stderr)
        return 2
    if args.dropper_size and not args.compare_server:
        print("ERROR: --dropper-size is an offline --compare-server option.", file=sys.stderr)
        return 2
    if not np.isfinite(args.grasp_z_offset):
        print("ERROR: --grasp-z-offset must be finite.", file=sys.stderr)
        return 2
    robot = None
    capture_dir = None
    stage = "configuration"
    try:
        config = ThreeStageConfig.load(args.config)
        calibration_path = validate_depth_calibration(config)
        checkpoint = auto_confirm if args.yes else confirm
        if (args.execute or args.observe) and (not config.physical_execution_validated or not config.base.calibration.validated):
            raise RuntimeError("Real grasp requires physical_execution_validated and validated hand-eye calibration.")
        model_xyz = None
        if args.live:
            stage = "observation_or_capture"
            robot = URRobot(config.base.robot)
            robot.connect()
            observation_plan = move_to_observation_pose(config, robot, checkpoint) if args.observe else None
            capture_dir = capture_current_view(config, robot, RealSenseWristCamera(config.base.camera))
            if observation_plan is not None:
                (capture_dir / "raw_depth_bulb_observation_plan.json").write_text(
                    json.dumps(observation_plan, ensure_ascii=False, indent=2), encoding="utf-8",
                )
        else:
            capture_dir = Path(args.capture_dir).expanduser().resolve() if args.capture_dir else latest_complete_depth_capture(config.base.output_dir)
            if args.compare_server:
                result_candidates = sorted(
                    capture_dir.glob("complex_server_result_with_beaker/complex_*/result.json")
                )
                result_path = result_candidates[-1] if result_candidates else None
                if result_path is None:
                    raise ValueError("No saved server result is available for --compare-server.")
                server_result = json.loads(result_path.read_text(encoding="utf-8"))
                if server_result.get("capture_id") != capture_dir.name:
                    raise ValueError("Server result does not belong to the selected capture.")
                # Verify the actual transform, not a serialization hash that
                # may differ solely because metadata/whitespace changed.
                _, _, _, _, local_transform = load_saved_capture(capture_dir)
                saved_transform = server_result.get("coordinate_conventions", {}).get("T_world_from_sam3d")
                if saved_transform is not None:
                    comparable = np.asarray(saved_transform, dtype=float) @ np.diag([-1, -1, 1, 1])
                    if comparable.shape != (4, 4) or not np.allclose(comparable, local_transform, atol=1e-6, rtol=0):
                        raise ValueError("Server and local camera-world transforms differ; comparison is invalid.")
                elif server_result.get("calibration", {}).get("id") != "sha256:" + hashlib.sha256(calibration_path.read_bytes()).hexdigest():
                    raise ValueError("Cannot verify the calibration of this server comparison.")
                objects = server_result["objects"]
                if any(name in objects for name in DROPPER_IDS):
                    if args.dropper_size is None:
                        raise ValueError("Dual-dropper comparison requires --dropper-size small or large.")
                    clouds = {name: _read_ply_xyz(result_path.parent / objects[name]["model_points_world_artifact"]) for name in DROPPER_IDS}
                    ranked = rank_dropper_models(clouds, config.dropper_minimum_size_ratio)
                    dropper = objects[ranked[f"{args.dropper_size}_id"]]
                else:
                    if args.dropper_size is not None:
                        raise ValueError("This is a legacy single-dropper result; it has no independently size-ranked small/large roles.")
                    dropper = objects["dropper"]
                region = dropper["affordance_regions"]["bulb"]
                points = np.asarray(region["points_world"], dtype=float)
                model_xyz, _ = weighted_modal_center(
                    points, _region_scores(result_path.parent, region, points),
                    dropper.get("main_axis_world", [0, 0, 1]), config=config.grasp_center,
                    bounds=(config.base.motion.workspace_min, config.base.motion.workspace_max),
                )
        stage = "raw_depth_estimation"
        print(f"采集目录: {capture_dir}")
        color, depth, intrinsics, scale, t_base_camera = load_saved_capture(capture_dir)
        source_t_base_camera = t_base_camera.copy()
        print("纯深度定位：使用原始手眼坐标，不做解算后桌面旋转、XY 平移或 Z 校正。")
        estimate = estimate_bulb_from_raw_depth(
            color, depth, intrinsics, scale, t_base_camera,
            args.roi, args.depth_min, args.depth_max,
            center_config=config.grasp_center,
        )
        target = estimate.surface_center_base_m + np.array([0.0, 0.0, args.grasp_z_offset])
        output = save_debug(
            capture_dir, color, estimate, target, model_xyz, args.roi,
            intrinsics=intrinsics, t_base_camera=t_base_camera,
            calibration_path=calibration_path, tcp_offset=config.base.robot.tcp_offset,
            source_t_base_camera=source_t_base_camera,
        )
        camera_center = estimate.surface_center_camera_m
        center_pixel = [camera_center[0] / camera_center[2] * intrinsics["fx"] + intrinsics["cx"], camera_center[1] / camera_center[2] * intrinsics["fy"] + intrinsics["cy"]]
        print(f"搜索 ROI: {list(estimate.roi_px)}；加权中心像素: {[round(float(v), 2) for v in center_pixel]}")
        print(f"当前 TCP 偏移: {list(config.base.robot.tcp_offset)}")
        print(f"原始深度可见表面中心 base(m): {estimate.surface_center_base_m.round(5).tolist()}")
        print(f"本次测试抓取 TCP base(m): {target.round(5).tolist()}")
        print(f"显式 Z 偏移: {args.grasp_z_offset * 1000:.1f} mm；坐标来自原始深度，不调用 Qwen/SAM/服务器。")
        print(f"颜色像素={estimate.color_pixels}, 有效深度={estimate.valid_depth_pixels}, 连续深度={estimate.depth_inlier_pixels}")
        if model_xyz is not None:
            print(f"服务器胶头 affordance 中心 base(m): {model_xyz.round(5).tolist()}")
            print(f"原始深度中心减服务器中心 mm: {((estimate.surface_center_base_m-model_xyz)*1000).round(2).tolist()}")
        print(f"诊断文件: {output}")
        if args.execute:
            stage = "grasp"
            grasp_from_current_pose(config, robot, DHGripper(config.base.gripper), target, checkpoint, record_dir=output)
            print("夹爪已按原始深度目标闭合并保持夹持；未执行滴管抽出。")
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
        if capture_dir is not None:
            diagnostic = capture_dir / "raw_depth_bulb_diagnostic"
            diagnostic.mkdir(exist_ok=True)
            (diagnostic / "failure.json").write_text(json.dumps({"stage": stage, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    sys.exit(main())
