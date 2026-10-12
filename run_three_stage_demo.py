#!/usr/bin/env python3
"""Plan or explicitly execute stopper removal, bottle pouring and dropper dosing."""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_workflow import STAGES, ThreeStageWorkflow


def _confirmation(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Cf").strip().upper()


def _confirm(label: str, message: str) -> None:
    phrase = "EXECUTE " + label.upper()
    print(f"\n{message}\n当前步骤: {label}\n确认示教器保护/急停可用、工作区无人。")
    if _confirmation(input(f"输入 {phrase} 继续；其他输入取消: ")) != phrase:
        raise RuntimeError(f"Operator cancelled at {label}.")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--config", default=str(Path(__file__).parent / "config" / "three_stage_demo.yaml"),
    )
    command.add_argument("--stage", choices=("all", *STAGES), default="all")
    command.add_argument(
        "--observe", type=int, choices=(0, 1), default=None,
        help="Override observation.enabled: 1 moves to the saved observation pose, captures and resolves; 0 reuses saved result",
    )
    command.add_argument(
        "--execute", action="store_true",
        help="Execute selected chemistry stage(s) after optional observation; requires validated config and typed confirmations",
    )
    command.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return command


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = ThreeStageConfig.load(args.config)
        observe = config.observation.enabled if args.observe is None else args.observe
        if observe == 1:
            from affordance.three_stage_observation import ThreeStageObservation

            print(
                "观测模式=1：确认后将移动机械臂到历史观察法兰位姿、采集 RGB-D、请求服务器并下载新结果。"
                "即使未传 --execute，此观测阶段也会运动。"
            )
            config = ThreeStageObservation(config).run(_confirm)
            print(f"新采集目录: {config.capture_dir}")
            print(f"新解算结果: {config.result_dir}")
        else:
            from affordance.three_stage_observation import reuse_latest_result

            config = reuse_latest_result(config)
            print(f"观测模式=0：沿用本地解算结果 {config.result_dir}；不连接相机或服务器。")
        workflow = ThreeStageWorkflow(config)
        scene = workflow.load_scene()
        captured_pose = workflow.captured_tcp_pose()
        names = STAGES if args.stage == "all" else (args.stage,)
        plans = (
            workflow.plan_all(scene, captured_pose)
            if args.stage == "all"
            else (workflow.plan_stage(args.stage, scene, captured_pose),)
        )
        output = workflow.save_offline_plan(scene, plans)
        print(f"离线路径已保存: {output}")
        print("本次定位:")
        print(f"  瓶塞抓取点: {scene.stopper_grasp.round(4).tolist()} m")
        print(f"  瓶身抓取点: {scene.bottle_body_grasp.round(4).tolist()} m")
        print(f"  烧杯接液中心: {scene.beaker_mouth_center.round(4).tolist()} m")
        print(f"  滴管胶头 affordance 中心: {scene.dropper_bulb_grasp.round(4).tolist()} m")
        dropper_plan = next((plan for plan in plans if plan.name == "dropper"), None)
        if dropper_plan is not None:
            grasp_pose = next(step.pose for step in dropper_plan.steps if step.name == "dropper_bulb_grasp")
            print(f"  实际胶头抓取 TCP: {[round(value, 4) for value in grasp_pose[:3]]} m")
        for plan in plans:
            print(f"  {plan.name}: {len(plan.steps)} 步，{sum(step.kind == 'moveL' for step in plan.steps)} 个 moveL")
            for warning in plan.warnings:
                print(f"    注意（需现场核对）: {warning}")
        if not args.execute:
            if observe == 1:
                print("已完成观察位姿移动、RGB-D 采集与服务器解算；未执行拔塞、倾倒或滴液动作。")
            else:
                print("仅规划模式：未连接机械臂、夹爪或相机，未发送任何运动命令。")
            return 0
        print(
            "警告：将使用保存的旧场景坐标执行真机动作。只有物体、底座、TCP、夹爪与停车位均已人工确认未变化时才可继续。"
        )
        result_dir = workflow.execute(scene, names, _confirm)
        print(f"执行记录: {result_dir}")
        return 0
    except KeyboardInterrupt:
        logging.error("Operator interrupted the workflow.")
        return 130
    except Exception as exc:
        logging.exception("Three-stage workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
