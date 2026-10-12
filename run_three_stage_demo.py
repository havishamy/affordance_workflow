#!/usr/bin/env python3
"""Plan or execute stopper, pouring, dosing and post-dose glass-rod stirring."""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_workflow import CHEMISTRY_STAGES, STAGES, ThreeStageWorkflow


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
        help="Execute selected stage(s) after optional observation; requires validated config and typed confirmations",
    )
    command.add_argument(
        "--stir-observe", type=int, choices=(0, 1), default=None,
        help="For stirring: 1 captures/submits after dropper return; 0 reuses the latest matching stirring result",
    )
    command.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return command


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = ThreeStageConfig.load(args.config)
        observe = config.observation.enabled if args.observe is None else args.observe
        if args.stage == "stirring" and observe == 1:
            raise ValueError(
                "Do not run complex-demo observation after the stopper was removed; use --observe 0 "
                "and --stir-observe 1 for the final glass-rod observation."
            )
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
        chemistry_names = (
            CHEMISTRY_STAGES if args.stage == "all"
            else ((args.stage,) if args.stage in CHEMISTRY_STAGES else ())
        )
        plans = (
            workflow.plan_all(scene, captured_pose)
            if args.stage == "all"
            else tuple(workflow.plan_stage(name, scene, captured_pose) for name in chemistry_names)
        )
        if plans:
            output = workflow.save_offline_plan(scene, plans)
            print(f"前三阶段离线路径已保存: {output}")
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
        stirring_observation = None
        if args.stage == "stirring":
            from affordance.stirring import StirringObservation, StirringWorkflow, reuse_latest_stirring

            stir_observe = 1 if args.stir_observe is None else args.stir_observe
            if stir_observe == 1:
                print("玻璃棒观测模式=1：将移动到最终观察位、采集 RGB-D 并调用 /v1/stirring/jobs。")
                stirring_observation = StirringObservation(config).run(scene, _confirm)
            else:
                stirring_observation = reuse_latest_stirring(config, scene)
                print(f"玻璃棒观测模式=0：复用 {stirring_observation.result_dir}")
            print(f"玻璃棒采集目录: {stirring_observation.capture_dir}")
            print(f"玻璃棒解算结果: {stirring_observation.result_dir}")
            print(f"玻璃棒长度: {stirring_observation.geometry.length_m:.4f} m")
            if config.stirring_motion.validated:
                stirring_workflow = StirringWorkflow(
                    config, scene, stirring_observation.geometry, stirring_observation.result_dir,
                )
                stirring_pose = __import__("json").loads(
                    (stirring_observation.capture_dir / "capture_info.json").read_text(encoding="utf-8")
                )["actual_tcp_pose"]
                candidate_plans = stirring_workflow.plan_candidates(stirring_pose)
                stir_plan = candidate_plans[0]
                stir_output = stirring_workflow.save_plan(stir_plan, stirring_observation.capture_dir)
                print(f"玻璃棒离线规划: {stir_output}")
                print(f"几何裕量最佳候选: {stir_plan.candidate.id}")
            elif not args.execute:
                print("玻璃棒感知已完成；stirring_motion.validated=false，尚未生成或执行搅拌运动。")
        elif args.stir_observe is not None:
            raise ValueError("--stir-observe only applies to --stage stirring; --stage all always observes after dropper return.")

        if not args.execute:
            if observe == 1:
                print("已完成初始观察位姿移动、RGB-D 采集与服务器解算；未执行实验动作。")
            else:
                if args.stage != "stirring" or args.stir_observe == 0:
                    print("仅规划/复用模式：未发送抓取、倾倒、滴液或搅拌动作。")
            if args.stage == "all":
                print("第四阶段必须在滴管归位后重新观察；本次未执行，因此玻璃棒规划被延后。")
            return 0
        if args.stage in {"all", "stirring"} and not config.stirring_motion.validated:
            raise RuntimeError(
                "stirring_motion.validated=false; measure every stirring parameter before any requested four-stage execution."
            )
        print(
            "警告：前三阶段使用保存的场景坐标；只有物体、底座、TCP、夹爪与停车位均已人工确认时才可继续。"
        )
        if chemistry_names:
            result_dir = workflow.execute(scene, chemistry_names, _confirm)
            print(f"前三阶段执行记录: {result_dir}")
        if args.stage in {"all", "stirring"}:
            from affordance.stirring import StirringObservation, StirringWorkflow

            if args.stage == "all":
                stirring_observation = StirringObservation(config).run(scene, _confirm)
            assert stirring_observation is not None
            stirring_workflow = StirringWorkflow(
                config, scene, stirring_observation.geometry, stirring_observation.result_dir,
            )
            stirring_run = stirring_workflow.execute(stirring_observation.capture_dir, _confirm)
            print(f"第四阶段执行记录: {stirring_run}")
        return 0
    except KeyboardInterrupt:
        logging.error("Operator interrupted the workflow.")
        return 130
    except Exception as exc:
        logging.exception("Four-stage workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
