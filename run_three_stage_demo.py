#!/usr/bin/env python3
"""Plan/execute stopper, base pour, size-ordered two-dropper doses and optional stirring."""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from dataclasses import replace
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


def _auto_confirm(label: str, message: str) -> None:
    """Acknowledge an operator checkpoint without reading from stdin."""
    print(f"\n{message}\n当前步骤: {label}\n--yes：自动通过，不等待键盘输入。", flush=True)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--config", default=str(Path(__file__).parent / "config" / "three_stage_demo.yaml"),
    )
    command.add_argument("--stage", choices=("all", *STAGES), default="all")
    command.add_argument(
        "--dropper-count", type=int, choices=(1, 2), default=None,
        help="Override droppers.count: 1 uses the legacy three-object experiment; 2 uses two droppers",
    )
    command.add_argument(
        "--observe", type=int, choices=(0, 1), default=None,
        help="Override observation.enabled: 1 moves to the saved observation pose, captures and resolves; 0 reuses saved result",
    )
    command.add_argument(
        "--execute", action="store_true",
        help="Execute selected stage(s) after optional observation; requires validated config",
    )
    command.add_argument(
        "--yes", action="store_true",
        help="Automatically accept every observation, stage and motion checkpoint without reading stdin",
    )
    command.add_argument(
        "--stir-observe", type=int, choices=(0, 1), default=None,
        help="For stirring: 1 captures/submits after dropper return; 0 reuses the latest matching stirring result",
    )
    command.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return command


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        confirm = _auto_confirm if args.yes else _confirm
        if args.yes:
            print(
                "全自动确认模式：所有人工确认点将立即通过；自动安全校验、控制器失败、保护停止和急停仍会中止。",
                flush=True,
            )
        config = ThreeStageConfig.load(args.config)
        if args.dropper_count is not None:
            config = replace(config, dropper_count=args.dropper_count)
            config.validate()
        chemistry_names = config.select_chemistry_stages(args.stage)
        do_stirring = args.stage == "stirring" or (args.stage == "all" and config.stirring_enabled)
        print(f"实验配置: {Path(args.config).expanduser().resolve()}", flush=True)
        print(f"桌面初始化参考: {'base z=0；仅服务器初始化使用，保留 XY；历史结果不补做校正' if config.table_constraint_enabled else '未请求'}；客户端不旋转或平移解算结果。", flush=True)
        print(
            f"实验模式: {'旧三物体/单滴瓶' if config.dropper_count == 1 else '四物体/双滴瓶'}；"
            f"请求物体: {', '.join(config.object_names)}；"
            f"化学阶段: {', '.join(chemistry_names) or '无'}；最终搅拌: {'开启' if do_stirring else '关闭'}",
            flush=True,
        )
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
            config = ThreeStageObservation(config).run(confirm)
            print(f"新采集目录: {config.capture_dir}")
            print(f"新解算结果: {config.result_dir}")
        else:
            from affordance.three_stage_observation import reuse_latest_result

            config = reuse_latest_result(config)
            print(f"观测模式=0：沿用本地解算结果 {config.result_dir}；不连接相机或服务器。")
        workflow = ThreeStageWorkflow(config)
        scene = workflow.load_scene()
        captured_pose = workflow.captured_tcp_pose()
        plans = (
            workflow.plan_all(scene, captured_pose)
            if args.stage == "all"
            else workflow.plan_sequence(chemistry_names, scene, captured_pose)
        )
        if plans:
            output = workflow.save_offline_plan(scene, plans)
            print(f"化学动作离线路径已保存: {output}")
        print("本次定位:")
        print(f"  瓶塞抓取点: {scene.stopper_grasp.round(4).tolist()} m")
        print(f"  瓶身抓取点: {scene.bottle_body_grasp.round(4).tolist()} m")
        print(f"  烧杯接液中心: {scene.beaker_mouth_center.round(4).tolist()} m")
        if config.dropper_count == 2:
            ranking = scene.dropper_size_ranking
            for role, label in (("small", "滴瓶1/小滴瓶/酚酞"), ("large", "滴瓶2/大滴瓶/酸液")):
                instance = scene.dropper_instance_ids[role]
                volume = ranking["models"][instance]["volume_m3"]
                print(f"  {label}: {instance}；米制模型包围体积={volume:.8g} m³；胶头中心={scene.dropper_grasp(role).round(4).tolist()} m")
            print(f"  大/小体积比: {ranking['large_to_small_ratio']:.4f}；仅按三维模型实际大小分配角色。")
        else:
            print(f"  单滴瓶胶头中心: {scene.dropper_bulb_grasp.round(4).tolist()} m")
        for dropper_plan in (p for p in plans if p.name.startswith("dropper")):
            grasp_pose = next(step.pose for step in dropper_plan.steps if step.name == "dropper_bulb_grasp")
            print(f"  {dropper_plan.name} 实际抓取 TCP: {[round(value, 4) for value in grasp_pose[:3]]} m；归位对象={dropper_plan.metadata['source_object_id']}")
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
                stirring_observation = StirringObservation(config).run(scene, confirm)
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
                print("最终搅拌需在所有滴管归位后重新观察；本次未执行，搅拌规划延后。" if do_stirring else "配置已关闭最终搅拌。")
            return 0
        if do_stirring and not config.stirring_motion.validated:
            raise RuntimeError(
                "stirring_motion.validated=false; measure stirring parameters before requesting final stirring."
            )
        print(
            "警告：化学动作使用保存的场景坐标；只有物体、底座、TCP、夹爪与停车位均已人工确认时才可继续。"
        )
        if chemistry_names:
            result_dir = workflow.execute(scene, chemistry_names, confirm)
            print(f"化学动作执行记录: {result_dir}")
        if do_stirring:
            from affordance.stirring import StirringObservation, StirringWorkflow

            if args.stage == "all":
                stirring_observation = StirringObservation(config).run(scene, confirm)
            assert stirring_observation is not None
            stirring_workflow = StirringWorkflow(
                config, scene, stirring_observation.geometry, stirring_observation.result_dir,
            )
            stirring_run = stirring_workflow.execute(stirring_observation.capture_dir, confirm)
            print(f"最终搅拌执行记录: {stirring_run}")
        return 0
    except KeyboardInterrupt:
        logging.error("Operator interrupted the workflow.")
        return 130
    except Exception as exc:
        logging.exception("Chemistry workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
