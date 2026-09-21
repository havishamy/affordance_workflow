#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

from affordance.complex_config import ComplexDemoConfig
from affordance.complex_workflow import STAGE_NAMES, ComplexDemoWorkflow, StagePlan


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return "".join(char for char in value if unicodedata.category(char) != "Cf").strip().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent four-stage UR chemistry demo (no ROS)")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "complex_demo.yaml"),
        help="Complex demo YAML configuration",
    )
    parser.add_argument(
        "--scene-json",
        help="Use a saved complex API response instead of opening the camera or contacting the server",
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_NAMES),
        help=f"Comma-separated ordered subset of: {','.join(STAGE_NAMES)}",
    )
    parser.add_argument("--execute", action="store_true", help="Execute the generated stages on real hardware")
    parser.add_argument("--yes", action="store_true", help="Skip per-stage typed confirmations")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    try:
        config = ComplexDemoConfig.load(args.config)

        def confirm(stage_name: str, plan: StagePlan) -> None:
            phrase = f"EXECUTE {stage_name.upper()}"
            print(
                f"阶段 {stage_name} 即将执行，共 {len(plan.actions)} 个动作。"
                "请检查物体状态、路径、夹持参数、液体条件和急停。"
            )
            if _normalized(input(f"输入 {phrase} 继续: ")) != phrase:
                raise RuntimeError(f"Operator cancelled before stage {stage_name}.")

        result = ComplexDemoWorkflow(config).run(
            execute=args.execute,
            stages=stages,
            scene_json=args.scene_json,
            before_stage=None if args.yes else confirm,
        )
        print(f"复杂实验流程完成。执行模式={args.execute}，记录目录={result['run_dir']}")
        return 0
    except KeyboardInterrupt:
        logging.error("Operator interrupted the complex demo.")
        return 130
    except Exception as exc:
        logging.exception("Complex demo failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
