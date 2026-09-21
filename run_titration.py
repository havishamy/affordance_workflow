#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

from affordance.config import AppConfig
from affordance.workflow import TitrationWorkflow


class ExecutionCancelled(Exception):
    pass


def normalize_confirmation(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = "".join(character for character in normalized if unicodedata.category(character) != "Cf")
    return normalized.strip().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR eye-in-hand dropper titration (no ROS)")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "config.yaml"),
        help="YAML configuration path",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the arm and gripper; omitted means capture/localize/plan only",
    )
    mode.add_argument("--capture-only", action="store_true", help="Save the four upload files without contacting the server or moving the arm/gripper")
    mode.add_argument(
        "--reuse-capture",
        metavar="DIR",
        help="Rebuild a plan from a completed capture without using the camera/server or moving the arm/gripper",
    )
    mode.add_argument(
        "--execute-saved",
        metavar="DIR",
        help="Execute from a completed capture without running localization again",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the typed operator confirmation used together with --execute",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = AppConfig.load(args.config)

        def confirm_execution() -> None:
            reused = bool(args.execute_saved)
            print(
                ("即将复用已保存的世界坐标。" if reused else "本次定位结果与路径已生成。")
                + "请检查结果 ZIP、两个基坐标及本次重新生成的路径，"
                + "确认滴管和烧杯从采集后完全未移动、TCP/偏移正确、急停可用。"
            )
            entered = normalize_confirmation(input("输入 EXECUTE 继续: "))
            if entered != "EXECUTE":
                raise ExecutionCancelled(f"未识别确认输入：{entered!r}")
        workflow = TitrationWorkflow(config)
        if args.reuse_capture or args.execute_saved:
            execute = bool(args.execute_saved)
            result = workflow.run_saved(
                args.execute_saved or args.reuse_capture,
                execute=execute,
                before_execute=confirm_execution if execute and not args.yes else None,
            )
        else:
            execute = args.execute
            result = workflow.run(
                execute=execute, capture_only=args.capture_only,
                before_execute=confirm_execution if execute and not args.yes else None,
            )
        print(f"流程完成。执行模式={execute}，采集目录={result.get('capture_dir', '未保存')}")
        if "reuse_run_dir" in result:
            print(f"复用记录={result['reuse_run_dir']}")
        return 0
    except ExecutionCancelled as exc:
        logging.error("Execution cancelled: %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.error("Operator interrupted the workflow.")
        return 130
    except Exception as exc:
        logging.exception("Workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
