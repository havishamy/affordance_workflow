#!/usr/bin/env python3
"""Move to the fixed view, locate the white stopper from aligned raw depth, and optionally grasp/pull it."""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

from affordance.depth_stopper_grasp import DirectDepthStopperWorkflow, DirectStopperConfig


def _normalized(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return "".join(char for char in text if unicodedata.category(char) != "Cf").strip().upper()


def confirm(label: str, message: str) -> None:
    phrase = "EXECUTE " + label.upper()
    print(f"\n{message}\n当前步骤: {label}\n确认急停可用、工作区无人。")
    if _normalized(input(f"输入 {phrase} 继续；其他输入取消: ")) != phrase:
        raise RuntimeError(f"Operator cancelled at {label}.")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--config", default=str(Path(__file__).parent / "config" / "depth_stopper_grasp.yaml"),
    )
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture-only", action="store_true",
        help="Move to the fixed observation pose, capture/localize/plan, but do not connect the gripper or grasp.",
    )
    mode.add_argument(
        "--execute", action="store_true",
        help="Move, recapture, then grasp and vertically pull the stopper after typed confirmations.",
    )
    command.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return command


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = DirectStopperConfig.load(args.config)
        result = DirectDepthStopperWorkflow(config).run(execute=args.execute, confirm=confirm)
        estimate = result["estimate"]
        print(f"采集目录: {result['capture_dir']}")
        print(f"瓶塞遮罩: {result['overlay']}")
        print(f"可见顶面 base(m): {[round(value, 5) for value in estimate['surface_center_base_m']]}")
        print(f"抓取目标 base(m): {[round(value, 5) for value in estimate['grasp_target_base_m']]}")
        print(f"有效白色深度像素: {estimate['white_depth_pixels']}，选中瓶塞像素: {estimate['selected_pixels']}")
        if result["executed"]:
            print("原始深度瓶塞抓取/竖直拔起已完成；夹爪保持闭合并继续夹持瓶塞。")
        else:
            print("仅完成观察、原始深度定位和规划；未连接夹爪，未执行抓取。")
        return 0
    except KeyboardInterrupt:
        print("操作者中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.exception("Direct raw-depth stopper workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

