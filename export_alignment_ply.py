#!/usr/bin/env python3
"""Merge observed depth and all single/dual-dropper scene models into colored PLY/GLB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from affordance.alignment_export import export_alignment_ply


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--capture-dir", help="Capture directory; defaults to captures/three_stage_latest_result.json")
    command.add_argument("--result-dir", help="Extracted complex result directory; defaults to the latest manifest")
    command.add_argument("--output", help="Output PLY path; defaults under the result diagnostics directory")
    command.add_argument("--calibration", default=str(ROOT / "config" / "T_ee_cam.json"))
    command.add_argument("--stride", type=int, default=1, help="Depth pixel stride; 1 preserves every valid pixel")
    command.add_argument("--min-depth", type=float, default=0.05)
    command.add_argument("--max-depth", type=float, default=2.0)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        if bool(args.capture_dir) != bool(args.result_dir):
            raise ValueError("Pass both --capture-dir and --result-dir, or omit both to use the latest manifest.")
        if args.capture_dir:
            capture_dir = Path(args.capture_dir).expanduser().resolve()
            result_dir = Path(args.result_dir).expanduser().resolve()
        else:
            manifest_path = ROOT / "captures" / "three_stage_latest_result.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            capture_dir = Path(manifest["capture_dir"]).expanduser().resolve()
            result_dir = Path(manifest["result_dir"]).expanduser().resolve()
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else result_dir / "diagnostics" / "depth_model_alignment_world.ply"
        )
        report = export_alignment_ply(
            capture_dir, result_dir, output, Path(args.calibration),
            stride=args.stride, min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        )
        print(f"PLY: {report.output_path}")
        print(f"GLB: {report.glb_path}")
        print(f"Manifest: {report.manifest_path}")
        print(f"Observed depth points: {report.depth_points}")
        for name, count in report.model_points.items():
            print(f"{name} model points: {count}")
        print(f"Local/server transform max abs difference: {report.local_transform_max_abs_difference:.3e}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
