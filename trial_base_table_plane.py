#!/usr/bin/env python3
"""Export a base z=0 initialization-reference request; never transform a scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from affordance.base_table_plane import TABLE_REFERENCE_PARAMETER, table_initialization_reference


def run_trial(capture_dir: Path, result_dir: Path, output_dir: Path) -> dict:
    capture_dir, result_dir, output_dir = (p.expanduser().resolve() for p in (capture_dir, result_dir, output_dir))
    result = json.loads((result_dir / "result.json").read_text())
    if result.get("capture_id") != capture_dir.name:
        raise ValueError("Capture/result IDs do not match.")
    if output_dir == result_dir or result_dir in output_dir.parents:
        raise ValueError("Keep the reference export separate from the source result.")
    reference = table_initialization_reference()
    report = {
        "capture_dir": str(capture_dir), "source_result_dir": str(result_dir),
        "status": "initialization_reference_only_not_submitted",
        "postsolve_scene_transform_applied": False,
        "original_model_files_changed": False,
        "requested_server_parameter": TABLE_REFERENCE_PARAMETER,
        "table_plane_reference": reference,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "server_plane_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.capture_dir / "table_initialization_reference"
    run_trial(args.capture_dir, args.result_dir, output)
    print(f"Initialization reference only: {output.resolve()}; no solved coordinates changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
