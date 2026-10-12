import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from affordance.camera import CapturedFrame
from affordance.geometry import pose_to_matrix
from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_observation import (
    ThreeStageObservation,
    _safe_extract_bundle,
    observation_pose_for_current_tcp,
    reuse_latest_result,
)
from run_three_stage_demo import parser


ROOT = Path(__file__).resolve().parents[1]


def sample_frame():
    return CapturedFrame(
        color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_u16=np.full((2, 3), 1000, dtype=np.uint16),
        intrinsics={"width": 3, "height": 2, "fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
        depth_scale_m=0.001,
        camera_serial="mock-camera",
        captured_at_unix_s=123.0,
    )


class FakeHealth:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        return None


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT / "config/three_stage_demo.yaml")

    def test_default_mode_is_saved_result(self):
        self.assertEqual(self.config.observation.enabled, 0)
        self.assertIsNone(parser().parse_args([]).observe)
        self.assertEqual(parser().parse_args(["--observe", "0"]).observe, 0)
        self.assertEqual(parser().parse_args(["--observe", "1"]).observe, 1)

    def test_historical_flange_pose_is_preserved_when_tcp_offset_changes(self):
        info = json.loads(self.config.observation.reference_capture_info.read_text())
        target = observation_pose_for_current_tcp(self.config)
        saved_flange = pose_to_matrix(info["actual_tcp_pose"]) @ np.linalg.inv(pose_to_matrix(info["tcp_offset"]))
        new_flange = pose_to_matrix(target) @ np.linalg.inv(pose_to_matrix(self.config.base.robot.tcp_offset))
        np.testing.assert_allclose(new_flange, saved_flange, atol=1e-8)

    def test_missing_token_stops_before_robot_connection(self):
        robot = Mock()
        observer = ThreeStageObservation(self.config, robot=robot)
        with patch.dict("os.environ", {"LAG_API_TOKEN": ""}):
            with self.assertRaisesRegex(ValueError, "LAG_API_TOKEN"):
                observer.run(lambda *_: None)
        robot.connect.assert_not_called()

    def test_observe_capture_upload_and_extract_with_mocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = replace(self.config.base, output_dir=Path(temporary))
            observation = replace(
                self.config.observation,
                latest_result_manifest=Path(temporary) / "latest.json",
            )
            config = replace(
                self.config, base=base, observation=observation,
                physical_execution_validated=True,
            )
            robot, camera = Mock(), Mock()
            start = [0.50, -0.20, 0.30, 3.14, 0, 0]
            target = observation_pose_for_current_tcp(config)
            robot.actual_tcp_pose.side_effect = [start, start, target, target, target]
            camera.capture.return_value = sample_frame()
            calls = []

            def fake_submit(args):
                calls.append(args)
                self.assertEqual(args.scope, "all")
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True)
                (output_dir / "job.json").write_text(json.dumps({"job_id": "test-job", "state": "completed"}))
                (output_dir / "result.json").write_text(json.dumps({"job_id": "test-job"}))
                with zipfile.ZipFile(output_dir / "complex_test-job_all.zip", "w") as archive:
                    archive.writestr("bundle_manifest.json", "{}")
                return 0

            observer = ThreeStageObservation(
                config,
                robot=robot,
                camera=camera,
                submitter=fake_submit,
                health_get=lambda *_, **__: FakeHealth(),
            )
            fake_cloud = np.array([[0.0, 0.0, 0.10], [0.0, 0.0, 0.19]])
            with patch.dict("os.environ", {"LAG_API_TOKEN": "test-only-token"}):
                with patch("affordance.three_stage_observation._read_ply_xyz", return_value=fake_cloud):
                    with patch(
                        "affordance.three_stage_observation.load_scene",
                        return_value=SimpleNamespace(source_job_id="test-job"),
                    ):
                        updated = observer.run(lambda *_: None)
            self.assertEqual(robot.move_linear.call_count, 3)
            camera.capture.assert_called_once()
            robot.disconnect.assert_called_once()
            camera.disconnect.assert_called_once()
            self.assertEqual(len(calls), 1)
            self.assertTrue((updated.capture_dir / "rgb.png").is_file())
            self.assertTrue((updated.capture_dir / "depth.png").is_file())
            self.assertTrue((updated.result_dir / "result.json").is_file())
            self.assertTrue(config.observation.latest_result_manifest.is_file())
            with patch(
                "affordance.three_stage_observation.load_scene",
                return_value=SimpleNamespace(source_job_id="test-job"),
            ):
                reused = reuse_latest_result(config)
            self.assertEqual(reused.result_dir, updated.result_dir)

    def test_rejects_unsafe_bundle_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
                _safe_extract_bundle(bundle, Path(temporary) / "extract")


if __name__ == "__main__":
    unittest.main()
