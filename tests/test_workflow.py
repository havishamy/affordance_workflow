import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from affordance.config import (
    AppConfig,
    Calibration,
    CameraConfig,
    GripperConfig,
    MotionConfig,
    RobotConfig,
    ServerConfig,
)
from affordance.workflow import TitrationWorkflow
from affordance.geometry import pose_to_matrix
from affordance.hardware import URRobot
from affordance.server import LocalizationClient
from test_server import sample_frame, world_result, response, bundle_response


class WorkflowPlanTests(unittest.TestCase):
    def make_config(self):
        return AppConfig(
            RobotConfig("127.0.0.1", 0.1, 0.1, 1.0, (0, 0, 0), (0, 0, 0, 0, 0, 0), 0.01),
            GripperConfig("dummy", True, 20, 100, 1000, 350, 0, 1),
            CameraConfig("dummy", 640, 480, 15, 0, 1000),
            ServerConfig("http://localhost", 1, 0, True, None, ee_frame="tcp"),
            MotionConfig(
                (3.14, 0, 0),
                (0, -1, 0),
                (1, 1, 1),
                0.5,
                (0, 0, 0),
                0.1,
                (0, 0, 0),
                0.05,
                0.1,
                0,
                0,
                0,
                1,
                0.5,
            ),
            Calibration("tcp", "camera", np.eye(4), True),
            Path(tempfile.gettempdir()),
            False,
        )

    def test_plan_uses_transit_height(self):
        workflow = TitrationWorkflow(self.make_config())
        plan = dict(
            workflow.build_plan(
                np.array([0.3, 0.1, 0.2]),
                np.array([0.6, 0.1, 0.2]),
                [0.4, 0.0, 0.3, 0, 0, 0],
            )
        )
        self.assertEqual(plan["above_rubber_bulb"][2], 0.5)
        self.assertEqual(plan["above_beaker"][2], 0.5)
        self.assertAlmostEqual(plan["dispense"][2], 0.25)

    def test_plan_rejects_outside_workspace(self):
        workflow = TitrationWorkflow(self.make_config())
        with self.assertRaises(RuntimeError):
            workflow.build_plan(
                np.array([1.2, 0.1, 0.2]),
                np.array([0.6, 0.1, 0.2]),
                [0.4, 0.0, 0.3, 0, 0, 0],
            )

    def make_workflow(self, root):
        config = replace(self.make_config(), output_dir=Path(root), save_capture=True)
        robot, gripper, camera = Mock(), Mock(), Mock()
        robot.actual_tcp_pose.return_value = [0.4, 0, 0.3, 0, 0, 0]
        camera.capture.return_value = sample_frame()
        return TitrationWorkflow(config, robot=robot, gripper=gripper, camera=camera)

    def test_full_plan_run_saves_upload_and_server_results_without_motion(self):
        with tempfile.TemporaryDirectory() as root, patch("affordance.server.requests.post", return_value=response({"job_id": "test-job"})), patch("affordance.server.requests.get", side_effect=[response({"state": "completed"}), response(world_result()), bundle_response()]):
            workflow = self.make_workflow(root)
            # A nonidentity calibration and TCP pose must not transform world/base points again.
            workflow.config.calibration.transform_parent_camera[:3, 3] = [10, 20, 30]
            record = workflow.run(execute=False)
            capture_dir = Path(record["capture_dir"])
            for name in ("rgb.png", "depth.png", "camera.json", "T_world_ee.json", "capture_info.json", "metadata.json", "server_result/result.json", "server_result/titration_test-job_results.zip"):
                self.assertTrue((capture_dir / name).exists(), name)
            self.assertEqual(record["positions_base_m"]["rubber_bulb"], [0.3, 0.1, 0.2])
            workflow.robot.move_linear.assert_not_called()
            workflow.gripper.connect.assert_not_called()

    def test_capture_only_never_calls_server(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.make_workflow(root)
            workflow.client = Mock()
            record = workflow.run(execute=False, capture_only=True)
            workflow.client.locate.assert_not_called()
            workflow.robot.move_linear.assert_not_called()
            workflow.gripper.connect.assert_not_called()
            payload = json.loads((Path(record["capture_dir"]) / "T_world_ee.json").read_text())
            np.testing.assert_allclose(np.array(payload["T_world_ee"])[:3, 3], [0.4, 0, 0.3])

    def test_failure_keeps_capture_without_gripper_initialization(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.make_workflow(root)
            workflow.client = Mock()
            workflow.client.locate.side_effect = RuntimeError("server failed")
            with self.assertRaisesRegex(RuntimeError, "server failed"):
                workflow.run(execute=True)
            capture_dir = next(Path(root).iterdir())
            self.assertTrue((capture_dir / "rgb.png").exists())
            self.assertTrue((capture_dir / "failure.json").exists())
            workflow.gripper.connect.assert_not_called()
            workflow.robot.move_linear.assert_not_called()

    def test_operator_confirmation_happens_after_localization_before_gripper(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.make_workflow(root)
            workflow.client = Mock()
            workflow.client.locate.return_value = LocalizationClient._parse_world_response(world_result())
            def decline():
                workflow.client.locate.assert_called_once()
                workflow.gripper.connect.assert_not_called()
                self.assertTrue(list(Path(root).glob("*/metadata.json")))
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                workflow.run(execute=True, before_execute=decline)
            workflow.robot.move_linear.assert_not_called()

    def test_pose_change_during_capture_or_wait_blocks_motion(self):
        initial = [0.4, 0, 0.3, 0, 0, 0]
        changed = [0.5, 0, 0.3, 0, 0, 0]
        for poses in ([initial, changed], [initial, initial, changed]):
            with self.subTest(poses=poses), tempfile.TemporaryDirectory() as root:
                workflow = self.make_workflow(root)
                workflow.robot.actual_tcp_pose.side_effect = poses
                workflow.client = Mock()
                workflow.client.locate.return_value = LocalizationClient._parse_world_response(world_result())
                with self.assertRaisesRegex(RuntimeError, "pose changed"):
                    workflow.run(execute=True)
                workflow.gripper.connect.assert_not_called()
                workflow.robot.move_linear.assert_not_called()

    def test_flange_upload_removes_tcp_offset_including_rotation(self):
        config = self.make_config()
        config = replace(config, robot=replace(config.robot, tcp_offset=(0, 0, 0.15, 0, 0, np.pi / 2)), server=replace(config.server, ee_frame="flange"))
        workflow = TitrationWorkflow(config)
        tcp_pose = [0.4, 0.1, 0.3, np.pi, 0, 0]
        np.testing.assert_allclose(
            workflow._world_ee_matrix(tcp_pose) @ pose_to_matrix(config.robot.tcp_offset),
            pose_to_matrix(tcp_pose), atol=1e-9,
        )

    def test_unknown_ee_and_unvalidated_calibration_block_before_hardware(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.make_workflow(root)
            workflow.config = replace(workflow.config, server=replace(workflow.config.server, ee_frame=None))
            with self.assertRaisesRegex(RuntimeError, "server.ee_frame"):
                workflow.run(execute=False)
            workflow.robot.connect.assert_not_called()
            workflow.config = replace(workflow.config, server=replace(workflow.config.server, ee_frame="tcp"), calibration=replace(workflow.config.calibration, validated=False))
            with self.assertRaisesRegex(RuntimeError, "not marked validated"):
                workflow.run(execute=True)
            workflow.robot.connect.assert_not_called()

    def test_tcp_configuration_failure_stops_connection(self):
        control, receive = Mock(), Mock()
        control.setTcp.return_value = False
        with patch.dict("sys.modules", {
            "rtde_control": Mock(RTDEControlInterface=Mock(return_value=control)),
            "rtde_receive": Mock(RTDEReceiveInterface=Mock(return_value=receive)),
        }):
            with self.assertRaisesRegex(RuntimeError, "setTcp returned False"):
                URRobot(self.make_config().robot).connect()
        control.setPayload.assert_not_called()
        control.disconnect.assert_called_once()
        receive.disconnect.assert_called_once()

    @staticmethod
    def write_saved_capture(root: str, tcp_offset=(0, 0, 0, 0, 0, 0)) -> Path:
        source = Path(root) / "capture_saved"
        (source / "server_result").mkdir(parents=True)
        (source / "capture_info.json").write_text(json.dumps({
            "capture_id": "capture_saved",
            "world_frame": "base",
            "ee_frame": "flange",
            "tcp_offset": list(tcp_offset),
        }))
        (source / "server_result" / "job.json").write_text(json.dumps({
            "job_id": "test-job", "state": "completed",
        }))
        (source / "server_result" / "result.json").write_text(json.dumps(world_result()))
        (source / "metadata.json").write_text(json.dumps({
            "positions_base_m": {
                "rubber_bulb": [0.3, 0.1, 0.2],
                "beaker_mouth": [0.6, 0.1, 0.2],
            }
        }))
        return source

    def test_saved_plan_skips_camera_server_gripper_and_rebuilds_from_current_pose(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.write_saved_capture(root)
            workflow = self.make_workflow(root)
            workflow.client = Mock()
            workflow.robot.actual_tcp_pose.return_value = [0.25, 0, 0.3, 0, 0, 0]
            record = workflow.run_saved(source, execute=False)
            workflow.camera.connect.assert_not_called()
            workflow.camera.capture.assert_not_called()
            workflow.client.locate.assert_not_called()
            workflow.gripper.connect.assert_not_called()
            workflow.robot.move_linear.assert_not_called()
            self.assertEqual(record["motion_plan"][0]["pose"][:3], [0.25, 0.0, 0.5])
            self.assertTrue((Path(record["reuse_run_dir"]) / "plan.json").is_file())

    def test_saved_execution_confirms_before_gripper_and_motion(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.write_saved_capture(root)
            workflow = self.make_workflow(root)
            workflow.client = Mock()
            confirmed = Mock()
            record = workflow.run_saved(source, execute=True, before_execute=confirmed)
            confirmed.assert_called_once()
            workflow.camera.connect.assert_not_called()
            workflow.client.locate.assert_not_called()
            workflow.gripper.connect.assert_called_once()
            self.assertEqual(workflow.robot.move_linear.call_count, 7)
            self.assertTrue((Path(record["reuse_run_dir"]) / "execution.json").is_file())

    def test_saved_capture_allows_tcp_change_and_records_replanned_offset(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.write_saved_capture(root, tcp_offset=(0, 0, 0.1, 0, 0, 0))
            workflow = self.make_workflow(root)
            with self.assertLogs("affordance.workflow", level="WARNING") as logs:
                record = workflow.run_saved(source, execute=False)
            self.assertTrue(record["tcp_offset_changed_since_capture"])
            self.assertEqual(record["source_tcp_offset"], [0, 0, 0.1, 0, 0, 0])
            self.assertEqual(record["current_tcp_offset"], [0, 0, 0, 0, 0, 0])
            self.assertIn("rebuilding every waypoint", " ".join(logs.output))
            workflow.gripper.connect.assert_not_called()

    def test_saved_capture_rejects_invalid_tcp_metadata_before_hardware(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.write_saved_capture(root)
            info_path = source / "capture_info.json"
            info = json.loads(info_path.read_text())
            info["tcp_offset"] = [0, 0]
            info_path.write_text(json.dumps(info))
            workflow = self.make_workflow(root)
            with self.assertRaisesRegex(ValueError, "valid six-value TCP offset"):
                workflow.run_saved(source, execute=False)
            workflow.robot.connect.assert_not_called()

    def test_saved_capture_rejects_incomplete_or_inconsistent_results(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.write_saved_capture(root)
            workflow = self.make_workflow(root)
            job_path = source / "server_result" / "job.json"
            job_path.write_text(json.dumps({"job_id": "test-job", "state": "failed"}))
            with self.assertRaisesRegex(ValueError, "not completed"):
                workflow.run_saved(source, execute=False)
            workflow.robot.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
