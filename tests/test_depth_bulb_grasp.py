import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from affordance.depth_bulb_grasp import (
    estimate_bulb_from_raw_depth,
    grasp_from_current_pose,
    load_saved_capture,
    move_to_observation_pose,
)
from affordance.three_stage_config import ThreeStageConfig


ROOT = Path(__file__).resolve().parents[1]


class DepthBulbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT / "config/three_stage_demo.yaml")

    def test_saved_current_raw_depth_bulb_center_is_close_to_server_region(self):
        capture = ROOT / "captures/20261103_111544_516661"
        color, depth, intrinsics, scale, transform = load_saved_capture(capture)
        estimate = estimate_bulb_from_raw_depth(
            color, depth, intrinsics, scale, transform,
            [180, 75, 240, 180], 0.20, 0.50,
        )
        self.assertGreater(estimate.depth_inlier_pixels, 1000)
        server = np.array([-0.553839671087233, 0.3543513237623491, 0.17506431486195895])
        self.assertLess(np.linalg.norm(estimate.surface_center_base_m - server), 0.012)

    def test_no_red_component_fails_before_motion(self):
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        depth = np.full((60, 80), 350, dtype=np.uint16)
        intr = {"width": 80, "height": 60, "fx": 100, "fy": 100, "cx": 40, "cy": 30}
        with self.assertRaisesRegex(ValueError, "No red/pink bulb"):
            estimate_bulb_from_raw_depth(color, depth, intr, 0.001, np.eye(4), [10, 10, 70, 50], .25, .4)

    def test_grasp_test_closes_but_never_extracts(self):
        robot, gripper, confirm = Mock(), Mock(), Mock()
        robot.actual_tcp_pose.return_value = [-0.31, 0.17, 0.23, 3.14, 0.0, 0.0]
        robot.has_inverse_kinematics.return_value = True
        robot.joints_within_safety_limits.return_value = True
        target = np.array([-0.55, 0.35, 0.165])
        grasp_from_current_pose(self.config, robot, gripper, target, confirm)
        self.assertEqual(robot.move_linear.call_count, 4)
        self.assertEqual(confirm.call_count, 3)
        gripper.set_position.assert_called_once_with(self.config.dropper.hold_position)
        gripper.disconnect.assert_called_once()

    def test_observation_move_uses_configured_reference_pose(self):
        robot, confirm = Mock(), Mock()
        current = [-0.31, 0.17, 0.23, 1.0, -2.5, -1.0]
        robot.actual_tcp_pose.return_value = current
        robot.has_inverse_kinematics.return_value = True
        robot.joints_within_safety_limits.return_value = True
        report = move_to_observation_pose(self.config, robot, confirm)
        self.assertEqual(len(report["waypoints"]), 3)
        self.assertEqual(robot.move_linear.call_count, 3)
        confirm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
