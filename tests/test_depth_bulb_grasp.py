import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from affordance.depth_bulb_grasp import (
    estimate_bulb_from_raw_depth,
    grasp_from_current_pose,
    load_saved_capture,
)
from affordance.three_stage_config import ThreeStageConfig


ROOT = Path(__file__).resolve().parents[1]


class DepthBulbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT / "config/three_stage_demo.yaml")

    def test_saved_raw_depth_bulb_center_matches_observed_region(self):
        capture = ROOT / "captures/20260922_210124_502296"
        color, depth, intrinsics, scale, transform = load_saved_capture(capture)
        estimate = estimate_bulb_from_raw_depth(
            color, depth, intrinsics, scale, transform,
            [345, 110, 400, 185], 0.25, 0.40,
        )
        self.assertGreater(estimate.depth_inlier_pixels, 800)
        np.testing.assert_allclose(estimate.surface_center_base_m[:2], [0.5458, -0.1744], atol=0.003)
        self.assertGreater(estimate.surface_center_base_m[2], 0.18)
        self.assertLess(estimate.surface_center_base_m[2], 0.20)

    def test_no_red_component_fails_before_motion(self):
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        depth = np.full((60, 80), 350, dtype=np.uint16)
        intr = {"width": 80, "height": 60, "fx": 100, "fy": 100, "cx": 40, "cy": 30}
        with self.assertRaisesRegex(ValueError, "No red/pink bulb"):
            estimate_bulb_from_raw_depth(color, depth, intr, 0.001, np.eye(4), [10, 10, 70, 50], .25, .4)

    def test_grasp_test_closes_but_never_extracts(self):
        robot, gripper, confirm = Mock(), Mock(), Mock()
        robot.actual_tcp_pose.return_value = [0.4, 0.0, 0.3, 3.14, 0.0, 0.0]
        target = np.array([0.545, -0.174, 0.181])
        grasp_from_current_pose(self.config, robot, gripper, target, confirm)
        self.assertEqual(robot.move_linear.call_count, 4)
        self.assertEqual(confirm.call_count, 3)
        gripper.set_position.assert_called_once_with(self.config.dropper.hold_position)
        gripper.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
