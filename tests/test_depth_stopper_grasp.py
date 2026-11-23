import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from affordance.depth_stopper_grasp import DirectDepthStopperWorkflow, DirectStopperConfig, estimate_stopper
from affordance.three_stage_config import ThreeStageConfig


class DirectDepthStopperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = ThreeStageConfig.load(Path(__file__).parents[1] / "config" / "three_stage_demo.yaml")

    def config(self, roi=(20, 10, 60, 50)):
        return DirectStopperConfig(
            source_path=Path(__file__).parents[1] / "config" / "depth_stopper_grasp.yaml",
            three_stage=self.stage,
            roi_px=roi,
            saturation_max=80,
            value_min=140,
            nearest_depth_percentile=10,
            depth_below_percentile_m=0.005,
            depth_above_percentile_m=0.015,
            minimum_component_pixels=100,
            depth_min_m=0.25,
            depth_max_m=0.8,
            grasp_offset_base_m=(0, 0, -0.01),
            grasp_speed_mps=0.02,
            grasp_acceleration_mps2=0.02,
            settle_s=0,
        )

    def test_white_near_component_becomes_world_grasp_target(self):
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        color[:] = (40, 80, 40)
        cv2.rectangle(color, (30, 20), (49, 39), (240, 240, 240), -1)
        depth = np.full((60, 80), 600, dtype=np.uint16)
        depth[20:40, 30:50] = 400
        intr = {"width": 80, "height": 60, "fx": 100.0, "fy": 100.0, "cx": 40.0, "cy": 30.0}
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        estimate = estimate_stopper(color, depth, intr, 0.001, transform, self.config())
        self.assertGreaterEqual(estimate.selected_pixels, 300)
        np.testing.assert_allclose(estimate.surface_center_camera_m, [-0.002, -0.002, 0.4], atol=0.005)
        np.testing.assert_allclose(estimate.grasp_target_base_m, [0.998, 1.998, 3.39], atol=0.005)

    def test_grasp_plan_uses_existing_stopper_parameters(self):
        robot = Mock()
        workflow = DirectDepthStopperWorkflow(self.config(), robot=robot, camera=Mock(), gripper=Mock())
        current = [-0.31, 0.17, 0.23, 1.0, -2.5, -1.0]
        target = np.array([-0.50, 0.42, 0.075])
        plan = workflow._grasp_plan(current, target)
        self.assertEqual(plan["grasp"][:3], target.tolist())
        self.assertAlmostEqual(plan["pregrasp"][2], target[2] + self.stage.stopper.approach_m)
        self.assertAlmostEqual(plan["pull"][2], target[2] + self.stage.stopper.pull_distance_m)
        self.assertEqual(plan["grasp"][3:], list(self.stage.stopper.orientation_rotvec))


if __name__ == "__main__":
    unittest.main()

