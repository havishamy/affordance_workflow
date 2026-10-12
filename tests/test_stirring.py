import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from affordance.stirring import StirringWorkflow, validate_stirring_result
from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_result import load_scene


ROOT = Path(__file__).resolve().parents[1]


def sample_result():
    endpoint_a = np.array([0.683, -0.089, 0.0225])
    endpoint_b = np.array([0.487, -0.066, 0.0225])
    direction = endpoint_b - endpoint_a
    length = float(np.linalg.norm(direction))
    direction /= length
    inset = 0.025
    return {
        "schema_version": "stirring-geometry-v1",
        "job_id": "test-stirring-job",
        "capture_id": "test-capture",
        "frame": "world",
        "units": "meters",
        "assumptions": {
            "glass_rod_lies_on_table": True,
            "endpoint_a_b_have_no_task_semantics": True,
            "centerline_is_offset_from_table_by_rod_radius_m": 0.0025,
        },
        "world_up": [0, 0, 1],
        "world_down": [0, 0, -1],
        "calibration": {"T_world_opencv_camera": np.eye(4).tolist(), "details": {}},
        "glass_rod": {
            "class_id": 12,
            "class_name": "glass rod",
            "segmentation_confidence": 0.923,
            "mask_artifact": "00_segmentation/glass_rod/mask.png",
            "mask_probability_artifact": "00_segmentation/glass_rod/mask_prob.png",
            "segmentation_overlay_artifact": "00_segmentation/glass_rod/overlay.png",
            "geometry_overlay_artifact": "glass_rod/geometry_overlay.png",
            "endpoint_a_world": endpoint_a.tolist(),
            "endpoint_b_world": endpoint_b.tolist(),
            "center_world": ((endpoint_a + endpoint_b) / 2).tolist(),
            "axis_world_unoriented": direction.tolist(),
            "length_m": length,
            "grasp_candidates": [
                {
                    "id": "grasp_near_a",
                    "grasp_point_world": (endpoint_a + direction * inset).tolist(),
                    "opposite_tip_id": "endpoint_b",
                    "opposite_tip_world": endpoint_b.tolist(),
                    "rod_direction_grasp_to_tip_world": direction.tolist(),
                },
                {
                    "id": "grasp_near_b",
                    "grasp_point_world": (endpoint_b - direction * inset).tolist(),
                    "opposite_tip_id": "endpoint_a",
                    "opposite_tip_world": endpoint_a.tolist(),
                    "rod_direction_grasp_to_tip_world": (-direction).tolist(),
                },
            ],
            "quality": {
                "valid": True,
                "mask_elongation": 18.34,
                "table_plane_inlier_fraction": 0.9689,
                "table_plane_rmse_m": 0.00118,
                "endpoint_uncertainty_m": 0.00268,
            },
        },
    }


class StirringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT / "config/three_stage_demo.yaml")
        cls.scene = load_scene(cls.config)

    def configured_motion(self):
        return replace(
            self.config,
            stirring_motion=replace(
                self.config.stirring_motion,
                validated=True,
                pregrasp_height_m=0.05,
                lift_height_m=0.22,
                preinsert_clearance_m=0.04,
                target_depth_below_mouth_m=0.02,
                stir_radius_m=0.003,
                stir_speed_mps=0.01,
                beaker_safe_inner_radius_m=0.020,
                bottom_clearance_m=0.015,
                rod_grasp_center_tool_m=(0.0, 0.0, 0.015),
            ),
        )

    def test_contract_and_quality_are_hard_gates(self):
        geometry = validate_stirring_result(sample_result(), self.config, "test-capture", np.eye(4))
        self.assertEqual(len(geometry.candidates), 2)
        self.assertAlmostEqual(geometry.length_m, 0.1973, delta=0.002)
        bad = sample_result()
        bad["glass_rod"]["quality"]["endpoint_uncertainty_m"] = 0.02
        with self.assertRaisesRegex(ValueError, "quality gates"):
            validate_stirring_result(bad, self.config, "test-capture", np.eye(4))
        with self.assertRaisesRegex(ValueError, "hand-eye"):
            shifted = np.eye(4)
            shifted[0, 3] = 0.01
            validate_stirring_result(sample_result(), self.config, "test-capture", shifted)

    def test_plans_vertical_tip_and_bounded_stir_circle(self):
        config = self.configured_motion()
        geometry = validate_stirring_result(sample_result(), config, "test-capture", np.eye(4))
        workflow = StirringWorkflow(config, self.scene, geometry, Path("/tmp/not-used"))
        plans = workflow.plan_candidates([0.55, -0.15, 0.30, 3.14, 0, 0])
        self.assertGreaterEqual(len(plans), 1)
        plan = plans[0]
        insert = next(step for step in plan.steps if step.name == "insert_rod_shallow")
        vertical = next(step for step in plan.steps if step.name == "rod_tip_preinsert")
        stir = [step for step in plan.steps if step.name.startswith("stir_")]
        self.assertEqual(len(stir), config.stirring_motion.stir_points_per_cycle)
        checkpoints = [step.name for step in plan.steps if step.kind == "checkpoint"]
        self.assertEqual(checkpoints, ["grasp", "insert"])
        grasp_index = next(i for i, step in enumerate(plan.steps) if step.name == "rod_grasp")
        close_index = next(i for i, step in enumerate(plan.steps) if step.name == "close_on_rod")
        self.assertEqual(plan.steps[grasp_index + 1].name, checkpoints[0])
        self.assertLess(grasp_index, close_index)
        preinsert_index = next(i for i, step in enumerate(plan.steps) if step.name == "rod_tip_preinsert")
        insert_index = next(i for i, step in enumerate(plan.steps) if step.name == "insert_rod_shallow")
        self.assertEqual(plan.steps[preinsert_index + 1].name, checkpoints[1])
        self.assertLess(preinsert_index, insert_index)
        self.assertAlmostEqual(insert.pose[2] - vertical.pose[2], -0.06, places=6)
        center_xy = np.mean(np.asarray([step.pose[:2] for step in stir]), axis=0)
        radii = np.linalg.norm(np.asarray([step.pose[:2] for step in stir]) - center_xy, axis=1)
        np.testing.assert_allclose(radii, config.stirring_motion.stir_radius_m, atol=1e-6)

    def test_unmeasured_motion_blocks_before_robot_connection(self):
        disabled = replace(
            self.config,
            stirring_motion=replace(self.config.stirring_motion, validated=False),
        )
        geometry = validate_stirring_result(sample_result(), disabled, "test-capture", np.eye(4))
        robot = Mock()
        workflow = StirringWorkflow(
            disabled, self.scene, geometry, Path("/tmp/not-used"), robot=robot,
        )
        with self.assertRaisesRegex(RuntimeError, "stirring_motion.validated"):
            workflow.execute(Path("/tmp/not-used"), lambda *_: None)
        robot.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
