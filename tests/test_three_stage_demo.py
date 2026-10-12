import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from affordance.geometry import rotvec_to_matrix
from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_result import _circle_from_partial_rim, load_scene
from affordance.three_stage_workflow import CHEMISTRY_STAGES, STAGES, ThreeStageWorkflow


ROOT = Path(__file__).resolve().parents[1]


class ThreeStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT / "config/three_stage_demo.yaml")
        cls.scene = load_scene(cls.config)

    def test_actual_result_quality_and_beaker_arc_fit(self):
        scene = self.scene
        self.assertEqual(scene.source_capture_id, "20260921_171530_302852")
        self.assertEqual(scene.source_job_id, "df530c2b-86be-4cc5-82b1-646908ab7886")
        self.assertGreater(scene.stopper_grasp[2] - scene.bottle_body_grasp[2], 0.035)
        self.assertAlmostEqual(scene.beaker_mouth_center[0], 0.6509, delta=0.002)
        self.assertAlmostEqual(scene.beaker_mouth_center[1], -0.0224, delta=0.002)
        self.assertAlmostEqual(scene.beaker_inner_radius_estimate_m, 0.0297, delta=0.002)
        report = scene.quality_report["beaker_circle_fit"]
        self.assertLess(report["residual_rms_m"], 0.001)
        # The raw region is a partial side arc: its median would target the rim.
        raw = np.asarray(report["raw_region_median_base_m"])
        self.assertGreater(np.linalg.norm(raw[:2] - scene.beaker_mouth_center[:2]), 0.02)

    def test_circle_fit_rejects_degenerate_arc(self):
        points = np.array([[float(x), 0, 0.1] for x in range(10)])
        with self.assertRaises(ValueError):
            _circle_from_partial_rim(points)

    def test_plans_all_stages_without_hardware(self):
        robot, gripper = Mock(), Mock()
        workflow = ThreeStageWorkflow(self.config, robot=robot, gripper=gripper)
        plans = workflow.plan_all(self.scene, workflow.captured_tcp_pose())
        self.assertEqual(STAGES, (*CHEMISTRY_STAGES, "stirring"))
        self.assertEqual(tuple(plan.name for plan in plans), CHEMISTRY_STAGES)
        self.assertEqual([sum(step.kind == "moveL" for step in plan.steps) for plan in plans], [10, 45, 15])
        self.assertEqual(sum(step.name.startswith("squeeze_") for step in plans[2].steps), 1)
        robot.connect.assert_not_called()
        robot.move_linear.assert_not_called()
        gripper.connect.assert_not_called()

    def test_changed_tcp_offset_warns_but_does_not_block_saved_world_targets(self):
        changed = replace(
            self.config,
            base=replace(
                self.config.base,
                robot=replace(self.config.base.robot, tcp_offset=(0, 0, 0.13, 0, 0, 0)),
            ),
        )
        workflow = ThreeStageWorkflow(changed)
        with self.assertLogs("affordance.three_stage_workflow", level="WARNING") as logs:
            pose = workflow.captured_tcp_pose()
        self.assertEqual(len(pose), 6)
        self.assertIn("TCP offset changed since capture", " ".join(logs.output))

    def test_bottle_mouth_pivot_and_tilt_toward_beaker(self):
        workflow = ThreeStageWorkflow(self.config)
        plan = workflow.plan_stage("bottle", self.scene, workflow.captured_tcp_pose())
        cfg = self.config.bottle
        grasp = self.scene.bottle_body_grasp + self.scene.bottle_axis_out * cfg.grasp_offset_along_axis_m
        mouth = self.scene.bottle_mouth_approx
        upright = rotvec_to_matrix(cfg.orientation_rotvec)
        mouth_tool = upright.T @ (mouth - grasp)
        bottle_to_beaker = self.scene.beaker_mouth_center[:2] - grasp[:2]
        bottle_to_beaker /= np.linalg.norm(bottle_to_beaker)
        tilt_steps = [step for step in plan.steps if step.name.startswith("tilt_high_")]
        self.assertEqual(len(tilt_steps), cfg.tilt_steps)
        pivots = []
        for step in tilt_steps:
            tcp = np.asarray(step.pose[:3])
            rotation = rotvec_to_matrix(step.pose[3:])
            pivots.append(tcp + rotation @ mouth_tool)
        for pivot in pivots:
            np.testing.assert_allclose(pivot, pivots[0], atol=1e-6)
        last_rot = rotvec_to_matrix(tilt_steps[-1].pose[3:])
        tilted_axis = last_rot @ upright.T @ self.scene.bottle_axis_out
        self.assertAlmostEqual(
            float(tilted_axis[:2] @ bottle_to_beaker),
            math.sin(math.radians(cfg.tilt_angle_deg)),
            places=5,
        )
        self.assertAlmostEqual(float(tilted_axis[2]), math.cos(math.radians(cfg.tilt_angle_deg)), places=5)

    def test_dropper_tip_clearance_uses_measured_length_parameter(self):
        workflow = ThreeStageWorkflow(self.config)
        plan = workflow.plan_stage("dropper", self.scene, workflow.captured_tcp_pose())
        dose = next(step.pose for step in plan.steps if step.name == "dropper_dose_position")
        expected = (
            self.scene.beaker_mouth_center[2]
            + self.config.dropper.tip_below_bulb_m
            + self.config.dropper.grasp_offset_z_m
            + self.config.dropper.tip_clearance_above_beaker_m
        )
        self.assertAlmostEqual(dose[2], expected, places=8)
        effective_tip_length = self.config.dropper.tip_below_bulb_m + self.config.dropper.grasp_offset_z_m
        self.assertAlmostEqual(
            dose[2] - effective_tip_length,
            self.scene.beaker_mouth_center[2] + self.config.dropper.tip_clearance_above_beaker_m,
            places=8,
        )

    def test_dropper_grasp_is_shifted_down_without_changing_nominal_affordance(self):
        workflow = ThreeStageWorkflow(self.config)
        plan = workflow.plan_stage("dropper", self.scene, workflow.captured_tcp_pose())
        grasp = next(step.pose for step in plan.steps if step.name == "dropper_bulb_grasp")
        np.testing.assert_allclose(grasp[:2], self.scene.dropper_bulb_grasp[:2])
        self.assertAlmostEqual(
            grasp[2] - self.scene.dropper_bulb_grasp[2],
            self.config.dropper.grasp_offset_z_m,
            places=8,
        )

    def test_execution_gate_prevents_hardware_connection(self):
        robot, gripper = Mock(), Mock()
        workflow = ThreeStageWorkflow(
            replace(self.config, physical_execution_validated=False), robot=robot, gripper=gripper,
        )
        with self.assertRaisesRegex(RuntimeError, "physical_execution_validated"):
            workflow.execute(self.scene, STAGES, lambda *_: None)
        robot.connect.assert_not_called()
        gripper.connect.assert_not_called()

    def test_stage_plan_starts_from_arbitrary_current_pose(self):
        current = [0.66, -0.22, 0.17, 3.14, 0, 0]
        workflow = ThreeStageWorkflow(self.config)
        plan = workflow.plan_stage("stopper", self.scene, current)
        first_move = next(step for step in plan.steps if step.kind == "moveL")
        self.assertAlmostEqual(first_move.pose[0], current[0])
        self.assertAlmostEqual(first_move.pose[1], current[1])
        self.assertAlmostEqual(first_move.pose[2], max(current[2], self.config.transit_z_m))

    def test_execution_does_not_require_capture_or_history_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            isolated = replace(
                self.config, result_dir=Path(temporary), physical_execution_validated=True,
                transit_z_m=0.4,
            )
            robot, gripper = Mock(), Mock()
            robot.actual_tcp_pose.return_value = [0.66, -0.22, 0.17, 3.14, 0, 0]
            workflow = ThreeStageWorkflow(isolated, robot=robot, gripper=gripper)
            fake_cloud = np.array([[0, 0, 0.1], [0.01, 0, 0.2]])
            with patch("affordance.three_stage_workflow._read_ply_xyz", return_value=fake_cloud):
                with self.assertRaisesRegex(RuntimeError, "operator cancelled"):
                    workflow.execute(
                        self.scene, ("stopper",),
                        lambda *_: (_ for _ in ()).throw(RuntimeError("operator cancelled")),
                    )
            robot.connect.assert_called_once()
            robot.move_linear.assert_not_called()
            gripper.connect.assert_not_called()

    def test_low_transit_height_warns_at_operator_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            robot = Mock()
            robot.actual_tcp_pose.return_value = [0.66, -0.22, 0.17, 3.14, 0, 0]
            config = replace(
                self.config, result_dir=Path(temporary),
                physical_execution_validated=True, transit_z_m=0.1,
            )
            workflow = ThreeStageWorkflow(config, robot=robot)
            prompts = []

            def decline(label, message):
                prompts.append((label, message))
                raise RuntimeError("operator cancelled")

            cloud = np.array([[0, 0, 0.1], [0.01, 0, 0.2]])
            with patch("affordance.three_stage_workflow._read_ply_xyz", return_value=cloud):
                with self.assertRaisesRegex(RuntimeError, "operator cancelled"):
                    workflow.execute(self.scene, ("stopper",), decline)
            robot.connect.assert_called_once()
            robot.move_linear.assert_not_called()
            self.assertIn("model-top+80 mm advisory", prompts[0][1])

    def test_low_result_quality_becomes_visible_advisory(self):
        strict = replace(
            self.config,
            quality=replace(self.config.quality, minimum_silhouette_iou=0.95),
        )
        with self.assertLogs("affordance.three_stage_result", level="WARNING"):
            scene = load_scene(strict)
        self.assertTrue(any("silhouette IoU" in warning for warning in scene.quality_report["warnings"]))

    def test_new_result_side_offset_is_advisory_not_bottle_blocker(self):
        capture = ROOT / "captures/20260922_210124_502296"
        result = capture / "complex_server_result_with_beaker/complex_d948ad1b-7055-4f95-838b-4a43e092e102_all"
        if not result.is_dir():
            self.skipTest("New server result is not available in this workspace.")
        config = replace(self.config, capture_dir=capture, result_dir=result)
        scene = load_scene(config)
        workflow = ThreeStageWorkflow(config)
        stopper = workflow.plan_stage("stopper", scene, workflow.captured_tcp_pose())
        self.assertEqual(stopper.name, "stopper")
        bottle = workflow.plan_stage("bottle", scene, stopper.last_pose)
        self.assertTrue(any("Pour side offset" in warning for warning in bottle.warnings))

    def test_low_pretilt_model_clearance_is_advisory(self):
        low = replace(self.config, bottle=replace(self.config.bottle, pretilt_mouth_height_m=0.08))
        workflow = ThreeStageWorkflow(low)
        with self.assertLogs("affordance.three_stage_workflow", level="WARNING"):
            plan = workflow.plan_stage("bottle", self.scene, workflow.captured_tcp_pose())
        self.assertTrue(any("beaker rim" in warning for warning in plan.warnings))


if __name__ == "__main__":
    unittest.main()
