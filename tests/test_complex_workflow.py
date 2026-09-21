import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from affordance.complex_config import ComplexDemoConfig
from affordance.complex_server import ComplexLocalizationClient, ComplexScene
from affordance.complex_workflow import ComplexDemoWorkflow
from affordance.geometry import pose_to_matrix, rotvec_to_matrix


ROOT = Path(__file__).resolve().parents[1]


class ComplexWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ComplexDemoConfig.load(ROOT / "config" / "complex_demo.yaml")
        cls.payload = json.loads((ROOT / "config" / "complex_scene_example.json").read_text())
        cls.scene = ComplexScene.from_response(cls.payload, cls.config.geometry)

    def setUp(self):
        self.workflow = ComplexDemoWorkflow(self.config)

    def test_builds_four_independent_stages(self):
        plans = self.workflow.build_plan(self.scene, [0.4, 0, 0.3, 3.14, 0, 0])
        self.assertEqual([plan.name for plan in plans], ["open_bottle", "pour", "dropper", "stir"])
        self.assertTrue(all(plan.actions for plan in plans))
        self.assertEqual(sum(action.name == "grasp_stopper" for action in plans[0].actions), 1)
        self.assertEqual(sum(action.name.startswith("squeeze_drop_") for action in plans[2].actions), 3)

    def test_pour_keeps_detected_bottle_mouth_at_target(self):
        plan = self.workflow.build_plan(
            self.scene, [0.4, 0, 0.3, 3.14, 0, 0], stages=("pour",),
        )[0]
        cfg = self.config.bottle
        grasp = self.scene.bottle_body_grasp + np.asarray(cfg.grasp_offset_m)
        upright = rotvec_to_matrix(cfg.orientation_rotvec)
        mouth_from_tcp = upright.T @ (self.scene.bottle_mouth - grasp)
        expected = (
            self.scene.beaker_mouth
            + self.scene.beaker_normal_out * cfg.mouth_clearance_m
            + np.asarray(cfg.pour_target_offset_m)
        )
        tilt_actions = [action for action in plan.actions if action.name.startswith("pour_tilt_")]
        self.assertEqual(len(tilt_actions), cfg.tilt_steps + 1)
        for action in tilt_actions:
            transform = pose_to_matrix(action.pose)
            actual_mouth = transform[:3, 3] + transform[:3, :3] @ mouth_from_tcp
            np.testing.assert_allclose(actual_mouth, expected, atol=1e-6)

    def test_stir_path_respects_configured_radius_and_depth(self):
        plan = self.workflow.build_plan(
            self.scene, [0.4, 0, 0.3, 3.14, 0, 0], stages=("stir",),
        )[0]
        cfg = self.config.stir
        rotation = rotvec_to_matrix(cfg.orientation_rotvec)
        grasp = self.scene.glass_rod_grasp + np.asarray(cfg.grasp_offset_m)
        lower_from_tcp = rotation.T @ (self.scene.glass_rod_lower_tip - grasp)
        center = (
            self.scene.stir_reference_mouth
            - self.scene.beaker_normal_out * cfg.target_depth_below_mouth_m
        )
        stir_actions = [action for action in plan.actions if action.kind == "move_linear" and action.name.startswith("stir_")]
        self.assertEqual(len(stir_actions), cfg.turns * cfg.points_per_turn + 1)
        for action in stir_actions:
            transform = pose_to_matrix(action.pose)
            lower_tip = transform[:3, 3] + transform[:3, :3] @ lower_from_tcp
            delta = lower_tip - center
            self.assertAlmostEqual(float(delta @ self.scene.beaker_normal_out), 0.0, places=7)
            self.assertAlmostEqual(float(np.linalg.norm(delta)), cfg.radius_m, places=7)

    def test_execute_gate_blocks_before_hardware(self):
        robot = Mock()
        workflow = ComplexDemoWorkflow(self.config, robot=robot)
        with self.assertRaisesRegex(RuntimeError, "physical geometry"):
            workflow.run(True, scene_json=ROOT / "config" / "complex_scene_example.json")
        robot.connect.assert_not_called()

    def test_scene_uses_regions_and_local_model_priors(self):
        stopper_points = np.asarray(
            self.payload["objects"]["wide_mouth_bottle"]["affordance_regions"]["stopper"]["points_world"]
        )
        np.testing.assert_allclose(self.scene.stopper_grasp, np.median(stopper_points, axis=0))
        np.testing.assert_allclose(self.scene.beaker_mouth, [0.62, -0.10, 0.12])
        np.testing.assert_allclose(self.scene.dropper_tip, [0.55, 0.20, 0.08])

    def test_scene_rejects_invalid_reconstruction_and_missing_rod(self):
        broken = json.loads(json.dumps(self.payload))
        broken["objects"]["beaker"]["scale"] = [1, 0, 1]
        with self.assertRaises(ValueError):
            ComplexScene.from_response(broken, self.config.geometry)
        no_rod = json.loads(json.dumps(self.payload))
        del no_rod["objects"]["glass_rod"]
        with self.assertRaises(ValueError):
            ComplexScene.from_response(no_rod, self.config.geometry, require_glass_rod=True)
        self.assertIsNone(
            ComplexScene.from_response(no_rod, self.config.geometry, require_glass_rod=False).glass_rod_grasp
        )

    def test_bundle_must_contain_declared_model_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = Path(root) / "result.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                for name in self.scene.model_artifacts:
                    archive.writestr(name, b"ply")
            ComplexLocalizationClient._validate_bundle_artifacts(bundle, self.scene.model_artifacts)
            with zipfile.ZipFile(bundle, "w") as archive:
                archive.writestr(self.scene.model_artifacts[0], b"ply")
            with self.assertRaisesRegex(ValueError, "missing declared"):
                ComplexLocalizationClient._validate_bundle_artifacts(bundle, self.scene.model_artifacts)


if __name__ == "__main__":
    unittest.main()
