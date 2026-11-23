import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from affordance.geometry import rotvec_to_matrix
from affordance.target_geometry import (
    GraspCenterConfig, OpeningConfig, opening_from_model, plane_basis,
    sample_opening_disk, weighted_modal_center,
)
from affordance.three_stage_config import ThreeStageConfig
from affordance.three_stage_result import _read_json, _region_scores, load_scene
from affordance.three_stage_workflow import ThreeStageWorkflow


ROOT = Path(__file__).resolve().parents[1]


class WeightedCenterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        # The cap forms a narrow height peak despite a larger, spread-out tail.
        cap = rng.normal([0.02, -0.01, 0.087], [.003, .003, .0007], (900, 3))
        tail = rng.uniform([.01, -.02, .020], [.03, 0, .075], (2200, 3))
        self.points = np.vstack((cap, tail))

    def test_dense_cap_layer_wins_over_global_tail(self):
        center, report = weighted_modal_center(self.points)
        self.assertLess(np.mean(self.points[:, 2]), .065)
        np.testing.assert_allclose(center, [.02, -.01, .087], atol=.001)
        self.assertEqual(report['method'], 'axial_density_mode_local_weighted_mean')

    def test_zero_score_points_never_vote(self):
        irrelevant = np.tile([1, 1, .2], (10000, 1))
        points = np.vstack((self.points, irrelevant))
        scores = np.r_[np.ones(len(self.points)), np.zeros(len(irrelevant))]
        center, report = weighted_modal_center(points, scores)
        reference, _ = weighted_modal_center(self.points)
        np.testing.assert_allclose(center, reference, atol=1e-12)
        self.assertEqual(report['positive_score_points'], len(self.points))

    def test_unreachable_base_peak_is_removed_before_selection(self):
        base = np.tile([.02, -.01, -.001], (5000, 1))
        center, report = weighted_modal_center(
            np.vstack((self.points, base)), bounds=([-1, -1, .015], [1, 1, .3]),
        )
        self.assertGreater(center[2], .08)
        self.assertEqual(report['excluded_by_bounds'], 5000)

    def test_rotated_axis_and_translation_preserve_geometric_result(self):
        rotation = rotvec_to_matrix([.4, -.6, .2])
        translation = np.array([.5, -.2, .3])
        original, _ = weighted_modal_center(self.points)
        transformed, _ = weighted_modal_center(self.points @ rotation.T + translation, axis=rotation[:, 2])
        np.testing.assert_allclose(transformed, rotation @ original + translation, atol=1e-10)

    def test_no_scores_is_explicit_uniform_weighting(self):
        first, report = weighted_modal_center(self.points)
        second, _ = weighted_modal_center(self.points, np.ones(len(self.points)))
        np.testing.assert_allclose(first, second)
        self.assertEqual(report['score_source'], 'uniform_no_scores')

    def test_rejects_invalid_or_empty_weight_support(self):
        for scores in (np.zeros(len(self.points)), np.ones(2), np.full(len(self.points), -1), np.full(len(self.points), np.nan)):
            with self.subTest(scores_shape=scores.shape), self.assertRaises(ValueError):
                weighted_modal_center(self.points, scores)
        with self.assertRaises(ValueError):
            weighted_modal_center(self.points, bounds=([1, 1, 1], [2, 2, 2]))
        with self.assertRaises(ValueError):
            weighted_modal_center(self.points, config=GraspCenterConfig(bandwidth_m=-1))

    def test_npz_scores_cannot_be_used_with_reordered_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root/'region.npz', points_world=self.points[::-1], scores=np.ones(len(self.points)))
            with self.assertRaisesRegex(ValueError, 'XYZ/order'):
                _region_scores(root, {'points_world_npz_artifact': 'region.npz'}, self.points)


class OpeningTests(unittest.TestCase):
    def model(self, rotation=np.eye(3)):
        rng = np.random.default_rng(12)
        angles = np.arange(360) * (2*np.pi/360)
        ring = np.c_[.03*np.cos(angles), .03*np.sin(angles), np.full(len(angles), .1)]
        wall = np.vstack([ring-[0, 0, depth] for depth in np.linspace(.001, .09, 30)])
        # Dense reconstructed interior/top fill must not become the rim radius.
        radial = .03*np.sqrt(rng.uniform(size=4000))
        phi = rng.uniform(0, 2*np.pi, len(radial))
        fill = np.c_[radial*np.cos(phi), radial*np.sin(phi), np.full(len(radial), .1)]
        shift = np.array([-.45, .3, .02])
        return np.vstack((ring, wall, fill)) @ rotation.T + shift, rotation@np.array([0, 0, .1])+shift

    def test_hollow_center_is_recovered_despite_wall_and_filled_mesh(self):
        model, expected = self.model()
        opening = opening_from_model(model)
        np.testing.assert_allclose(opening['center_world_m'], expected, atol=.0005)
        self.assertAlmostEqual(opening['outer_radius_m'], .03, delta=.0005)
        self.assertAlmostEqual(opening['inner_radius_estimate_m'], .028, delta=.0005)
        self.assertTrue(opening['synthetic_target_geometry'])

    def test_tilted_virtual_disk_lies_in_plane_and_inside_inset(self):
        rotation = rotvec_to_matrix([.3, -.2, .1])
        model, expected = self.model(rotation)
        opening = opening_from_model(model, rotation[:, 2])
        center = np.asarray(opening['center_world_m'])
        np.testing.assert_allclose(center, expected, atol=.0005)
        disk = sample_opening_disk(opening)
        delta = disk-center
        np.testing.assert_allclose(delta @ np.asarray(opening['normal_world']), 0, atol=1e-12)
        self.assertLessEqual(np.max(np.linalg.norm(delta, axis=1)), opening['receiving_radius_m']+1e-12)

    def test_bad_geometry_does_not_fall_back_to_wall_center(self):
        with self.assertRaises(ValueError):
            opening_from_model(np.c_[np.zeros(50), np.zeros(50), np.linspace(0, .1, 50)])
        model, _ = self.model()
        with self.assertRaises(ValueError):
            opening_from_model(model, config=OpeningConfig(wall_margin_m=.04))


class SavedSceneTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capture = ROOT/'captures/20261119_114451_850611'
        result = capture/'complex_server_result_with_beaker/complex_6e965876-cddf-49df-92c2-c7e482462dce_all'
        if not (result/'result.json').is_file():
            raise unittest.SkipTest('Saved November regression capture is unavailable.')
        cls.config = replace(ThreeStageConfig.load(ROOT/'config/three_stage_demo.yaml'), capture_dir=capture, result_dir=result)
        cls.scene = load_scene(cls.config)

    def test_all_three_scene_grasps_use_scores_and_modal_center(self):
        reports = self.scene.quality_report['grasp_centers']
        self.assertEqual(set(reports), {'stopper', 'body_grasp', 'bulb'})
        for report in reports.values():
            self.assertEqual(report['method'], 'axial_density_mode_local_weighted_mean')
            self.assertEqual(report['score_source'], 'affordance_scores')
        self.assertGreater(self.scene.stopper_grasp[2], .083)
        self.assertLess(self.scene.stopper_grasp[2], .091)
        self.assertGreater(self.scene.bottle_body_grasp[2], self.config.base.motion.workspace_min[2])
        self.assertGreater(reports['body_grasp']['excluded_by_bounds'], 0)

    def test_receiving_target_does_not_require_wall_projected_affordance(self):
        result_path = self.config.result_dir/'result.json'
        response = _read_json(result_path)
        response['objects']['beaker']['affordance_regions'] = {}
        with patch('affordance.three_stage_result._read_json', side_effect=lambda path: response if path == result_path else _read_json(path)):
            scene = load_scene(self.config)
        np.testing.assert_allclose(scene.beaker_mouth_center, self.scene.beaker_mouth_center)
        self.assertIsNone(scene.quality_report['beaker_circle_fit']['raw_region_median_base_m'])

    def test_default_pour_aims_at_opening_center_and_large_offset_is_bounded(self):
        for requested in (0, 1):
            config = replace(self.config, bottle=replace(self.config.bottle, pour_side_offset_m=requested))
            workflow = ThreeStageWorkflow(config, robot=Mock(), gripper=Mock())
            plan = workflow._plan_bottle(self.scene, workflow.captured_tcp_pose())
            grasp = self.scene.bottle_body_grasp+self.scene.bottle_axis_out*config.bottle.grasp_offset_along_axis_m
            upright = next(s for s in plan.steps if s.name == 'bottle_high_upright')
            rotation = rotvec_to_matrix(upright.pose[3:])
            mouth_tool = rotation.T@(self.scene.bottle_mouth_approx-grasp)
            low = next(s for s in plan.steps if s.name == f'lower_tilted_bottle_{config.bottle.lower_steps:02d}')
            mouth = np.array(low.pose[:3])+rotvec_to_matrix(low.pose[3:])@mouth_tool
            receiving = mouth-np.array([0, 0, config.bottle.pour_mouth_clearance_m])
            opening = self.scene.quality_report['beaker_opening']
            delta = receiving-self.scene.beaker_mouth_center
            self.assertAlmostEqual(float(delta@opening['normal_world']), 0, places=10)
            self.assertLessEqual(np.linalg.norm(delta), opening['receiving_radius_m']+1e-10)
            if requested == 0:
                np.testing.assert_allclose(receiving, self.scene.beaker_mouth_center, atol=1e-10)
            else:
                self.assertTrue(any('limited' in warning for warning in plan.warnings))

    def test_offline_artifacts_are_explicitly_synthetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(self.config, result_dir=Path(temporary))
            workflow = ThreeStageWorkflow(config, robot=Mock(), gripper=Mock())
            path = workflow.save_offline_plan(self.scene, [])
            record = json.loads(path.read_text())
            self.assertTrue(record['derived_geometry']['beaker_opening']['synthetic_target_geometry'])
            ply = Path(record['target_artifacts']['synthetic_beaker_opening_ply'])
            self.assertIn(b'SYNTHETIC', ply.read_bytes()[:200])

    def test_all_live_start_plans_use_updated_targets_without_hardware(self):
        robot, gripper = Mock(), Mock()
        workflow = ThreeStageWorkflow(self.config, robot=robot, gripper=gripper)
        plans = workflow.plan_all(self.scene, workflow.captured_tcp_pose())
        stopper = next(step for step in plans[0].steps if step.name == 'stopper_grasp')
        body = next(step for step in plans[1].steps if step.name == 'bottle_body_grasp')
        bulb = next(step for step in plans[2].steps if step.name == 'dropper_bulb_grasp')
        np.testing.assert_allclose(stopper.pose[:3], self.scene.stopper_grasp)
        np.testing.assert_allclose(body.pose[:3], self.scene.bottle_body_grasp+self.scene.bottle_axis_out*self.config.bottle.grasp_offset_along_axis_m)
        np.testing.assert_allclose(bulb.pose[:3], self.scene.dropper_bulb_grasp+[0, 0, self.config.dropper.grasp_offset_z_m])
        robot.connect.assert_not_called()
        robot.move_linear.assert_not_called()
        gripper.connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
