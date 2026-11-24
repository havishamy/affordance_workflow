import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from affordance.depth_bulb_grasp import (
    estimate_bulb_from_raw_depth, grasp_from_current_pose,
    latest_complete_depth_capture, load_saved_capture, validate_depth_calibration,
)
from affordance.three_stage_config import ThreeStageConfig
from run_depth_bulb_grasp import auto_confirm, main, parse_args


ROOT = Path(__file__).resolve().parents[1]


class BulbCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT/'config/three_stage_demo.yaml')

    def sample(self, hue=2, second=False):
        hsv = np.zeros((100, 180, 3), dtype=np.uint8)
        hsv[25:65, 100:125] = (hue, 150, 190)
        if second:
            hsv[25:65, 25:50] = (hue, 150, 190)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        depth = np.full((100, 180), 350, dtype=np.uint16)
        camera = dict(width=180, height=100, fx=200., fy=200., cx=90., cy=50., depth_scale_m_per_unit=.001)
        return image, depth, camera

    def save_capture(self, root):
        root.mkdir(parents=True, exist_ok=True)
        image, depth, camera = self.sample()
        cv2.imwrite(str(root/'rgb.png'), image)
        cv2.imwrite(str(root/'depth.png'), depth)
        (root/'camera.json').write_text(json.dumps(camera))
        (root/'T_world_ee.json').write_text(json.dumps({'T_world_ee': np.eye(4).tolist()}))
        (root/'capture_info.json').write_text(json.dumps({'capture_id': root.name, 'world_frame': 'base', 'ee_frame': 'flange'}))

    def test_defaults_use_auto_roi_and_zero_calibration_offset(self):
        args = parse_args([])
        self.assertIsNone(args.roi)
        self.assertEqual(args.grasp_z_offset, 0)
        self.assertFalse(args.compare_server)

    def test_full_image_finds_bulb_outside_old_roi(self):
        image, depth, camera = self.sample()
        estimate = estimate_bulb_from_raw_depth(image, depth, camera, .001, np.eye(4), None, .2, .5)
        self.assertGreater(estimate.depth_inlier_pixels, 800)
        self.assertEqual(estimate.roi_px, (0, 0, 180, 100))
        self.assertGreater(estimate.median_pixel_uv[0], 100)

    def test_red_hue_wraparound_is_supported(self):
        image, depth, camera = self.sample(hue=175)
        estimate = estimate_bulb_from_raw_depth(image, depth, camera, .001, np.eye(4), None, .2, .5)
        self.assertGreater(estimate.depth_inlier_pixels, 800)

    def test_two_comparable_bulbs_require_explicit_roi(self):
        image, depth, camera = self.sample(second=True)
        with self.assertRaisesRegex(ValueError, 'Multiple comparable'):
            estimate_bulb_from_raw_depth(image, depth, camera, .001, np.eye(4), None, .2, .5)
        estimate = estimate_bulb_from_raw_depth(image, depth, camera, .001, np.eye(4), [90, 10, 140, 85], .2, .5)
        self.assertGreater(estimate.median_pixel_uv[0], 100)

    def test_newest_capture_does_not_need_successful_server_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.save_capture(root/'20260101_010101_000001')
            self.save_capture(root/'20260101_010102_000001')
            (root/'20260101_010103_000001').mkdir()  # incomplete attempt
            self.assertEqual(latest_complete_depth_capture(root).name, '20260101_010102_000001')

    def test_offline_cli_never_connects_hardware_or_loads_server_center(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)/'capture'
            self.save_capture(capture)
            # Deliberately unusable server result must have no effect by default.
            fake = capture/'complex_server_result_with_beaker/complex_bad'
            fake.mkdir(parents=True)
            (fake/'result.json').write_text('not valid JSON')
            with patch('run_depth_bulb_grasp.URRobot', side_effect=AssertionError('hardware')), patch('run_depth_bulb_grasp._region_scores', side_effect=AssertionError('server')):
                self.assertEqual(main(['--capture-dir', str(capture)]), 0)
            result = json.loads((capture/'raw_depth_bulb_diagnostic/result.json').read_text())
            np.testing.assert_allclose(result['grasp_target_base_m'], result['surface_center_base_m'])
            self.assertIn('hand_eye_sha256', result)
            self.assertIn('T_base_opencv_camera', result)
            self.assertTrue(Path(result['observed_points_ply']).is_file())

    def test_rejects_tcp_frame_capture_instead_of_double_applying_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            self.save_capture(capture)
            (capture/'capture_info.json').write_text(json.dumps({'world_frame': 'base', 'ee_frame': 'tcp'}))
            with self.assertRaisesRegex(ValueError, 'ee_frame'):
                load_saved_capture(capture)

    def test_raw_depth_cli_preserves_xy_when_initialization_reference_is_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)/'capture'
            self.save_capture(capture)
            color, depth, camera, scale, source_transform = load_saved_capture(capture)
            config = replace(self.config, table_constraint_enabled=True)
            with patch('run_depth_bulb_grasp.ThreeStageConfig.load', return_value=config), \
                 patch('run_depth_bulb_grasp.URRobot', side_effect=AssertionError('hardware')), \
                 patch('run_depth_bulb_grasp._region_scores', side_effect=AssertionError('server')):
                self.assertEqual(main(['--capture-dir', str(capture), '--grasp-z-offset', '0.010']), 0)
            result = json.loads((capture/'raw_depth_bulb_diagnostic/result.json').read_text())
            expected = estimate_bulb_from_raw_depth(color, depth, camera, scale, source_transform, None, .2, .5)
            np.testing.assert_array_equal(result['source_T_base_opencv_camera'], source_transform)
            np.testing.assert_array_equal(result['T_base_opencv_camera'], source_transform)
            np.testing.assert_allclose(result['grasp_target_base_m'][:2], expected.surface_center_base_m[:2], atol=1e-12)
            np.testing.assert_allclose(result['grasp_target_base_m'], expected.surface_center_base_m+[0,0,.010])
            self.assertFalse(result['postsolve_table_transform_applied'])
            self.assertNotIn('table_constraint', result)

    def test_inconsistent_tcp_and_camera_files_fail_before_motion(self):
        self.assertTrue(validate_depth_calibration(self.config).is_file())
        changed = replace(self.config, base=replace(self.config.base, robot=replace(self.config.base.robot, tcp_offset=(0, 0, .14, 0, 0, 0))))
        with self.assertRaisesRegex(ValueError, 'disagree'):
            validate_depth_calibration(changed)

    def test_direct_module_file_accepts_help_from_any_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            response = subprocess.run([sys.executable, str(ROOT/'affordance/depth_bulb_grasp.py'), '--help'], cwd=temporary, capture_output=True, text=True)
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertIn('--live', response.stdout)
        self.assertIn('--yes', response.stdout)

    def test_auto_confirmation_does_not_read_stdin(self):
        with patch('builtins.input', side_effect=AssertionError('stdin')):
            auto_confirm('test checkpoint')

    def test_saved_plan_and_completion_record_no_extraction(self):
        robot, gripper, confirm = Mock(), Mock(), Mock()
        target = np.array([-.4, .28, .174])
        robot.actual_tcp_pose.side_effect = [[-.35, .15, .19, 1, -2, -1], [*target, 3.14159, 0, 0]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grasp_from_current_pose(self.config, robot, gripper, target, confirm, record_dir=root)
            plan = json.loads((root/'grasp_plan.json').read_text())
            completed = json.loads((root/'grasp_completed.json').read_text())
        self.assertFalse(plan['extract_after_close'])
        self.assertEqual(len(plan['waypoints']), 4)
        self.assertEqual(robot.move_linear.call_count, 4)
        self.assertAlmostEqual(completed['tcp_position_error_m'], 0)
        gripper.set_position.assert_called_once_with(self.config.dropper.hold_position)

    def test_saved_current_arrangement_has_a_valid_bulb_target(self):
        capture = ROOT/'captures/20261119_180947_021542'
        if not (capture/'rgb.png').exists():
            self.skipTest('Current arrangement capture is unavailable.')
        image, depth, camera, scale, transform = load_saved_capture(capture)
        estimate = estimate_bulb_from_raw_depth(image, depth, camera, scale, transform, None, .2, .5)
        self.assertGreater(estimate.depth_inlier_pixels, 1000)
        u, v = estimate.median_pixel_uv
        self.assertTrue(450 < u < 520 and 185 < v < 280)
        self.assertTrue(np.all(estimate.surface_center_base_m >= self.config.base.motion.workspace_min))
        self.assertTrue(np.all(estimate.surface_center_base_m <= self.config.base.motion.workspace_max))


if __name__ == '__main__':
    unittest.main()
