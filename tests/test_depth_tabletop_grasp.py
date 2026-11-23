import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from affordance.depth_tabletop_grasp import (
    TabletopConfig, estimate_tabletop_stopper, fit_table_plane,
    save_tabletop_diagnostic, tabletop_grasp_plan,
)
from affordance.geometry import pose_to_matrix, rotvec_to_matrix
from affordance.three_stage_config import ThreeStageConfig
from run_depth_tabletop_grasp import main, saved_start_for_current_tcp


ROOT = Path(__file__).resolve().parents[1]


class TabletopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ThreeStageConfig.load(ROOT/'config/three_stage_demo.yaml')

    def scene(self, second=False, border=False, invalid=False):
        image = np.full((120, 160, 3), [90, 120, 90], np.uint8)
        depth = np.full((120, 160), 500, np.uint16)
        x = 0 if border else 70
        image[45:70, x:x+22] = [150, 175, 185]
        depth[45:70, x:x+22] = 470
        if second:
            image[45:70, 110:132] = [150, 175, 185]
            depth[45:70, 110:132] = 470
        if invalid:
            depth[52:57, x+8:x+12] = 0
        intrinsics = dict(width=160, height=120, fx=200., fy=200., cx=80., cy=60.)
        transform = np.diag([1., -1., -1., 1.])
        transform[:3,3] = [-.45, .25, .5]
        return image, depth, intrinsics, .001, transform

    def estimate(self, **options):
        return estimate_tabletop_stopper(*self.scene(), **options)

    def test_depth_separates_stopper_from_table(self):
        report, points, mask = self.estimate()
        self.assertGreater(len(points), 500)
        self.assertAlmostEqual(report['center_height_above_table_m'], .03, places=6)
        np.testing.assert_allclose(points[:,2], .03, atol=1e-8)
        self.assertGreater(report['table_plane']['inlier_fraction'], .9)
        self.assertEqual(report['center_method'], 'modal')

    def test_depth_only_mode_needs_no_color_model(self):
        image, *rest = self.scene()
        image[:] = 0
        report, _, _ = estimate_tabletop_stopper(image, *rest, color_hint='none')
        self.assertAlmostEqual(report['center_height_above_table_m'], .03, places=6)

    def test_multiple_foreground_objects_need_selection(self):
        with self.assertRaisesRegex(ValueError, 'found 2 candidates'):
            estimate_tabletop_stopper(*self.scene(second=True))
        report, _, _ = estimate_tabletop_stopper(*self.scene(second=True), target_pixel=[80,55])
        self.assertLess(report['center_pixel_uv'][0], 100)

    def test_partial_boundary_object_is_not_grasped(self):
        with self.assertRaisesRegex(ValueError, 'found 0 candidates'):
            estimate_tabletop_stopper(*self.scene(border=True))

    def test_invalid_depth_is_not_restored_by_morphology(self):
        image, depth, intrinsics, scale, transform = self.scene(invalid=True)
        _, points, mask = estimate_tabletop_stopper(image, depth, intrinsics, scale, transform)
        self.assertFalse(np.any(mask[depth == 0]))
        self.assertLess(len(points), 550)

    def test_tilted_table_keeps_physical_height(self):
        image, depth, intrinsics, scale, transform = self.scene()
        rotation = rotvec_to_matrix([.18, -.12, 0])
        moved = transform.copy();moved[:3,:3] = rotation@transform[:3,:3]
        moved[:3,3] = rotation@transform[:3,3]
        report, _, _ = estimate_tabletop_stopper(image, depth, intrinsics, scale, moved)
        np.testing.assert_allclose(report['table_plane']['normal_world'], rotation[:,2], atol=1e-8)
        self.assertAlmostEqual(report['center_height_above_table_m'], .03, places=6)

    def test_no_table_or_invalid_parameters_fail(self):
        with self.assertRaises(ValueError):
            fit_table_plane(np.ones((20,3)))
        with self.assertRaises(ValueError):
            TabletopConfig(min_height_m=.001).validate()

    def test_table_clearance_requires_measurement_and_blocks_finger_collision(self):
        report, _, _ = self.estimate()
        current=[-.35,.15,.19,1,-2,-1]
        unknown=tabletop_grasp_plan(self.config,current,report)
        self.assertEqual(unknown['table_clearance']['status'],'measurement_required')
        safe=tabletop_grasp_plan(self.config,current,report,finger_below_tcp_m=.01)
        self.assertEqual(safe['table_clearance']['status'],'clear')
        unsafe=tabletop_grasp_plan(self.config,current,report,finger_below_tcp_m=.025)
        self.assertEqual(unsafe['table_clearance']['status'],'blocked')

    def test_workspace_floor_is_never_lowered(self):
        report, _, _ = self.estimate()
        with self.assertRaisesRegex(ValueError, 'workspace'):
            tabletop_grasp_plan(self.config,[-.35,.15,.19,1,-2,-1],report,offset=[0,0,-.05],finger_below_tcp_m=0)

    def test_operator_can_explicitly_confirm_unmeasured_clearance(self):
        report, _, _ = self.estimate()
        plan=tabletop_grasp_plan(self.config,[-.35,.15,.19,1,-2,-1],report,operator_confirmed_clearance=True)
        self.assertEqual(plan['table_clearance']['status'],'operator_confirmed_unmeasured')
        self.assertFalse(plan['table_clearance']['finger_extension_measured'])
        self.assertNotIn('finger_below_tcp_m',plan['table_clearance'])

    def test_operator_confirmation_does_not_override_measured_collision(self):
        report, _, _ = self.estimate()
        plan=tabletop_grasp_plan(self.config,[-.35,.15,.19,1,-2,-1],report,finger_below_tcp_m=.025,operator_confirmed_clearance=True)
        self.assertEqual(plan['table_clearance']['status'],'blocked')

    def test_offset_and_lift_are_explicit(self):
        report, _, _ = self.estimate()
        current=[-.35,.15,.19,1,-2,-1]
        ordinary=tabletop_grasp_plan(self.config,current,report,finger_below_tcp_m=0)
        self.assertNotIn('lift',ordinary['poses'])
        lifted=tabletop_grasp_plan(self.config,current,report,finger_below_tcp_m=0,lift_m=.05)
        self.assertAlmostEqual(lifted['poses']['lift'][2]-lifted['poses']['grasp'][2],.05)

    def test_changed_tcp_reexpresses_saved_start_without_changing_flange(self):
        saved=dict(actual_tcp_pose=[-.35,.15,.19,1,-2,-1],tcp_offset=[0,0,.15,0,0,0])
        new=[0,0,.163,0,0,0]
        converted=saved_start_for_current_tcp(saved,new)
        first=pose_to_matrix(saved['actual_tcp_pose'])@np.linalg.inv(pose_to_matrix(saved['tcp_offset']))
        second=pose_to_matrix(converted)@np.linalg.inv(pose_to_matrix(new))
        np.testing.assert_allclose(first,second,atol=1e-12)

    def test_grasp_is_blocked_before_hardware_motion_when_fingers_too_long(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture=Path(temporary)
            robot=Mock();robot.actual_tcp_pose.return_value=[-.35,.15,.19,1,-2,-1]
            with patch('run_depth_tabletop_grasp.URRobot',return_value=robot), patch('run_depth_tabletop_grasp.capture_current_view',return_value=capture), patch('run_depth_tabletop_grasp.load_saved_capture',return_value=self.scene()), patch('run_depth_tabletop_grasp.DHGripper',side_effect=AssertionError('must not connect gripper')):
                status=main(['--live','--execute','--finger-below-tcp-mm','25','--yes'])
            self.assertEqual(status,1)
            robot.move_linear.assert_not_called()
            self.assertTrue((capture/'tabletop_stopper_diagnostic/failure.json').is_file())

    def test_current_real_capture_locates_central_stopper(self):
        from affordance.depth_bulb_grasp import load_saved_capture
        capture=ROOT/'captures/20261120_121947_874934'
        if not (capture/'rgb.png').is_file():
            self.skipTest('Current tabletop capture is unavailable.')
        report,points,mask=estimate_tabletop_stopper(*load_saved_capture(capture))
        self.assertGreater(len(points),1500)
        self.assertTrue(.025 < report['center_height_above_table_m'] < .035)
        self.assertTrue(320 < report['center_pixel_uv'][0] < 370)
        self.assertTrue(260 < report['center_pixel_uv'][1] < 310)


if __name__=='__main__':
    unittest.main()
