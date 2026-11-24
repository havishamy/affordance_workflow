import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from affordance.geometry import rotvec_to_matrix
from affordance.hardware import URRobot
from affordance.three_stage_result import load_scene
from affordance.three_stage_workflow import StagePlan, Step, ThreeStageWorkflow
from fixture_scenes import build_scene


class BottleRotationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config, _ = build_scene(self.directory.name, dual=False)
        self.scene = load_scene(self.config)
        self.current = [-0.35, 0.15, 0.3, 3.14159, 0, 0]
        self.robot, self.gripper = Mock(), Mock()
        self.robot.actual_joints.return_value = [0.1, -1, 1.2, -1.4, -1.5, 0.2]
        self.robot.actual_tcp_pose.return_value = list(self.current)
        self.workflow = ThreeStageWorkflow(self.config, robot=self.robot, gripper=self.gripper)

    def test_offline_chooses_nearest_equivalent_rotation_for_every_direction(self):
        for angle in np.linspace(-math.pi, math.pi, 17):
            scene = replace(self.scene, beaker_mouth_center=self.scene.bottle_body_grasp +
                            [0.12*math.cos(angle), 0.12*math.sin(angle), 0.03])
            plan = self.workflow._plan_bottle(scene, self.current)
            self.assertAlmostEqual(plan.metadata['orientation_transition_deg'],
                                   min(plan.metadata['equivalent_branch_transition_deg']))
            self.assertLessEqual(plan.metadata['orientation_transition_deg'], 90.001)

    def test_both_yaw_branches_preserve_pouring_toward_beaker(self):
        direction = self.scene.beaker_mouth_center[:2]-self.scene.bottle_body_grasp[:2]
        direction /= np.linalg.norm(direction)
        for branch in (0, 1):
            plan = self.workflow._plan_bottle(self.scene, self.current, orientation_branch=branch)
            upright = next(s.pose for s in plan.steps if s.name == 'bottle_high_upright')
            tilted = next(s.pose for s in plan.steps if s.name == f'tilt_high_{self.config.bottle.tilt_steps:02d}')
            axis = rotvec_to_matrix(tilted[3:]) @ rotvec_to_matrix(upright[3:]).T @ self.scene.bottle_axis_out
            self.assertAlmostEqual(float(axis[:2]@direction),
                                   math.sin(math.radians(self.config.bottle.tilt_angle_deg)), places=6)

    def test_controller_invalid_nearest_branch_falls_back_before_gripping(self):
        nearest = self.workflow._plan_bottle(self.scene, self.current).metadata['orientation_branch']

        def check(plan, *_):
            if plan.metadata['orientation_branch'] == nearest:
                raise RuntimeError('test joint limit')
            return {'total_joint_travel_rad': 10.0}

        with patch.object(self.workflow, '_bottle_joint_preflight', side_effect=check):
            plan = self.workflow._select_live_bottle_plan(self.scene, self.current)
        self.assertNotEqual(plan.metadata['orientation_branch'], nearest)
        self.assertIn('test joint limit', str(plan.metadata['rejected_orientation_branches']))
        self.robot.move_linear.assert_not_called()
        self.gripper.connect.assert_not_called()

    def test_execution_uses_joint_checked_branch(self):
        def check(plan, *_):
            return {'total_joint_travel_rad': 1.0 if plan.metadata['orientation_branch'] == 1 else 3.0}

        with patch.object(self.workflow, '_bottle_joint_preflight', side_effect=check), \
             patch.object(self.workflow, '_run_stage') as execute:
            output = self.workflow.execute(self.scene, ('bottle',), lambda *_: None)
        executed = execute.call_args.args[0]
        self.assertEqual(executed.metadata['orientation_branch'], 1)
        stored = json.loads((output/'plan_bottle.json').read_text())
        self.assertEqual(stored['metadata']['orientation_branch'], 1)
        self.assertEqual(stored['metadata']['joint_preflight']['total_joint_travel_rad'], 1.0)

    def test_no_valid_joint_route_stops_before_bottle_motion(self):
        with patch.object(self.workflow, '_bottle_joint_preflight', side_effect=RuntimeError('unreachable')):
            with self.assertRaisesRegex(RuntimeError, 'No continuous bottle route'):
                self.workflow.execute(self.scene, ('bottle',), lambda *_: None)
        self.gripper.connect.assert_not_called()
        self.robot.move_linear.assert_not_called()

    def test_preflight_seeds_each_ik_from_previous_sample(self):
        self.robot.joints_within_safety_limits.return_value = True
        self.robot.inverse_kinematics.side_effect = lambda pose, qnear: (np.array(qnear)+0.001).tolist()
        plan = StagePlan('bottle', (Step('moveL', 'rotate', pose=(0, 0, 0.2, 0, 0, 0.2)),))
        start = [0.0]*6
        report = self.workflow._bottle_joint_preflight(plan, [0, 0, 0.2, 0, 0, 0], start)
        self.assertGreater(report['samples'], 1)
        calls = self.robot.inverse_kinematics.call_args_list
        np.testing.assert_allclose(calls[0].kwargs['qnear'], start)
        np.testing.assert_allclose(calls[1].kwargs['qnear'], np.full(6, 0.001))

    def test_preflight_rejects_full_turn_joint_jump_without_wrapping(self):
        self.robot.joints_within_safety_limits.return_value = True
        self.robot.inverse_kinematics.return_value = [0, 0, 0, 0, 0, 2*math.pi]
        plan = StagePlan('bottle', (Step('moveL', 'rotate', pose=(0, 0, 0.2, 0, 0, 0.1)),))
        with self.assertRaisesRegex(RuntimeError, 'discontinuous IK joint jump'):
            self.workflow._bottle_joint_preflight(plan, [0, 0, 0.2, 0, 0, 0], [0]*6)

    def test_hardware_wrapper_forwards_qnear(self):
        robot = URRobot(self.config.base.robot)
        robot.control = Mock()
        robot.control.getInverseKinematics.return_value = [0.1]*6
        result = robot.inverse_kinematics([0.2]*6, qnear=[0.3]*6)
        robot.control.getInverseKinematics.assert_called_once_with([0.2]*6, [0.3]*6)
        self.assertEqual(result, [0.1]*6)


if __name__ == '__main__':
    unittest.main()
