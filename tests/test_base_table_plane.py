import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from affordance.base_table_plane import (
    TABLE_REFERENCE_PARAMETER, server_accepts_table_reference,
    table_initialization_reference, validate_table_initialization_reference,
)
from affordance.three_stage_observation import ThreeStageObservation
from affordance.three_stage_result import load_scene
from affordance.three_stage_workflow import ThreeStageWorkflow
from complex_capture_client import build_parser, submit_and_wait
from fixture_scenes import build_scene
from trial_base_table_plane import run_trial


def schema_with_reference():
    return {'paths': {'/v1/complex-demo/jobs': {'post': {'requestBody': {'content': {
        'multipart/form-data': {'schema': {'$ref': '#/components/schemas/Upload'}}}}}}},
        'components': {'schemas': {'Upload': {'properties': {TABLE_REFERENCE_PARAMETER: {'type': 'string'}}}}}}


class Reply:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def raise_for_status(self): pass
    def json(self): return self.value


class BaseTableInitializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config, self.payload = build_scene(self.directory.name, dual=False)

    def test_reference_permits_only_z_translation_at_initialization(self):
        reference = table_initialization_reference()
        self.assertEqual(reference['normal'], [0, 0, 1])
        self.assertEqual(reference['offset_m'], 0)
        self.assertEqual(reference['apply_stage'], 'pose_initialization')
        self.assertEqual(reference['allowed_adjustment'], 'z_translation_only')
        self.assertTrue(reference['preserve_object_xy'])
        self.assertTrue(reference['preserve_object_rotation'])
        self.assertTrue(reference['preserve_object_scale'])
        self.assertFalse(reference['postsolve_scene_transform'])
        for key, value in [('preserve_object_xy', False), ('postsolve_scene_transform', True),
                           ('allowed_adjustment', 'rigid_transform')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_table_initialization_reference({**reference, key: value})

    def test_server_support_requires_a_declared_parameter(self):
        self.assertFalse(server_accepts_table_reference({}))
        self.assertTrue(server_accepts_table_reference(schema_with_reference()))

    def test_enabling_reference_never_changes_cached_models_or_grasp_coordinates(self):
        original = load_scene(self.config)
        snapshots = {p: p.read_bytes() for p in self.config.capture_dir.rglob('*') if p.is_file()}
        enabled = replace(self.config, table_constraint_enabled=True)
        scene = load_scene(enabled)
        for field in ('stopper_grasp', 'bottle_body_grasp', 'beaker_mouth_center', 'dropper_bulb_grasp'):
            np.testing.assert_array_equal(getattr(scene, field), getattr(original, field))
        self.assertEqual(scene.model_artifacts, original.model_artifacts)
        self.assertFalse(scene.quality_report['table_initialization_reference']['postsolve_scene_transform_applied'])
        self.assertFalse((self.config.result_dir/'local_table_constraint').exists())
        robot, gripper = Mock(), Mock()
        robot.actual_tcp_pose.return_value = [-0.35, 0.15, 0.3, 3.14159, 0, 0]
        workflow = ThreeStageWorkflow(enabled, robot=robot, gripper=gripper)
        with patch.object(workflow, '_run_stage') as execute:
            workflow.execute(scene, ('stopper',), lambda *_: None)
        plan = execute.call_args.args[0]
        target = next(step.pose[:3] for step in plan.steps if step.name == 'stopper_grasp')
        np.testing.assert_array_equal(target, original.stopper_grasp)
        for path, contents in snapshots.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_unsupported_server_stops_before_any_robot_motion(self):
        robot, camera = Mock(), Mock()
        observer = ThreeStageObservation(replace(self.config, table_constraint_enabled=True),
            robot=robot, camera=camera, health_get=lambda *_, **__: Reply({}))
        with patch.dict('os.environ', {'LAG_API_TOKEN': 'test-token'}):
            with self.assertRaisesRegex(RuntimeError, 'table_plane_reference'):
                observer.run(lambda *_: None)
        robot.connect.assert_not_called()
        robot.move_linear.assert_not_called()
        camera.connect.assert_not_called()

    def test_supported_server_gets_initialization_reference_without_fake_robot_pose(self):
        enabled = replace(self.config, table_constraint_enabled=True)
        calls = []
        observer = ThreeStageObservation(enabled, robot=Mock(), camera=Mock(),
            submitter=lambda args: calls.append(args) or 0,
            health_get=lambda *_, **__: Reply(schema_with_reference()))
        observer._preflight_server('test-token')
        observer._submit(self.config.capture_dir, Path(self.directory.name)/'upload', 'test-token')
        self.assertEqual(calls[0].table_plane_reference, table_initialization_reference())
        args = build_parser().parse_args(['--token', 'test-token', 'run', '--capture-dir',
                                         str(self.config.capture_dir), '--objects', 'wide_mouth_bottle,beaker,dropper'])
        args.table_plane_reference = calls[0].table_plane_reference
        measured_pose = json.loads((self.config.capture_dir/'T_world_ee.json').read_text())

        def inspect_upload(url, **kwargs):
            self.assertEqual(json.loads(kwargs['data'][TABLE_REFERENCE_PARAMETER]), table_initialization_reference())
            self.assertEqual(json.load(kwargs['files']['world_ee'][1]), measured_pose)
            raise RuntimeError('upload inspected without a real request')

        with patch('complex_capture_client.requests.post', side_effect=inspect_upload):
            with self.assertRaisesRegex(RuntimeError, 'upload inspected'):
                submit_and_wait(args)

    def test_reference_export_does_not_transform_any_source_data(self):
        snapshots = {p: p.read_bytes() for p in self.config.capture_dir.rglob('*') if p.is_file()}
        output = self.config.capture_dir/'initialization_reference'
        report = run_trial(self.config.capture_dir, self.config.result_dir, output)
        self.assertFalse(report['postsolve_scene_transform_applied'])
        self.assertFalse((output/'models').exists())
        for path, contents in snapshots.items(): self.assertEqual(path.read_bytes(), contents)


if __name__ == '__main__':
    unittest.main()
