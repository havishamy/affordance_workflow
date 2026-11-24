import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from fixture_scenes import build_scene
from affordance.dropper_instances import DUAL_OBJECTS, SIZE_METHOD, metric_model_size, rank_dropper_models, requested_regions
from affordance.geometry import rotvec_to_matrix
from affordance.three_stage_result import _read_ply_xyz, load_scene
from affordance.three_stage_workflow import ThreeStageWorkflow
from affordance.three_stage_observation import ThreeStageObservation
from complex_capture_client import DEFAULT_OBJECTS, build_parser, submit_and_wait, validate_complex_result
from affordance.alignment_export import export_alignment_ply
from run_three_stage_demo import main, parser


class SizeRankingTests(unittest.TestCase):
    def model(self,size,center=(0,0,0)):
        rng=np.random.default_rng(12)
        return rng.uniform(-.5,.5,(4000,3))*np.array(size)+center

    def test_size_uses_metric_model_not_id_or_pixel_scale(self):
        clouds={'dropper_a':self.model([.06,.05,.18],[-.4,.3,.1]),'dropper_b':self.model([.03,.025,.1],[-.6,.2,.1])}
        ranking=rank_dropper_models(clouds)
        self.assertEqual(ranking['small_id'],'dropper_b')
        self.assertEqual(ranking['large_id'],'dropper_a')
        swapped=rank_dropper_models(dict(dropper_a=clouds['dropper_b'],dropper_b=clouds['dropper_a']))
        self.assertEqual(swapped['small_id'],'dropper_a')

    def test_volume_is_invariant_to_pose_and_has_cubic_scale(self):
        points=self.model([.06,.04,.17]);rotation=rotvec_to_matrix([.2,-.7,.4])
        first=metric_model_size(points)['volume_m3']
        second=metric_model_size(points@rotation.T+[.3,-.2,.5])['volume_m3']
        self.assertAlmostEqual(first,second,places=12)
        self.assertAlmostEqual(metric_model_size(points*2)['volume_m3']/first,8,places=10)

    def test_ties_missing_and_duplicate_models_do_not_guess(self):
        first=self.model([.04,.04,.12])
        for clouds in ({'dropper_a':first}, {'dropper_a':first,'dropper_b':first.copy()}, {'dropper_a':first,'dropper_b':first+[.2,0,0]}):
            with self.subTest(keys=list(clouds)),self.assertRaises(ValueError):
                rank_dropper_models(clouds)


class DualPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.config,self.payload=build_scene(self.directory.name)
        self.scene=load_scene(self.config)

    def test_upload_contract_has_four_independent_instances(self):
        self.assertEqual(DEFAULT_OBJECTS,DUAL_OBJECTS)
        args=build_parser().parse_args(['run','--capture-dir','example'])
        self.assertEqual(tuple(args.objects.split(',')),DUAL_OBJECTS)
        self.assertEqual(requested_regions(DUAL_OBJECTS)['dropper_b'],['bulb'])
        self.assertEqual(set(validate_complex_result(self.payload,DUAL_OBJECTS)),set(self.scene.model_artifacts.values()))

    def test_multipart_upload_result_and_complete_bundle_with_mock_http(self):
        class Response:
            ok=True
            status_code=200
            def __init__(self,payload=None,data=b''):self.payload=payload;self.data=data
            def json(self):return self.payload
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def raise_for_status(self):pass
            def iter_content(self,size):yield self.data
        bundle=io.BytesIO()
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in self.config.result_dir.rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(self.config.result_dir))
        output=Path(self.directory.name)/'download'
        args=build_parser().parse_args(['--token','test-token','run','--capture-dir',str(self.config.capture_dir),'--output-dir',str(output),'--scope','all'])
        def post(url,**kwargs):
            self.assertEqual(set(kwargs['files']),{'rgb','depth','camera','world_ee'})
            self.assertEqual(json.loads(kwargs['data']['requested_objects']),list(DUAL_OBJECTS))
            self.assertEqual(json.loads(kwargs['data']['requested_affordance_regions']),requested_regions(DUAL_OBJECTS))
            self.assertEqual(float(kwargs['data']['minimum_dropper_size_ratio']),1.10)
            self.assertTrue(kwargs['files']['rgb'][1].read())
            return Response({'job_id':'synthetic-job','state':'queued'})
        def get(url,**kwargs):
            if url.endswith('/bundle'):return Response(data=bundle.getvalue())
            if url.endswith('/result'):return Response(self.payload)
            return Response({'job_id':'synthetic-job','state':'completed'})
        with patch('complex_capture_client.requests.post',side_effect=post),patch('complex_capture_client.requests.get',side_effect=get),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(submit_and_wait(args),0)
        self.assertTrue((output/'complex_synthetic-job_all.zip').is_file())
        self.assertEqual(set(json.loads((output/'result.json').read_text())['objects']),set(DUAL_OBJECTS))

    def test_alignment_export_includes_both_models_and_size_mapping(self):
        calibration=Path(self.directory.name)/'calibration.json'
        calibration.write_text(json.dumps({'T_ee_cam':np.eye(4).tolist()}))
        output=self.config.result_dir/'diagnostics/alignment.ply'
        report=export_alignment_ply(self.config.capture_dir,self.config.result_dir,output,calibration)
        self.assertEqual(set(report.model_points),set(DUAL_OBJECTS))
        manifest=json.loads(report.manifest_path.read_text())
        self.assertEqual(manifest['dropper_size_ranking']['small_id'],'dropper_b')
        self.assertNotEqual(manifest['models']['dropper_a']['source_id'],manifest['models']['dropper_b']['source_id'])

    def test_independent_large_dropper_config_is_validated(self):
        bad=replace(self.config,dropper_large=replace(self.config.dropper,squeeze_position=self.config.dropper.hold_position))
        with self.assertRaisesRegex(ValueError,'large dropper'):
            bad.validate()

    def test_observer_sends_both_ids_instead_of_single_dropper(self):
        calls=[]
        observer=ThreeStageObservation(self.config,submitter=lambda args: calls.append(args) or 0)
        observer._submit(self.config.capture_dir,Path(self.directory.name)/'out','test-token')
        self.assertEqual(calls[0].objects,','.join(DUAL_OBJECTS))

    def test_role_assignment_recomputes_scaled_world_geometry(self):
        self.assertGreater(self.payload['objects']['dropper_b']['scale'][0],self.payload['objects']['dropper_a']['scale'][0])
        self.assertEqual(self.scene.dropper_instance_ids,{'small':'dropper_b','large':'dropper_a'})
        self.assertEqual(self.scene.dropper_size_ranking['method'],SIZE_METHOD)
        self.assertGreater(np.linalg.norm(self.scene.dropper_grasp('small')-self.scene.dropper_grasp('large')),.1)

    def test_all_order_reuses_motion_with_independent_returns_and_counts(self):
        config=replace(self.config,dropper_small=replace(self.config.dropper,tip_below_bulb_m=.09,squeeze_cycles=1),
                       dropper_large=replace(self.config.dropper,tip_below_bulb_m=.16,squeeze_cycles=3))
        workflow=ThreeStageWorkflow(config,robot=Mock(),gripper=Mock())
        plans=workflow.plan_all(self.scene,workflow.captured_tcp_pose())
        self.assertEqual([p.name for p in plans],['stopper','bottle','dropper_small','dropper_large'])
        for index,role in ((2,'small'),(3,'large')):
            plan=plans[index];cfg=config.dropper_motion(role)
            target=self.scene.dropper_grasp(role)+[0,0,cfg.grasp_offset_z_m]
            grab=next(s for s in plan.steps if s.name=='dropper_bulb_grasp')
            returned=next(s for s in plan.steps if s.name=='replace_dropper')
            np.testing.assert_allclose(grab.pose[:3],target)
            np.testing.assert_allclose(returned.pose[:3],target)
            self.assertEqual(sum(s.name.startswith('squeeze_') for s in plan.steps),cfg.squeeze_cycles)
            self.assertEqual(plan.metadata['source_object_id'],self.scene.dropper_instance_ids[role])
            dose=next(s for s in plan.steps if s.name=='dropper_dose_position')
            self.assertAlmostEqual(dose.pose[2],self.scene.beaker_mouth_center[2]+cfg.tip_below_bulb_m+cfg.grasp_offset_z_m+cfg.tip_clearance_above_beaker_m)
        first=next(s for s in plans[3].steps if s.kind=='moveL')
        np.testing.assert_allclose(first.pose[:2],plans[2].last_pose[:2])

    def test_missing_second_instance_and_bad_server_ranking_are_rejected(self):
        missing=json.loads(json.dumps(self.payload));del missing['objects']['dropper_b']
        with self.assertRaises(ValueError):validate_complex_result(missing,DUAL_OBJECTS)
        missing['dropper_size_ranking']={'method':SIZE_METHOD,'small_id':'dropper_a','large_id':'dropper_b'}
        (self.config.result_dir/'result.json').write_text(json.dumps({**self.payload,'dropper_size_ranking':missing['dropper_size_ranking']}))
        with self.assertRaisesRegex(ValueError,'disagrees'):load_scene(self.config)

    def test_default_dual_config_rejects_old_single_result(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy,payload=build_scene(temp,dual=False)
            self.assertEqual(load_scene(legacy).dropper_instance_ids,{})
            with self.assertRaisesRegex(ValueError,'dropper_a'):
                load_scene(replace(legacy,dropper_count=2))

    def test_cli_all_and_dropper_alias_are_noninteractive_offline(self):
        for stage in ('all','dropper','dropper_1','dropper_2'):
            with self.subTest(stage=stage),patch('run_three_stage_demo.ThreeStageConfig.load',return_value=self.config),patch('affordance.three_stage_observation.reuse_latest_result',return_value=self.config),patch('builtins.input',side_effect=AssertionError('stdin')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--observe','0','--stage',stage,'--yes']),0)
        self.assertEqual(parser().parse_args(['--stage','dropper_small']).stage,'dropper_small')

    def test_cli_single_override_plans_legacy_stages_and_submits_three_objects(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy, _ = build_scene(temp, dual=False)
            configured_dual = replace(legacy, dropper_count=2)
            output = io.StringIO()
            with patch('run_three_stage_demo.ThreeStageConfig.load', return_value=configured_dual), \
                 patch('affordance.three_stage_observation.reuse_latest_result', side_effect=lambda config: config) as reuse, \
                 patch('builtins.input', side_effect=AssertionError('stdin')), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(main(['--dropper-count','1','--observe','0','--stage','all','--yes']), 0)
            selected = reuse.call_args.args[0]
            self.assertEqual(selected.dropper_count, 1)
            self.assertEqual(selected.select_chemistry_stages('all'), ('stopper','bottle','dropper'))
            plan = json.loads((legacy.result_dir/'local_three_stage_plan.json').read_text())
            self.assertEqual([stage['name'] for stage in plan['stages']], ['stopper','bottle','dropper'])
            calls = []
            observer = ThreeStageObservation(selected, robot=Mock(), camera=Mock(),
                submitter=lambda args: calls.append(args) or 0)
            observer._submit(selected.capture_dir, Path(temp)/'out', 'test-token')
            self.assertEqual(calls[0].objects, 'wide_mouth_bottle,beaker,dropper')
            self.assertIn('旧三物体/单滴瓶', output.getvalue())

    def test_saved_plan_has_role_and_source_audit(self):
        workflow=ThreeStageWorkflow(self.config);plans=workflow.plan_all(self.scene,workflow.captured_tcp_pose())
        record=json.loads(workflow.save_offline_plan(self.scene,plans).read_text())
        self.assertEqual(record['derived_geometry']['dropper_instance_ids'],{'small':'dropper_b','large':'dropper_a'})
        self.assertEqual(record['stages'][2]['metadata']['reagent_role'],'phenolphthalein')
        self.assertEqual(record['stages'][3]['metadata']['reagent_role'],'acid')

    def test_execution_logs_two_distinct_stages_and_only_returns_each_to_own_bottle(self):
        robot,gripper=Mock(),Mock();pose=[-.35,.15,.3,3.14159,0,0]
        robot.actual_tcp_pose.side_effect=lambda:list(pose)
        robot.move_linear.side_effect=lambda target:pose.__setitem__(slice(None),list(target))
        workflow=ThreeStageWorkflow(self.config,robot=robot,gripper=gripper)
        with patch('affordance.three_stage_workflow.time.sleep'):
            output=workflow.execute(self.scene,('dropper',),lambda *_:None)
        done=json.loads((output/'completed.json').read_text())
        self.assertEqual(done['stages'],['dropper_small','dropper_large'])
        events=[json.loads(s) for s in (output/'events.jsonl').read_text().splitlines()]
        releases=[e['stage'] for e in events if e['step']=='release_dropper' and e['status']=='completed']
        self.assertEqual(releases,['dropper_small','dropper_large'])
        robot.disconnect.assert_called_once();gripper.disconnect.assert_called_once()

    def test_failed_first_dropper_does_not_execute_second(self):
        robot,gripper=Mock(),Mock();robot.actual_tcp_pose.return_value=[-.35,.15,.3,3.14159,0,0]
        workflow=ThreeStageWorkflow(self.config,robot=robot,gripper=gripper)
        seen=[]
        def fail(plan,*args):
            seen.append(plan.name);raise RuntimeError('first dropper failed')
        with patch.object(workflow,'_run_stage',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'first dropper failed'):
                workflow.execute(self.scene,('dropper',),lambda *_:None)
        self.assertEqual(seen,['dropper_small'])


if __name__=='__main__':unittest.main()
