import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from affordance.base_table_plane import table_initialization_reference, validate_table_initialization_result
from affordance.geometry import rotvec_to_matrix
from affordance.three_stage_result import _read_ply_xyz
from complex_capture_client import build_parser, submit_and_wait
from fixture_scenes import build_scene
from server_base_table_reference import (
    reference_request, parse_reference, initialize_pose_z_only, result_confirmation, reference_in_camera,
)


class ServerBaseTableReferenceTests(unittest.TestCase):
    def initialization(self):
        vertices = np.random.default_rng(9).uniform(-0.5, 0.5, (1200, 3))
        pose = np.eye(4)
        pose[:3,:3] = rotvec_to_matrix([0.2, -0.3, 0.7])
        pose[:3,3] = [-0.45, 0.21, 0.18]
        scale = np.array([0.16, 0.12, 0.2])
        after, audit = initialize_pose_z_only(vertices, pose, scale, reference_request())
        return vertices, pose, scale, after, audit

    def test_server_client_and_json_sample_share_the_same_contract(self):
        self.assertEqual(reference_request(), table_initialization_reference())
        path = Path(__file__).resolve().parents[1]/'config/table_plane_reference.json'
        self.assertEqual(json.loads(path.read_text()), reference_request())
        self.assertIsNone(parse_reference(None))
        self.assertEqual(parse_reference(json.dumps(reference_request())), reference_request())
        with self.assertRaises(ValueError):
            parse_reference(json.dumps({**reference_request(), 'preserve_object_xy': False}))

    def test_real_initialization_changes_only_z_and_leaves_model_inputs_unchanged(self):
        vertices, before, scale, after, audit = self.initialization()
        self.assertTrue(np.array_equal(before[:2,3], after[:2,3]))
        self.assertTrue(np.array_equal(before[:3,:3], after[:3,:3]))
        world_before = (vertices*scale)@before[:3,:3].T+before[:3,3]
        world_after = (vertices*scale)@after[:3,:3].T+after[:3,3]
        np.testing.assert_array_equal(world_before[:,:2], world_after[:,:2])
        self.assertLess(abs(world_after[:,2].min()), 1e-12)
        np.testing.assert_allclose(np.ptp(world_before,axis=0), np.ptp(world_after,axis=0), atol=1e-12)
        np.testing.assert_array_equal(scale, audit['scale_after'])
        np.testing.assert_array_equal(vertices, np.random.default_rng(9).uniform(-0.5,0.5,(1200,3)))
        payload = {'table_plane_reference': result_confirmation(reference_request(), {'wide_mouth_bottle': audit})}
        validate_table_initialization_result(payload, ['wide_mouth_bottle'])

    def test_fixed_base_plane_is_expressed_in_camera_coordinates_without_moving_points(self):
        transform = np.eye(4)
        transform[:3,:3] = rotvec_to_matrix([0.4,-0.2,0.8])
        transform[:3,3] = [-0.3,0.1,0.4]
        points = np.random.default_rng(5).normal(size=(100,3))
        plane = reference_in_camera(transform)
        base = points@transform[:3,:3].T+transform[:3,3]
        np.testing.assert_allclose(points@np.array(plane['normal'])+plane['offset_m'], base[:,2], atol=1e-12)

    def test_client_rejects_missing_confirmation_xy_rotation_or_scale_changes(self):
        _, _, _, _, audit = self.initialization()
        ack = result_confirmation(reference_request(), {'wide_mouth_bottle': audit})
        invalid = [{}, {'table_plane_reference': {**ack,'applied':False}}]
        for mutation in ('xy','rotation','scale','support','missing_object'):
            changed = json.loads(json.dumps(ack))
            obj = changed['objects']['wide_mouth_bottle']
            if mutation=='xy':obj['T_world_object_after'][0][3]+=0.01
            if mutation=='rotation':
                obj['T_world_object_after'] = (np.block([[rotvec_to_matrix([0,0,0.1]),np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]) @ np.array(obj['T_world_object_after'])).tolist()
            if mutation=='scale':obj['scale_after'][0]*=1.01
            if mutation=='support':obj['support_min_z_after_m']=0.02
            if mutation=='missing_object':changed['objects']={}
            invalid.append({'table_plane_reference':changed})
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_table_initialization_result(payload,['wide_mouth_bottle'])

    def test_single_and_dual_multipart_round_trip_with_real_server_initialization_audits(self):
        class Response:
            ok=True
            status_code=200
            def __init__(self,payload=None,data=b''):self.payload,self.data=payload,data
            def json(self):return self.payload
            def __enter__(self):return self
            def __exit__(self,*_):return False
            def raise_for_status(self):pass
            def iter_content(self,_):yield self.data

        for dual in (False,True):
            with self.subTest(dual=dual), tempfile.TemporaryDirectory() as directory:
                config, payload = build_scene(directory,dual=dual)
                audits={}
                for name,obj in payload['objects'].items():
                    vertices=_read_ply_xyz(config.result_dir/'models'/f'{name}_canonical.ply')
                    _,audits[name]=initialize_pose_z_only(vertices,obj['T_world_object'],obj['scale'],reference_request())
                payload['table_plane_reference']=result_confirmation(reference_request(),audits)
                bundle=io.BytesIO()
                with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
                    for path in config.result_dir.rglob('*'):
                        if path.is_file() and path.name!='result.json':archive.write(path,path.relative_to(config.result_dir))
                    archive.writestr('result.json',json.dumps(payload))
                args=build_parser().parse_args(['--token','test-token','run','--capture-dir',str(config.capture_dir),
                    '--output-dir',str(Path(directory)/'download'),'--scope','all','--objects',','.join(config.object_names),
                    '--table-plane-reference',str(Path(__file__).resolve().parents[1]/'config/table_plane_reference.json')])
                measured_pose=json.loads((config.capture_dir/'T_world_ee.json').read_text())
                def post(url,**kwargs):
                    self.assertEqual(json.loads(kwargs['data']['table_plane_reference']),reference_request())
                    self.assertEqual(json.load(kwargs['files']['world_ee'][1]),measured_pose)
                    return Response({'job_id':'synthetic-job','state':'queued'})
                def get(url,**kwargs):
                    if url.endswith('/result'):return Response(payload)
                    if url.endswith('/bundle'):return Response(data=bundle.getvalue())
                    return Response({'job_id':'synthetic-job','state':'completed'})
                with patch('complex_capture_client.requests.post',side_effect=post), \
                     patch('complex_capture_client.requests.get',side_effect=get), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(submit_and_wait(args),0)
                saved=json.loads((Path(args.output_dir)/'result.json').read_text())
                validate_table_initialization_result(saved,config.object_names)
                for name in config.object_names:
                    np.testing.assert_array_equal(saved['objects'][name]['T_world_object'],payload['objects'][name]['T_world_object'])
                    self.assertEqual(saved['objects'][name]['scale'],payload['objects'][name]['scale'])


if __name__=='__main__':unittest.main()
