"""Self-contained metric-model fixtures; no private capture or hardware required."""
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from affordance.three_stage_config import ThreeStageConfig

ROOT = Path(__file__).resolve().parents[1]


def write_ply(path, xyz):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype='<f4')
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    path.write_bytes(header.encode('ascii')+xyz.tobytes())


def cylinder(center, radius, height, seed):
    rng = np.random.default_rng(seed)
    angle = np.arange(180)*2*np.pi/180
    wall = np.vstack([np.c_[radius*np.cos(angle), radius*np.sin(angle), np.full(len(angle), z)] for z in np.linspace(-height/2,height/2,20)])
    r = radius*np.sqrt(rng.uniform(size=1500)); t = rng.uniform(0,2*np.pi,len(r))
    top = np.c_[r*np.cos(t),r*np.sin(t),np.full(len(r),height/2)]
    return np.vstack((wall,top))+center


def build_scene(root, dual=True):
    root = Path(root)
    capture = root/'synthetic_capture'
    result_dir = capture/'complex_server_result_with_beaker/complex_synthetic_all'
    result_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    objects = {}
    specs = {
        'wide_mouth_bottle': ([-.52,.2,.06],.025,.09,.1),
        'beaker': ([-.43,.35,.04],.03,.08,.1),
    }
    if dual:
        # A is physically bigger while B has the numerically larger canonical
        # scale. Correct classification must inspect restored metric geometry.
        specs.update(dropper_a=([-.52,.38,.08],.025,.16,.08),
                     dropper_b=([-.4,.22,.06],.015,.10,.20))
    else:
        specs['dropper'] = ([-.4,.22,.06],.015,.10,.20)
    for index,(name,(center,radius,height,scale)) in enumerate(specs.items()):
        center = np.array(center)
        world = cylinder(center,radius,height,index)
        canonical = (world-center)/scale
        transform = np.eye(4);transform[:3,3]=center
        artifact = f'models/{name}_world.ply'
        write_ply(result_dir/artifact,world)
        write_ply(result_dir/f'models/{name}_world_mesh.ply',world)
        write_ply(result_dir/f'models/{name}_canonical.ply',canonical)
        pose = result_dir/f'02_sam3d/{name}/pose_result.json'
        pose.parent.mkdir(parents=True);pose.write_text(json.dumps({'silhouette_iou':.95}))
        if name == 'wide_mouth_bottle':
            regions = {'stopper':[-.52,.2,.10], 'body_grasp':[-.52,.2,.055]}
        elif name == 'beaker':
            regions = {'mouth':[-.43,.35,.08]}
        else:
            regions = {'bulb':(center+[0,0,height/2+.01]).tolist()}
        data = {}
        for region,position in regions.items():
            points = rng.normal(position,[.002,.002,.0005],(240,3))
            if name == 'beaker':
                angle = np.linspace(-.8,.8,240)
                points = np.c_[position[0]+radius*np.cos(angle),position[1]+radius*np.sin(angle),np.full(len(angle),position[2])]
            ply = f'affordance_regions/{name}/{region}_world.ply'
            npz = f'affordance_regions/{name}/{region}_world.npz'
            write_ply(result_dir/ply,points)
            np.savez(result_dir/npz,points_world=points.astype(np.float32),scores=np.ones(len(points),np.float32))
            data[region] = dict(points_world=points.tolist(),points_world_artifact=ply,points_world_npz_artifact=npz)
            quality=result_dir/f'01_affordance/{name}/{region}/quality_metrics.json'
            quality.parent.mkdir(parents=True);quality.write_text(json.dumps({'valid':True}))
        objects[name]=dict(model_id=f'synthetic-{name}',T_world_object=transform.tolist(),scale=[scale]*3,
                           model_points_world_artifact=artifact,affordance_regions=data,main_axis_world=[0,0,1])
    (result_dir/'01_affordance/wide_mouth_bottle/separation_metrics.json').write_text(json.dumps({'valid':True,'top_region_iou':0}))
    payload=dict(job_id='synthetic-job',capture_id=capture.name,frame='world',units='meters',
                 model_transform_convention='p_world = T_world_object @ [scale * p_object, 1]',
                 calibration={'ee_frame':'tool0','world_frame':'base'},objects=objects)
    (result_dir/'result.json').write_text(json.dumps(payload))
    (result_dir/'job.json').write_text(json.dumps({'job_id':'synthetic-job','state':'completed'}))
    info=dict(capture_id=capture.name,world_frame='base',ee_frame='flange',actual_tcp_pose=[-.35,.15,.3,3.14159,0,0],
              tcp_offset=[0,0,.163,0,0,0],depth_scale_m_per_unit=.001)
    (capture/'capture_info.json').write_text(json.dumps(info))
    (capture/'T_world_ee.json').write_text(json.dumps({'T_world_ee':np.eye(4).tolist()}))
    (capture/'camera.json').write_text(json.dumps(dict(width=3,height=2,fx=100,fy=100,cx=1,cy=1,depth_scale_m_per_unit=.001)))
    cv2.imwrite(str(capture/'rgb.png'),np.zeros((2,3,3),np.uint8))
    cv2.imwrite(str(capture/'depth.png'),np.full((2,3),400,np.uint16))
    (result_dir/'pipeline.log').write_text('[STAGE] calibration_validation\n'+json.dumps({'T_world_opencv_camera':np.eye(4).tolist()}))
    config=replace(ThreeStageConfig.load(ROOT/'config/three_stage_demo.yaml'),capture_dir=capture,result_dir=result_dir,
                   dropper_count=2 if dual else 1,stirring_enabled=False,table_constraint_enabled=False)
    return config,payload
