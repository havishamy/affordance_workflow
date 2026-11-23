#!/usr/bin/env python3
"""Locate a tabletop stopper with raw depth; optionally grasp from the current view."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from affordance.camera import RealSenseWristCamera
from affordance.depth_bulb_grasp import (
    capture_current_view, latest_complete_depth_capture, load_saved_capture,
    validate_depth_calibration,
)
from affordance.depth_tabletop_grasp import (
    estimate_tabletop_stopper, save_tabletop_diagnostic, tabletop_grasp_plan,
)
from affordance.hardware import DHGripper, URRobot
from affordance.geometry import pose_to_matrix, rotvec_to_matrix
from affordance.three_stage_config import ThreeStageConfig


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).parent/'config/three_stage_demo.yaml'))
    p.add_argument('--capture-dir', help='Offline RGB-D capture; default newest complete capture')
    p.add_argument('--live', action='store_true', help='Capture the current wrist view; no observation-pose motion')
    p.add_argument('--execute', action='store_true', help='Requires --live and measured or explicitly operator-confirmed table clearance; approach and close')
    p.add_argument('--yes', action='store_true', help='Accept motion checkpoints without reading stdin')
    p.add_argument('--roi', type=int, nargs=4, metavar=('X0','Y0','X1','Y1'))
    p.add_argument('--target-pixel', type=int, nargs=2, metavar=('U','V'))
    p.add_argument('--color', choices=('beige','none'), default='beige', help='Depth handles segmentation; beige is an optional color filter')
    p.add_argument('--center', choices=('modal','mean','median'), default='modal')
    p.add_argument('--offset-mm', type=float, nargs=3, default=[0,0,0], metavar=('X','Y','Z'))
    p.add_argument('--finger-below-tcp-mm', type=float, help='Measured lowest finger extension below TCP in the downward grasp orientation; 0 only if TCP is at the lowest tip')
    p.add_argument('--operator-confirmed-table-clearance', action='store_true', help='Operator explicitly confirms finger/table clearance instead of supplying a finger measurement; does not override a measured collision or workspace/IK limits')
    p.add_argument('--lift-mm', type=float, default=0, help='Optional vertical lift after closing; default holds at grasp position')
    p.add_argument('--gripper-position', type=int, help='Defaults to configured stopper grasp position')
    return p


def checkpoint(message, automatic):
    print(message, flush=True)
    if automatic:
        print('--yes：自动通过确认点。', flush=True)
    elif input('输入 EXECUTE TABLETOP 继续，其他输入取消: ').strip().upper() != 'EXECUTE TABLETOP':
        raise RuntimeError('Operator cancelled tabletop grasp.')


def saved_start_for_current_tcp(info, tcp_offset):
    saved = pose_to_matrix(info['actual_tcp_pose'])
    converted = saved@np.linalg.inv(pose_to_matrix(info['tcp_offset']))@pose_to_matrix(tcp_offset)
    rotation, _ = cv2.Rodrigues(converted[:3,:3])
    return [*converted[:3,3].tolist(), *rotation.reshape(3).tolist()]


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.execute and not args.live:
        p.error('--execute requires --live; saved points cannot authorize motion in a changed scene')
    if args.live and args.capture_dir:
        p.error('--capture-dir is offline only')
    if args.execute and args.finger_below_tcp_mm is None and not args.operator_confirmed_table_clearance:
        p.error('--execute requires --finger-below-tcp-mm or explicit --operator-confirmed-table-clearance')
    if args.finger_below_tcp_mm is not None and (not np.isfinite(args.finger_below_tcp_mm) or args.finger_below_tcp_mm < 0):
        p.error('finger extension must be finite and nonnegative')
    if not np.all(np.isfinite(args.offset_mm)) or not np.isfinite(args.lift_mm) or args.lift_mm < 0:
        p.error('offset/lift must be finite; lift must be nonnegative')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    robot=gripper=None;directory=out=None
    try:
        config=ThreeStageConfig.load(args.config)
        calibration=validate_depth_calibration(config)
        if args.execute and (not config.physical_execution_validated or not config.base.calibration.validated):
            raise RuntimeError('Physical execution/calibration must be validated.')
        if args.live:
            robot=URRobot(config.base.robot);robot.connect()
            directory=capture_current_view(config,robot,RealSenseWristCamera(config.base.camera))
        else:
            directory=Path(args.capture_dir).expanduser().resolve() if args.capture_dir else latest_complete_depth_capture(config.base.output_dir)
        color,depth,intrinsics,scale,transform=load_saved_capture(directory)
        estimate,points,mask=estimate_tabletop_stopper(
            color,depth,intrinsics,scale,transform,roi=args.roi,target_pixel=args.target_pixel,
            color_hint=args.color,center_method=args.center,center_config=config.grasp_center,
        )
        estimate['calibration_path']=str(calibration)
        estimate['calibration_sha256']=hashlib.sha256(calibration.read_bytes()).hexdigest()
        estimate['tcp_offset']=list(config.base.robot.tcp_offset)
        estimate['capture_id']=directory.name
        info_path=directory/'capture_info.json'
        if info_path.is_file():
            estimate['capture_tcp_offset']=json.loads(info_path.read_text())['tcp_offset']
        current=robot.actual_tcp_pose() if args.live else saved_start_for_current_tcp(
            json.loads((directory/'capture_info.json').read_text()),config.base.robot.tcp_offset,
        )
        plan=tabletop_grasp_plan(
            config,current,estimate,offset=np.array(args.offset_mm)/1000,
            finger_below_tcp_m=None if args.finger_below_tcp_mm is None else args.finger_below_tcp_mm/1000,
            lift_m=args.lift_mm/1000,
            operator_confirmed_clearance=args.operator_confirmed_table_clearance,
        )
        position=config.stopper.gripper_position if args.gripper_position is None else args.gripper_position
        if not 0 <= position < config.base.gripper.open_position:
            raise ValueError('Gripper position must be within [0, open_position).')
        plan['gripper_position']=position
        plan['gripper_force']=config.base.gripper.force
        out=save_tabletop_diagnostic(directory,color,estimate,points,mask,plan)
        print(f'当前采集: {directory}',flush=True)
        print(f'橡胶塞点数: {len(points)}；中心算法: {args.center}',flush=True)
        print(f'点云中心 base(m): {estimate["center_world_m"]}',flush=True)
        print(f'中心距桌面: {estimate["center_height_above_table_m"]*1000:.2f} mm',flush=True)
        print(f'桌面/夹指净空: {plan["table_clearance"]}',flush=True)
        print(f'点云、掩码及计划: {out}',flush=True)
        if not args.execute:
            print('仅定位/规划，未发送机械臂运动或夹爪指令。',flush=True)
            return 0
        if plan['table_clearance']['status'] not in ('clear', 'operator_confirmed_unmeasured'):
            raise RuntimeError(f'Finger-to-table clearance rejects grasp: {plan["table_clearance"]}')
        for name,pose in plan['poses'].items():
            if not robot.has_inverse_kinematics(pose):
                raise RuntimeError(f'No IK for tabletop {name}.')
            if not robot.joints_within_safety_limits(robot.inverse_kinematics(pose)):
                raise RuntimeError(f'Joint safety limits reject tabletop {name}.')
        checkpoint(f'Review {out/"mask_overlay.png"} and {out/"grasp_plan.json"}; gripper must be empty and travel clear.',args.yes)
        # Replan if the operator moved the robot during target review.
        after=robot.actual_tcp_pose()
        if np.linalg.norm(np.array(after[:3])-np.array(current[:3])) > config.base.robot.position_tolerance_m:
            raise RuntimeError('Robot moved during target review; recapture/replan required.')
        relative = rotvec_to_matrix(current[3:]).T@rotvec_to_matrix(after[3:])
        if np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)) > config.orientation_tolerance_rad:
            raise RuntimeError('Robot rotated during target review; recapture/replan required.')
        gripper=DHGripper(config.base.gripper);gripper.connect()
        gripper.set_force(config.base.gripper.force);gripper.open()
        events=[]
        def move(name):
            target=plan['poses'][name]
            robot.move_linear(target,speed=plan['speed_mps'],acceleration=plan['acceleration_mps2'])
            events.append(dict(step=name,target_tcp_pose=target,actual_tcp_pose=robot.actual_tcp_pose()))
            (out/'motion_events.json').write_text(json.dumps(events,indent=2))
        for name in ('raise_to_transit','over_object','pregrasp'):
            move(name)
        checkpoint('At pregrasp: verify finger-to-table clearance before descending.',args.yes)
        move('grasp')
        checkpoint('At cloud center: verify the jaws straddle the stopper before closing.',args.yes)
        gripper.set_position(position)
        time.sleep(config.stopper.settle_s)
        if 'lift' in plan['poses']:
            checkpoint('Verify grasp before the optional vertical lift.',args.yes)
            move('lift')
        (out/'completed.json').write_text(json.dumps(dict(
            final_tcp_pose=robot.actual_tcp_pose(),gripper_position=position,
            lift_m=args.lift_mm/1000,object_grasp_success='not independently measured',
        ),indent=2))
        print('已到达点云中心并闭爪保持；实际夹持是否成功需现场观察。',flush=True)
        return 0
    except KeyboardInterrupt:
        if robot is not None:robot.stop()
        logging.error('Operator interrupted tabletop grasp.')
        return 130
    except Exception as exc:
        if robot is not None:robot.stop()
        logging.exception('Tabletop grasp failed: %s',exc)
        if directory is not None:
            failure=directory/'tabletop_stopper_diagnostic';failure.mkdir(exist_ok=True)
            (failure/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc)),ensure_ascii=False,indent=2))
        return 1
    finally:
        if gripper is not None:gripper.disconnect()
        if robot is not None:robot.disconnect()


if __name__=='__main__':
    sys.exit(main())
