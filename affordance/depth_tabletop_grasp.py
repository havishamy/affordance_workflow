"""Depth-table subtraction and point-cloud centers for a visible tabletop stopper."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .target_geometry import GraspCenterConfig, weighted_modal_center
from .geometry import rotvec_to_matrix, validate_transform


@dataclass(frozen=True)
class TabletopConfig:
    depth_min_m: float = .15
    depth_max_m: float = 1.0
    plane_tolerance_m: float = .003
    min_height_m: float = .005
    max_height_m: float = .065
    minimum_pixels: int = 80
    minimum_table_fraction: float = .4
    minimum_up_cosine: float = .85

    def validate(self):
        vals = (self.depth_min_m, self.depth_max_m, self.plane_tolerance_m, self.min_height_m, self.max_height_m)
        if not np.all(np.isfinite(vals)) or not 0 < self.depth_min_m < self.depth_max_m:
            raise ValueError('Invalid tabletop depth range.')
        if not 0 < self.plane_tolerance_m < self.min_height_m < self.max_height_m:
            raise ValueError('Require plane tolerance < object minimum height < maximum height.')
        if self.minimum_pixels < 3 or not 0 < self.minimum_table_fraction <= 1 or not 0 < self.minimum_up_cosine <= 1:
            raise ValueError('Invalid table support or component settings.')


def fit_table_plane(points, config=TabletopConfig()):
    config.validate()
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.all(np.isfinite(points)):
        raise ValueError('Insufficient valid depth points for the table plane.')
    rng = np.random.default_rng(20)
    sample = points[rng.choice(len(points), min(6000, len(points)), replace=False)]
    best_count, best = 0, None
    for _ in range(250):
        q = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(q[1]-q[0], q[2]-q[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < config.minimum_up_cosine:
            continue
        offset = -float(normal@q[0])
        count = np.count_nonzero(np.abs(sample@normal+offset) <= config.plane_tolerance_m)
        if count > best_count:
            best_count, best = count, (normal, offset)
    if best is None or best_count/len(sample) < config.minimum_table_fraction:
        raise ValueError('No sufficiently supported upward-facing tabletop plane; do not grasp.')
    normal, offset = best
    for _ in range(3):
        selected = sample[np.abs(sample@normal+offset) <= config.plane_tolerance_m]
        origin = selected.mean(axis=0)
        _, _, vectors = np.linalg.svd(selected-origin, full_matrices=False)
        normal = vectors[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -float(normal@origin)
    if normal[2] < config.minimum_up_cosine:
        raise ValueError('Refined table plane is not sufficiently horizontal in UR base.')
    residual = sample@normal+offset
    inliers = np.abs(residual) <= config.plane_tolerance_m
    fraction = float(inliers.mean())
    if fraction < config.minimum_table_fraction:
        raise ValueError('Refined table plane lost depth support.')
    return dict(normal_world=normal.tolist(), offset_m=offset,
                inlier_fraction=fraction, tolerance_m=config.plane_tolerance_m,
                residual_rms_m=float(np.sqrt(np.mean(residual[inliers]**2))))


def estimate_tabletop_stopper(color, depth, intrinsics, scale, transform, *,
                              roi=None, target_pixel=None, color_hint='beige',
                              center_method='modal', config=TabletopConfig(),
                              center_config=GraspCenterConfig()):
    config.validate()
    transform = validate_transform(transform, 'T_base_camera')
    if depth.dtype != np.uint16 or color.dtype != np.uint8 or color.shape != (*depth.shape, 3):
        raise ValueError('Expected aligned uint16 depth and uint8 BGR.')
    height, width = depth.shape
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy'))
    if intrinsics.get('width') != width or intrinsics.get('height') != height or not np.all(np.isfinite([fx, fy, cx, cy, scale])) or min(fx, fy, scale) <= 0:
        raise ValueError('Invalid aligned-depth intrinsics or scale.')
    valid = (depth*scale >= config.depth_min_m)&(depth*scale <= config.depth_max_m)
    v, u = np.nonzero(valid)
    z = depth[v, u]*scale
    camera = np.column_stack(((u-cx)*z/fx, (v-cy)*z/fy, z))
    world = camera@transform[:3, :3].T+transform[:3, 3]
    plane = fit_table_plane(world, config)
    normal, offset = np.asarray(plane['normal_world']), plane['offset_m']
    height_map = np.full(depth.shape, np.nan)
    height_map[v, u] = world@normal+offset
    roi = tuple(roi) if roi is not None else (0, 0, width, height)
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
        raise ValueError('Invalid object ROI.')
    eligible = valid&(height_map >= config.min_height_m)&(height_map <= config.max_height_m)
    crop = np.zeros(depth.shape, dtype=bool)
    crop[roi[1]:roi[3], roi[0]:roi[2]] = True
    eligible &= crop
    if color_hint == 'beige':
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        # Hue is unstable near gray: allow low saturation under the camera's
        # green cast; depth-above-table, not color, separates the foreground.
        beige = ((hsv[:, :, 1] <= 65)|((hsv[:, :, 0] <= 45)&(hsv[:, :, 1] <= 160)))&(hsv[:, :, 2] >= 65)
        eligible &= beige
    elif color_hint != 'none':
        raise ValueError('Color hint must be beige or none.')
    mask = cv2.morphologyEx(eligible.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Morphology must never introduce table pixels or invalid depth.
    mask &= eligible
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates, excluded = [], []
    for index in range(1, count):
        x, y, w, h, area = (int(a) for a in stats[index])
        if area < config.minimum_pixels:
            continue
        record = dict(label=index, bbox_xyxy=[x, y, x+w, y+h], pixels=area,
                      center_pixel_uv=centers[index].tolist())
        if x <= roi[0] or y <= roi[1] or x+w >= roi[2] or y+h >= roi[3]:
            excluded.append(dict(record, reason='partial object touches search boundary'))
        else:
            candidates.append(record)
    if target_pixel is not None:
        px, py = target_pixel
        if not (0 <= px < width and 0 <= py < height):
            raise ValueError('Target pixel is outside the image.')
        candidates = [c for c in candidates if c['label'] == int(labels[py, px])]
    if len(candidates) != 1:
        raise ValueError(f'Expected one fully visible stopper, found {len(candidates)} candidates: {candidates}. Use --roi or --target-pixel.')
    selected_mask = labels == candidates[0]['label']
    selected = selected_mask[v, u]
    points = world[selected]
    mean, median = points.mean(axis=0), np.median(points, axis=0)
    modal, center_report = weighted_modal_center(points, axis=normal, config=center_config)
    if center_method not in ('modal', 'mean', 'median'):
        raise ValueError('Unknown point-cloud center method.')
    center = dict(modal=modal, mean=mean, median=median)[center_method]
    point_camera = transform[:3, :3].T@(center-transform[:3, 3])
    projected = [point_camera[0]/point_camera[2]*fx+cx, point_camera[1]/point_camera[2]*fy+cy]
    report = dict(method='raw_depth_table_subtraction', color_hint=color_hint,
                  selected_pixels=len(points), selected_bbox_xyxy=candidates[0]['bbox_xyxy'],
                  center_method=center_method, center_estimation=center_report,
                  center_world_m=center.tolist(), center_camera_m=point_camera.tolist(),
                  center_pixel_uv=projected, mean_world_m=mean.tolist(), median_world_m=median.tolist(),
                  modal_world_m=modal.tolist(), table_plane=plane,
                  center_height_above_table_m=float(center@normal+offset),
                  object_height_quantiles_m=np.quantile(points@normal+offset, [.05, .5, .95]).tolist(),
                  excluded_components=excluded, roi_px=list(roi),
                  T_base_opencv_camera=transform.tolist(),
                  note='Center of observed depth surface, not an inferred full solid-volume center.')
    return report, points, selected_mask.astype(np.uint8)*255


def tabletop_grasp_plan(stage, current, estimate, *, offset=(0, 0, 0),
                       finger_below_tcp_m=None, lift_m=0, clearance_m=.003,
                       operator_confirmed_clearance=False):
    center = np.asarray(estimate['center_world_m'])
    delta = np.asarray(offset, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)) or not np.isfinite(lift_m) or lift_m < 0:
        raise ValueError('Invalid grasp offset or lift distance.')
    if not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('Invalid table clearance.')
    target = center+delta
    approach = stage.stopper.approach_m
    transit = max(stage.transit_z_m, current[2], target[2]+approach, target[2]+lift_m)
    orientation = list(stage.stopper.orientation_rotvec)
    if rotvec_to_matrix(orientation)[2, 2] > -.99:
        raise ValueError('Tabletop finger clearance requires the configured downward grasp orientation.')
    poses = dict(raise_to_transit=[*current[:2], transit, *current[3:]],
                 over_object=[*target[:2], transit, *orientation],
                 pregrasp=[*target[:2], target[2]+approach, *orientation],
                 grasp=[*target, *orientation])
    if lift_m:
        poses['lift'] = [*target[:2], target[2]+lift_m, *orientation]
    lower, upper = np.asarray(stage.base.motion.workspace_min), np.asarray(stage.base.motion.workspace_max)
    for name, pose in [('current', current), *poses.items()]:
        p = np.asarray(pose, dtype=float)
        if p.shape != (6,) or not np.all(np.isfinite(p)) or not np.all((p[:3] >= lower)&(p[:3] <= upper)):
            raise ValueError(f'Tabletop {name} pose is outside the configured workspace: {pose}')
    normal = np.asarray(estimate['table_plane']['normal_world'])
    height = float(target@normal+estimate['table_plane']['offset_m'])
    margin = clearance_m+stage.base.robot.position_tolerance_m+estimate['table_plane']['tolerance_m']
    if finger_below_tcp_m is None:
        check = dict(status='measurement_required', target_height_m=height,
                     maximum_finger_below_tcp_m=float(max(0, (height-margin)/normal[2])), required_margin_m=margin)
        if operator_confirmed_clearance:
            check.update(
                status='operator_confirmed_unmeasured' if height >= margin else 'blocked',
                confirmation_source='operator_explicitly_confirmed_no_table_collision',
                finger_extension_measured=False,
            )
    else:
        if not np.isfinite(finger_below_tcp_m) or finger_below_tcp_m < 0:
            raise ValueError('Finger extension below TCP must be a finite nonnegative measurement.')
        gap = float(height-normal[2]*finger_below_tcp_m)
        check = dict(status='clear' if gap >= margin else 'blocked', target_height_m=height,
                     finger_below_tcp_m=finger_below_tcp_m, estimated_finger_gap_m=gap, required_margin_m=margin)
    return dict(poses={k:[float(v) for v in p] for k,p in poses.items()},
                center_method=estimate['center_method'], grasp_offset_base_m=delta.tolist(),
                target_world_m=target.tolist(), table_clearance=check,
                tcp_offset=list(stage.base.robot.tcp_offset), speed_mps=.01, acceleration_mps2=.02)


def save_tabletop_diagnostic(capture_dir, color, estimate, points, mask, plan):
    out = Path(capture_dir)/'tabletop_stopper_diagnostic'
    out.mkdir(exist_ok=True)
    overlay, tint = color.copy(), color.copy()
    tint[mask > 0] = (255, 0, 255)
    overlay = cv2.addWeighted(overlay, .6, tint, .4, 0)
    x0,y0,x1,y1 = estimate['selected_bbox_xyxy']
    cv2.rectangle(overlay, (x0,y0), (x1-1,y1-1), (0,255,255), 1)
    cv2.drawMarker(overlay, tuple(round(float(v)) for v in estimate['center_pixel_uv']), (0,255,0), cv2.MARKER_CROSS, 14, 2)
    cv2.imwrite(str(out/'mask_overlay.png'), overlay)
    cv2.imwrite(str(out/'mask.png'), mask)
    data = points.astype('<f4')
    header = f'ply\nformat binary_little_endian 1.0\ncomment observed tabletop stopper points in UR base\nelement vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    (out/'points_world.ply').write_bytes(header.encode('ascii')+data.tobytes())
    (out/'estimate.json').write_text(json.dumps(estimate,ensure_ascii=False,indent=2))
    (out/'grasp_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    return out
