"""Shared point-region grasp estimation and virtual receiving-opening geometry.

Only NumPy is required. A receiving opening is inferred free space, never a
claim that a camera measured material in the hollow center of a vessel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _xyz(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1:] != (3,) or len(value) < 3 or not np.all(np.isfinite(value)):
        raise ValueError("Target geometry requires at least three finite XYZ points.")
    return value


def unit_axis(axis) -> np.ndarray:
    value = np.asarray(axis, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.linalg.norm(value) < 1e-9:
        raise ValueError("Target axis must contain three finite values and be nonzero.")
    return value / np.linalg.norm(value)


def plane_basis(normal) -> np.ndarray:
    normal = unit_axis(normal)
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = unit_axis(np.cross(normal, helper))
    return np.column_stack((first, np.cross(normal, first)))


@dataclass(frozen=True)
class GraspCenterConfig:
    bandwidth_m: float = 0.003
    bin_width_m: float = 0.0005

    def validate(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (self.bandwidth_m, self.bin_width_m)):
            raise ValueError("Grasp-center bandwidth/bin width must be finite and positive.")
        if self.bin_width_m > self.bandwidth_m:
            raise ValueError("Grasp-center bin width must not exceed bandwidth.")


def weighted_modal_center(points, scores=None, axis=(0, 0, 1), *, config=None, bounds=None):
    """Find the dominant axial layer, then compute its full XYZ weighted center.

    Bounds constrain *candidate points before peak selection*. This avoids
    picking an inaccessible base peak and then clipping its center afterward.
    Zero-score points never vote; absent scores explicitly mean uniform weights.
    """
    config = config or GraspCenterConfig()
    config.validate()
    points = _xyz(points)
    normal = unit_axis(axis)
    weights = np.ones(len(points)) if scores is None else np.asarray(scores, dtype=float)
    if weights.shape != (len(points),) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Affordance scores must be finite, nonnegative and aligned with XYZ points.")
    positive = weights > 0
    eligible = positive.copy()
    if bounds is not None:
        lower, upper = (np.asarray(v, dtype=float) for v in bounds)
        if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper):
            raise ValueError("Grasp-center bounds must be finite ordered XYZ bounds.")
        eligible &= np.all((points >= lower) & (points <= upper), axis=1)
    if np.count_nonzero(eligible) < 3:
        raise ValueError("Fewer than three positive-weight grasp points remain within candidate bounds.")
    selected, score = points[eligible], weights[eligible]
    score = score / np.max(score)
    origin = np.average(selected, axis=0, weights=score)
    axial = (selected - origin) @ normal
    step, width = config.bin_width_m, config.bandwidth_m
    padding = int(np.ceil(4 * width / step))
    start = np.floor(axial.min() / step) * step - padding * step
    end = np.ceil(axial.max() / step) * step + padding * step
    bins = int(round((end - start) / step)) + 1
    if bins > 20000:
        raise ValueError("Grasp-region axial extent is too large for configured bin width.")
    edges = start + np.arange(bins + 1) * step
    histogram = np.histogram(axial, edges, weights=score)[0]
    offsets = np.arange(-padding, padding + 1) * step
    kernel = np.exp(-0.5 * (offsets / width) ** 2)
    density = np.convolve(histogram, kernel / kernel.sum(), mode="same")
    peak = int(np.argmax(density))
    mode = float((edges[peak] + edges[peak + 1]) / 2)
    local_weights = score * np.exp(-0.5 * ((axial - mode) / width) ** 2)
    center = np.average(selected, axis=0, weights=local_weights)
    report = {
        "method": "axial_density_mode_local_weighted_mean",
        "axis_world": normal.tolist(), "bandwidth_m": width, "bin_width_m": step,
        "input_points": len(points), "positive_score_points": int(positive.sum()),
        "eligible_points": int(eligible.sum()),
        "excluded_by_bounds": int(np.count_nonzero(positive & ~eligible)),
        "score_source": "uniform_no_scores" if scores is None else "affordance_scores",
        "mode_axis_coordinate_m": float(origin @ normal + mode),
        "effective_weighted_points": float(local_weights.sum() ** 2 / (local_weights @ local_weights)),
        "center_world_m": center.tolist(),
    }
    return center, report


@dataclass(frozen=True)
class OpeningConfig:
    rim_band_m: float = 0.004
    upper_quantile: float = 0.995
    wall_margin_m: float = 0.002
    receiving_margin_m: float = 0.003
    angular_bins: int = 72

    def validate(self) -> None:
        if not math.isfinite(self.rim_band_m) or self.rim_band_m <= 0:
            raise ValueError("Opening rim band must be finite and positive.")
        if not math.isfinite(self.upper_quantile) or not 0.9 <= self.upper_quantile < 1:
            raise ValueError("Opening upper quantile must be in [0.9,1).")
        if any(not math.isfinite(v) or v < 0 for v in (self.wall_margin_m, self.receiving_margin_m)):
            raise ValueError("Opening margins must be finite and nonnegative.")
        if self.angular_bins < 12 or self.angular_bins > 720:
            raise ValueError("Opening angular bins must be within [12,720].")


def opening_from_model(points, axis=(0, 0, 1), *, config=None):
    """Infer a rim plane and its hollow disk from the upper model perimeter.

    Angular perimeter selection rejects dense filled top faces and lower cup
    walls. Equal angular support and robust circle refinement limit spout bias.
    The fitted radius is reduced by a configured wall margin, not described as
    a physically measured inner-wall thickness.
    """
    config = config or OpeningConfig()
    config.validate()
    points = _xyz(points)
    normal = unit_axis(axis)
    if normal[2] < 0:
        normal = -normal
    basis = plane_basis(normal)
    origin = points.mean(axis=0)
    heights = (points - origin) @ normal
    level = float(np.quantile(heights, config.upper_quantile))
    upper = points[(heights >= level - config.rim_band_m) & (heights <= level)]
    if len(upper) < 12:
        raise ValueError("Not enough upper-rim model points to infer a receiving opening.")
    xy = (upper - origin) @ basis
    seed = np.median(xy, axis=0)
    angles = np.mod(np.arctan2(xy[:, 1] - seed[1], xy[:, 0] - seed[0]), 2 * np.pi)
    sectors = np.minimum((angles * config.angular_bins / (2 * np.pi)).astype(int), config.angular_bins - 1)
    radius = np.linalg.norm(xy - seed, axis=1)
    selected, initial_weights = [], []
    for sector in range(config.angular_bins):
        ids = np.flatnonzero(sectors == sector)
        if len(ids):
            boundary = ids[radius[ids] >= np.quantile(radius[ids], 0.8)]
            selected.extend(boundary.tolist())
            initial_weights.extend([1 / len(boundary)] * len(boundary))
    rim = xy[selected]
    base_weights = np.asarray(initial_weights)
    if len(rim) < 12:
        raise ValueError("Upper rim has insufficient perimeter support.")
    weights = base_weights.copy()
    A = np.column_stack((2 * rim, np.ones(len(rim))))
    b = np.sum(rim * rim, axis=1)
    for _ in range(6):
        coefficient, _, rank, _ = np.linalg.lstsq(A * np.sqrt(weights[:, None]), b * np.sqrt(weights), rcond=None)
        if rank < 3:
            raise ValueError("Receiving rim is degenerate; cannot fit an opening.")
        center_xy = coefficient[:2]
        radius_squared = float(coefficient[2] + center_xy @ center_xy)
        if radius_squared <= 0:
            raise ValueError("Receiving rim radius is invalid.")
        outer_radius = float(np.sqrt(radius_squared))
        residual = np.linalg.norm(rim - center_xy, axis=1) - outer_radius
        scale = max(0.0005, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = base_weights * np.minimum(1, 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
    center = origin + basis @ center_xy + normal * level
    inner_radius = outer_radius - config.wall_margin_m
    receiving_radius = inner_radius - config.receiving_margin_m
    if receiving_radius <= 0:
        raise ValueError("Receiving opening is smaller than the configured wall/edge margins.")
    ordered = np.sort(np.mod(np.arctan2(rim[:, 1] - center_xy[1], rim[:, 0] - center_xy[0]), 2 * np.pi))
    span = float(2 * np.pi - np.max(np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])))
    return {
        "method": "upper_model_perimeter_circle_on_opening_plane",
        "geometry_type": "virtual_hollow_opening_disk", "synthetic_target_geometry": True,
        "center_world_m": center.tolist(), "normal_world": normal.tolist(),
        "basis_world": basis.tolist(), "outer_radius_m": outer_radius,
        "inner_radius_estimate_m": inner_radius, "receiving_radius_m": receiving_radius,
        "wall_margin_m": config.wall_margin_m, "receiving_margin_m": config.receiving_margin_m,
        "rim_band_m": config.rim_band_m, "upper_quantile": config.upper_quantile,
        "rim_points": len(rim), "upper_model_points": len(upper),
        "residual_rms_m": float(np.sqrt(np.average(residual ** 2, weights=weights))),
        "arc_span_rad": span,
        "note": "Inferred free space at the rim plane; not measured wall or liquid surface points.",
    }


def sample_opening_disk(opening, rings=12, sectors=72):
    """Generate explicitly synthetic points *inside* the receiving opening."""
    center = np.asarray(opening["center_world_m"])
    basis = np.asarray(opening["basis_world"])
    radius = float(opening["receiving_radius_m"])
    angles = np.arange(sectors) * (2 * np.pi / sectors)
    direction = np.column_stack((np.cos(angles), np.sin(angles))) @ basis.T
    return np.vstack([center, *(center + radius * (ring / rings) * direction for ring in range(1, rings + 1))])
