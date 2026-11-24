"""Shared dual-instance contract and metric model-size ranking (NumPy only)."""
from __future__ import annotations

import numpy as np

BASE_OBJECTS = ("wide_mouth_bottle", "beaker")
DROPPER_IDS = ("dropper_a", "dropper_b")
SINGLE_OBJECTS = (*BASE_OBJECTS, "dropper")
DUAL_OBJECTS = (*BASE_OBJECTS, *DROPPER_IDS)
LEGACY_CHEMISTRY_STAGES = ("stopper", "bottle", "dropper")
DUAL_CHEMISTRY_STAGES = ("stopper", "bottle", "dropper_small", "dropper_large")
SIZE_METHOD = "pca_obb_volume_m3_q01_q99"


def scene_objects(count: int) -> tuple[str, ...]:
    if count not in (1, 2):
        raise ValueError("dropper count must be 1 or 2.")
    return DUAL_OBJECTS if count == 2 else SINGLE_OBJECTS


def requested_regions(objects) -> dict[str, list[str]]:
    regions = {"wide_mouth_bottle": ["stopper", "body_grasp"], "beaker": ["mouth"]}
    regions.update({name: ["bulb"] for name in ("dropper", *DROPPER_IDS)})
    unknown = set(objects)-set(regions)
    if unknown:
        raise ValueError(f"Unsupported requested objects: {sorted(unknown)}")
    return {name: regions[name] for name in objects}


def metric_model_size(points) -> dict:
    """Robust PCA-oriented box in metres, including restored physical scale.

    Comparing canonical scale values alone is invalid for independently
    generated canonical models. This metric always uses world model vertices.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 10 or not np.all(np.isfinite(points)):
        raise ValueError("Dropper size ranking requires at least 10 finite metric model points.")
    origin = points.mean(axis=0)
    centered = points-origin
    covariance = centered.T@centered/len(points)
    eigenvalues, axes = np.linalg.eigh(covariance)
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    projected = centered@axes
    bounds = np.quantile(projected, [.01, .99], axis=0)
    dimensions = bounds[1]-bounds[0]
    if np.any(dimensions <= 1e-6):
        raise ValueError("Dropper model is degenerate; physical size cannot be ranked.")
    return {"method": SIZE_METHOD, "dimensions_m": dimensions.tolist(),
            "volume_m3": float(np.prod(dimensions)), "point_count": len(points),
            "center_world_m": origin.tolist(), "pca_axes_world": axes.tolist(),
            "quantiles": [.01, .99]}


def rank_dropper_models(model_clouds, minimum_ratio=1.10):
    if not np.isfinite(minimum_ratio) or minimum_ratio <= 1:
        raise ValueError("minimum dropper size ratio must be finite and greater than 1.")
    if any(name not in model_clouds for name in DROPPER_IDS):
        raise ValueError("Dual-dropper result requires two independently reconstructed instances: dropper_a and dropper_b. Re-observe after updating the server, or explicitly use legacy count=1.")
    first, second = (np.asarray(model_clouds[n]) for n in DROPPER_IDS)
    if first.shape == second.shape and np.allclose(first, second, rtol=0, atol=1e-7):
        raise ValueError("Two dropper instances contain the same world model; refusing duplicate dosing.")
    sizes = {name: metric_model_size(model_clouds[name]) for name in DROPPER_IDS}
    small, large = sorted(DROPPER_IDS, key=lambda name: sizes[name]["volume_m3"])
    ratio = sizes[large]["volume_m3"]/sizes[small]["volume_m3"]
    if ratio < minimum_ratio:
        raise ValueError(f"Dropper physical sizes are ambiguous: large/small volume ratio {ratio:.4f} < {minimum_ratio:.4f}; no position/color/name fallback is permitted.")
    return {"method": SIZE_METHOD, "small_id": small, "large_id": large,
            "large_to_small_ratio": float(ratio), "minimum_ratio": float(minimum_ratio),
            "models": sizes,
            "roles": {"small": "dropper_1_phenolphthalein", "large": "dropper_2_acid"}}
