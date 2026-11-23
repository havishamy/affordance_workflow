import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from affordance.alignment_export import _stage_json, depth_to_world, write_colored_glb, write_colored_ply
from affordance.three_stage_result import _read_ply_xyz


class AlignmentExportTests(unittest.TestCase):
    def test_depth_projection_and_world_transform(self):
        depth = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)
        intrinsics = {"width": 2, "height": 2, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
        transform = np.eye(4)
        transform[:3, 3] = [10.0, 20.0, 30.0]
        points = depth_to_world(depth, intrinsics, 0.001, transform, max_depth_m=2.1)
        np.testing.assert_allclose(points, [
            [10.0, 20.0, 31.0],
            [10.0, 22.0, 32.0],
            [11.0, 21.0, 31.0],
        ])

    def test_stage_json_extracts_calibration(self):
        log = 'prefix\n[STAGE] calibration_validation\n{"valid": true, "T": [[1]]}\nnext'
        self.assertEqual(_stage_json(log, "calibration_validation"), {"valid": True, "T": [[1]]})

    def test_binary_colored_ply_is_readable_by_existing_reader(self):
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        sources = np.array([0, 1, 2], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "combined.ply"
            write_colored_ply(path, points, colors, sources, ["test"])
            np.testing.assert_allclose(_read_ply_xyz(path), points)
            header = path.read_bytes().split(b"end_header\n", 1)[0]
            self.assertIn(b"property uchar source_id", header)

    def test_glb_contains_point_primitives_and_vertex_colors(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "combined.glb"
            write_colored_glb(path, [
                ("depth", np.array([[0, 0, 0], [1, 1, 1]], dtype=float), (170, 170, 170)),
                ("model", np.array([[2, 2, 2]], dtype=float), (255, 0, 0)),
            ])
            raw = path.read_bytes()
            magic, version, total_length = struct.unpack_from("<III", raw)
            self.assertEqual(magic, 0x46546C67)
            self.assertEqual(version, 2)
            self.assertEqual(total_length, len(raw))
            json_length, json_type = struct.unpack_from("<II", raw, 12)
            self.assertEqual(json_type, 0x4E4F534A)
            document = json.loads(raw[20:20 + json_length].decode("utf-8"))
            self.assertEqual(len(document["meshes"][0]["primitives"]), 2)
            self.assertEqual(document["meshes"][0]["primitives"][0]["mode"], 0)
            self.assertIn("COLOR_0", document["meshes"][0]["primitives"][0]["attributes"])


if __name__ == "__main__":
    unittest.main()
