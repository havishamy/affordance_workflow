import unittest

import numpy as np

from affordance.geometry import pose_to_matrix, transform_point, validate_transform


class GeometryTests(unittest.TestCase):
    def test_pose_and_point_transform(self):
        transform = pose_to_matrix([1, 2, 3, 0, 0, np.pi / 2])
        np.testing.assert_allclose(transform_point(transform, [1, 0, 0]), [1, 3, 3], atol=1e-8)

    def test_rejects_non_rigid_calibration(self):
        bad = np.eye(4)
        bad[0, 0] = 2
        with self.assertRaises(ValueError):
            validate_transform(bad, "bad")


if __name__ == "__main__":
    unittest.main()
