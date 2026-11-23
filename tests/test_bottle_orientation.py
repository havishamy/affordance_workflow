import unittest

import numpy as np

from affordance.three_stage_workflow import _bottle_upright_rotation


class BottleOrientationTests(unittest.TestCase):
    def test_dynamic_yaw_keeps_pour_perpendicular_to_jaw_closing_axis(self):
        up = np.array([0.0, 0.0, 1.0])
        closing_tool = np.array([1.0, 0.0, 0.0])
        rotations = []
        for requested_pour in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-0.3, 0.7, 0.0]),
        ):
            rotation, pour_world, tilt_axis = _bottle_upright_rotation(
                up, requested_pour, closing_tool,
            )
            closing_world = rotation @ closing_tool
            self.assertAlmostEqual(float(closing_world @ pour_world), 0.0, places=7)
            np.testing.assert_allclose(closing_world, tilt_axis, atol=1e-7)
            np.testing.assert_allclose(rotation[:, 2], -up, atol=1e-7)
            rotations.append(rotation)
        self.assertFalse(np.allclose(rotations[0], rotations[1]))

    def test_rejects_pour_direction_parallel_to_bottle(self):
        with self.assertRaisesRegex(ValueError, "parallel to the bottle axis"):
            _bottle_upright_rotation([0, 0, 1], [0, 0, 2], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
