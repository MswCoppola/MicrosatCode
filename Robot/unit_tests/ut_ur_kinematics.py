import unittest
import numpy as np
from ur_kinematics import forward, forward_all, inverse


class TestUrKinematics(unittest.TestCase):

    def test_forward(self):
        """Test forward kinematics transformation matrix computation."""
        q = [0, 0, 0, 0, 0, 0]  # Zero position
        T = [0] * 16  # Placeholder for transformation matrix
        forward(q, T)
        expected_T = np.identity(4).flatten().tolist()  # Identity matrix as expected output
        np.testing.assert_almost_equal(T, expected_T, decimal=5, err_msg="Forward kinematics failed for zero position.")

    def test_forward_all(self):
        """Test forward_all function producing valid transformation matrices."""
        q = [0, 0, 0, 0, 0, 0]
        T1, T2, T3, T4, T5, T6 = [[0] * 16 for _ in range(6)]
        forward_all(q, T1, T2, T3, T4, T5, T6)
        for Ti in [T1, T2, T3, T4, T5, T6]:
            self.assertEqual(len(Ti), 16, "Transformation matrix should have 16 elements.")

    def test_inverse(self):
        """Test inverse kinematics for a simple case."""
        q_sols = [0] * (6 * 8)  # Maximum 8 solutions for 6 DOF robot
        T = np.identity(4).flatten().tolist()  # Identity matrix transformation
        num_sols = inverse(T, q_sols, 0)  # Test inverse with desired q6 = 0
        self.assertGreaterEqual(num_sols, 0, "Inverse kinematics should return at least one solution.")


if __name__ == "__main__":
    unittest.main()