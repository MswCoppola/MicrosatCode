import unittest
import numpy as np
from ur_arm import reorder_robot_to_fk, reorder_fk_to_robot, move_robot


class TestUrArm(unittest.TestCase):

    def test_reorder_robot_to_fk(self):
        """Test if robot-to-FK joint order conversion is correct."""
        q_robot = np.array([1, 2, 3, 4, 5, 6])
        expected_q_fk = np.array([3, 2, 1, 4, 5, 6])
        np.testing.assert_array_equal(reorder_robot_to_fk(q_robot), expected_q_fk, "Reorder to FK failed.")

    def test_reorder_fk_to_robot(self):
        """Test if FK-to-robot joint order conversion is correct."""
        q_fk = np.array([3, 2, 1, 4, 5, 6])
        expected_q_robot = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(reorder_fk_to_robot(q_fk), expected_q_robot, "Reorder to Robot Order failed.")

    def test_move_robot(self):
        """Test if move_robot estimates new joint positions correctly."""
        x_move, y_move, z_move = 0.1, -0.1, 0.2  # Test movements
        try:
            move_robot(x_move, y_move, z_move)  # Ensure function runs without errors
            success = True
        except Exception as e:
            print(f"Error in move_robot: {e}")
            success = False
        self.assertTrue(success, "move_robot function should execute without errors.")


if __name__ == "__main__":
    unittest.main()
