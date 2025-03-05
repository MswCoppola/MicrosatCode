#!/usr/bin/env python
# Originally implement_IK.py, now ur_arm.py
# This script estimates joint positions and moves the robot arm in Cartesian space
# using forward kinematics to determine new joint angles.

import numpy as np
import rospy
import actionlib
from ur_arm import forward_all
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState

# Global variable to store the latest joint positions received from the robot
latest_joint_positions = None

def joint_state_callback(msg):
    """
    Callback function for the /robot/arm/joint_states topic.
    Updates the global variable with the latest joint positions.
    """
    global latest_joint_positions
    latest_joint_positions = np.array(msg.position)

def get_current_joint_positions():
    """
    Retrieves the latest joint positions from the robot.
    Waits for the first message from the /robot/arm/joint_states topic before proceeding.
    """
    global latest_joint_positions

    if not rospy.get_node_uri():
        rospy.init_node('joint_state_listener', anonymous=True)

    rospy.Subscriber("/robot/arm/joint_states", JointState, joint_state_callback)

    while latest_joint_positions is None:
        rospy.sleep(0.1)  # Wait until joint positions are received

    print("\n" + "_" * 50)
    print("🔹 Current Joint Positions (from /robot/arm/joint_states) in Robot Order:")
    print("  ", np.round(reorder_robot_to_fk(latest_joint_positions), 6))
    print("_" * 50 + "\n")

    return latest_joint_positions

def move_robot(x_move, y_move, z_move):
    """
    Moves the robot’s wrist in Cartesian space while maintaining orientation.
    - X direction: forward/backward movement (elbow)
    - Y direction: left/right movement (shoulder pan)
    - Z direction: up/down movement (shoulder lift)
    """
    print("\n" + "=" * 50)
    print(f"📢 Movement Command Received: Move ({x_move} m, {y_move} m, {z_move} m)")
    print("=" * 50 + "\n")

    # Get current joint positions in the robot's order
    q_current_robot_order = get_current_joint_positions()
    q_current = reorder_robot_to_fk(q_current_robot_order)

    # Compute forward kinematics to determine the current end-effector pose
    T1, T2, T3, T4, T5, T6 = [np.zeros(16) for _ in range(6)]
    forward_all(q_current, T1, T2, T3, T4, T5, T6)

    # Compute full transformation matrix
    T1, T2, T3, T4, T5, T6 = [T.reshape(4, 4) for T in [T1, T2, T3, T4, T5, T6]]
    T_current_matrix = np.dot(np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5), T6)

    print("_" * 50)
    print("🔹 Current Transformation Matrix (T_current):")
    print(T_current_matrix)
    print("_" * 50 + "\n")

    # Modify Cartesian position (X, Y, Z) while keeping orientation unchanged
    T_target_matrix = np.copy(T_current_matrix)
    T_target_matrix[0, 3] += x_move  # Adjust X position
    T_target_matrix[1, 3] += y_move  # Adjust Y position
    T_target_matrix[2, 3] += z_move  # Adjust Z position

    print("_" * 50)
    print("🔹 Target Transformation Matrix (T_target) After Modification:")
    print(T_target_matrix)
    print("_" * 50 + "\n")

    # Estimate new joint positions (without full inverse kinematics)
    q_target = np.copy(q_current)

    # Adjust joints based on movement direction
    if abs(z_move) > 0:
        q_target[1] += z_move  # Shoulder lift for vertical movement

    if abs(x_move) > 0:
        q_target[2] += x_move  # Elbow for forward/backward movement

    if abs(y_move) > 0:
        q_target[0] += y_move  # Shoulder pan for left/right movement

    print("_" * 50)
    print(f"✅ Estimated New Joint Configuration (FK Order, Adjusted for Movement):")
    print("  ", np.round(q_target, 6))
    print("_" * 50 + "\n")

    # Convert joint angles back to the robot's order
    q_target_robot_order = reorder_fk_to_robot(q_target)

    print("_" * 50)
    print(f"✅ Final Joint Configuration (Robot Order, Ready to Send):")
    print("  ", np.round(q_target_robot_order, 6))
    print("_" * 50 + "\n")

    # Send the new joint positions to the robot for execution
    send_joint_trajectory(q_target_robot_order)

def send_joint_trajectory(q_solution):
    """
    Sends a trajectory command to the robot arm using ROS action client.
    Moves the arm to the specified joint positions.
    :param q_solution: A list of 6 joint angles representing a valid solution.
    """
    client = actionlib.SimpleActionClient(
        "/robot/arm/scaled_pos_traj_controller/follow_joint_trajectory",
        FollowJointTrajectoryAction
    )
    client.wait_for_server()  # Ensure the action server is available

    goal = FollowJointTrajectoryGoal()
    trajectory = JointTrajectory()
    trajectory.joint_names = [
        "robot_arm_elbow_joint",
        "robot_arm_shoulder_lift_joint",
        "robot_arm_shoulder_pan_joint",
        "robot_arm_wrist_1_joint",
        "robot_arm_wrist_2_joint",
        "robot_arm_wrist_3_joint"
    ]

    point = JointTrajectoryPoint()
    point.positions = q_solution  # Assign target joint positions
    point.time_from_start = rospy.Duration(5)  # Set movement duration

    trajectory.points.append(point)
    goal.trajectory = trajectory

    client.send_goal(goal)  # Send command
    client.wait_for_result()  # Wait for execution completion
    rospy.loginfo("✅ Robot arm moved to the target pose.")

def reorder_robot_to_fk(q):
    """
    Converts joint order from the robot's convention to the forward kinematics convention.
    :param q: Array of joint positions in the robot’s order.
    :return: Joint positions reordered for forward kinematics calculations.
    """
    return np.array([q[2], q[1], q[0], q[3], q[4], q[5]])

def reorder_fk_to_robot(q):
    """
    Converts joint order from forward kinematics convention back to the robot's convention.
    :param q: Array of joint positions in FK order.
    :return: Joint positions reordered for execution on the robot.
    """
    return np.array([q[2], q[1], q[0], q[3], q[4], q[5]])

if __name__ == "__main__":
    """
    Main execution block: Prompts user for movement commands and moves the robot accordingly.
    """
    try:
        x_move, y_move, z_move = map(float, input("\n🔹 Enter movement in meters (X Y Z): ").split())
        move_robot(x_move, y_move, z_move)
    except ValueError:
        print("❌ Invalid input. Please enter three numbers for X, Y, and Z movement (e.g., '0.05 0.10 0').")
