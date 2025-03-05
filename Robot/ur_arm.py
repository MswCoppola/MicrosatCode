#!/usr/bin/env python
# use to be implement_IK.py, now ur_arm.py to to estimate joint positions and move the robot arm in Cartesian space.
# This script uses the forward kinematics to estimate new joint positions for the robot arm and then moves the arm to the new position.


import numpy as np
import rospy
import actionlib
from ur_arm import forward_all
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
 
# Global variable to store the latest joint positions
latest_joint_positions = None
 
def joint_state_callback(msg):
    """
    Callback function that updates the global variable with the latest joint positions.
    """
    global latest_joint_positions
    latest_joint_positions = np.array(msg.position)
 
def get_current_joint_positions():
    """
    Returns the most recent joint positions from the /robot/arm/joint_states topic.
    Waits until the first message is received.
    """
    global latest_joint_positions
 
    if not rospy.get_node_uri():
        rospy.init_node('joint_state_listener', anonymous=True)
 
    rospy.Subscriber("/robot/arm/joint_states", JointState, joint_state_callback)
 
    while latest_joint_positions is None:
        rospy.sleep(0.1)
 
    print("\n" + "_" * 50)
    print("🔹 Current Joint Positions (from /robot/arm/joint_states) in Robot Order:")
    print("  ", np.round(reorder_robot_to_fk(latest_joint_positions), 6))
    print("_" * 50 + "\n")
 
    return latest_joint_positions
 
def move_robot(x_move, y_move, z_move):
    """
    Moves the robot’s wrist 3 in Cartesian space while keeping orientation fixed.
    Uses shoulder pan for left/right, shoulder lift for up/down, and elbow for forward/backward.
    """
    print("\n" + "=" * 50)
    print(f"📢 Movement Command Received: Move ({x_move} m, {y_move} m, {z_move} m)")
    print("=" * 50 + "\n")
 
    # Get current joint positions
    q_current_robot_order = get_current_joint_positions()
    q_current = reorder_robot_to_fk(q_current_robot_order)
 
    # Compute the forward kinematics to get the current Cartesian pose
    T1, T2, T3, T4, T5, T6 = [np.zeros(16) for _ in range(6)]
    forward_all(q_current, T1, T2, T3, T4, T5, T6)
 
    # Compute full transformation matrix
    T1, T2, T3, T4, T5, T6 = [T.reshape(4, 4) for T in [T1, T2, T3, T4, T5, T6]]
    T_current_matrix = np.dot(np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5), T6)
 
    print("_" * 50)
    print("🔹 Current Transformation Matrix (T_current):")
    print(T_current_matrix)
    print("_" * 50 + "\n")
 
    # Modify Cartesian position while keeping orientation the same
    T_target_matrix = np.copy(T_current_matrix)
    T_target_matrix[0, 3] += x_move  # Move in X direction (forward/backward)
    T_target_matrix[1, 3] += y_move  # Move in Y direction (left/right)
    T_target_matrix[2, 3] += z_move  # Move in Z direction (up/down)
 
    print("_" * 50)
    print("🔹 Target Transformation Matrix (T_target) After Modification:")
    print(T_target_matrix)
    print("_" * 50 + "\n")
 
    # **Estimate New Joint Positions Without Full IK**
    q_target = np.copy(q_current)
 
    # Adjust **shoulder lift** for vertical (Z) movement
    if abs(z_move) > 0:
        q_target[1] += z_move * 1  # Shoulder lift adjusts up/down (negative is down) (forward - and back +)
 
    # Adjust **elbow** for forward/backward (X) movement
    if abs(x_move) > 0:
        q_target[2] += x_move * 1  # Elbow extends/retracts (forward is positive)
 
    # Adjust **shoulder pan** for left/right (Y) movement
    if abs(y_move) > 0:
        q_target[0] += y_move * 1  # Shoulder pan rotates (positive is right)
 
    print("_" * 50)
    print(f"✅ Estimated New Joint Configuration (FK Order, Adjusted for Movement):")
    print("  ", np.round(q_target, 6))
    print("_" * 50 + "\n")
 
    # Convert back to Robot Order
    q_target_robot_order = reorder_fk_to_robot(q_target)
 
    print("_" * 50)
    print(f"✅ Final Joint Configuration (Robot Order, Ready to Send):")
    print("  ", np.round(q_target_robot_order, 6))
    print("_" * 50 + "\n")
 
    # Send the new joint positions to the robot
    send_joint_trajectory(q_target_robot_order)
 
def send_joint_trajectory(q_solution):
    """
    Sends a joint trajectory command to move the robot arm using a ROS action client.
    :param q_solution: A list of 6 joint angles (one valid IK solution)
    """
    client = actionlib.SimpleActionClient(
        "/robot/arm/scaled_pos_traj_controller/follow_joint_trajectory",
        FollowJointTrajectoryAction
    )
    client.wait_for_server()
 
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
    point.positions = q_solution  
    point.time_from_start = rospy.Duration(5)  
 
    trajectory.points.append(point)
    goal.trajectory = trajectory
 
    client.send_goal(goal)
    client.wait_for_result()
    rospy.loginfo("✅ Robot arm moved to the target pose.")
 
def reorder_robot_to_fk(q):
    """Convert from (elbow, shoulder lift, shoulder pan, wrist 1, wrist 2, wrist 3)
       to (shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)"""
    return np.array([q[2], q[1], q[0], q[3], q[4], q[5]])
 
def reorder_fk_to_robot(q):
    """Convert from (shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)
       to (elbow, shoulder lift, shoulder pan, wrist 1, wrist 2, wrist 3)"""
    return np.array([q[2], q[1], q[0], q[3], q[4], q[5]])
 
if __name__ == "__main__":
    try:
        x_move, y_move, z_move = map(float, input("\n🔹 Enter movement in meters (X Y Z): ").split())
        move_robot(x_move, y_move, z_move)
    except ValueError:
        print("❌ Invalid input. Please enter three numbers for X, Y, and Z movement (e.g., '0.05 0.10 0').")