#!/usr/bin/env python
 
import numpy as np
import rospy
import actionlib
from robot_kin import forward, inverse
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
 
# Global variable to store the latest joint positions
latest_joint_positions = None
 
def joint_state_callback(msg):
    """
    Callback function that updates the global variable with the latest joint positions.
    The last joint (wrist 3) is corrected by subtracting -pi/4.
    """
    global latest_joint_positions
    latest_joint_positions = np.array(msg.position)  # Extract joint positions in radians
    latest_joint_positions[-1] -= np.pi / 4  # Apply wrist 3 correction
 
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
    print("🔹 Current Joint Positions (from /robot/arm/joint_states):")
    print("  ", np.round(latest_joint_positions, 6))
    print("_" * 50 + "\n")
 
    return latest_joint_positions
 
def move_robot(command):
    """
    Moves the robot based on a given command (e.g., "up", "down", "left", "right").
    Uses the latest joint state information from ROS.
    """
    print("\n" + "=" * 50)
    print(f"📢 Movement Command Received: {command.upper()}")
    print("=" * 50 + "\n")
 
    q_current = get_current_joint_positions()
 
    T_current = np.zeros(16)
    forward(q_current, T_current)  
 
    print("_" * 50)
    print("🔹 Current Cartesian Pose (1D format):")
    print("  ", np.round(T_current, 6))
    print("_" * 50 + "\n")
 
    T_target = np.copy(T_current)
 
    if command == "up":
        T_target[11] += 0.05  
    elif command == "down":
        T_target[11] -= 0.05  
    elif command == "right":
        T_target[3] -= 0.05   
    elif command == "left":
        T_target[3] += 0.05   
    elif command == "forward":
        T_target[7] += 0.05   
    elif command == "backward":
        T_target[7] -= 0.05   
    else:
        print("❌ Invalid command. Use: 'up', 'down', 'left', 'right', 'forward', 'backward'")
        return
 
    print("_" * 50)
    print("🔹 Target Cartesian Pose (1D format) After Modification:")
    print("  ", np.round(T_target, 6))
    print("_" * 50 + "\n")
 
    q_sols = np.zeros(8 * 6)
    num_sols = inverse(T_target, q_sols, 0)  
 
    print(f"🔹 IK Solutions Found: {num_sols}")
 
    if num_sols == 0:
        print("❌ No valid IK solution found for the target pose.")
        return
 
    print("_" * 50)
    print(f"✅ Available IK Solutions:")
    for i in range(num_sols):
        solution = q_sols[i * 6:(i + 1) * 6]
        print(f"Solution {i + 1}: {np.round(solution, 6)}")
    print("_" * 50 + "\n")
 
    # Select a solution where the shoulder lift joint (joint 2) is negative
    q_target = None
    for i in range(num_sols):
        solution = q_sols[i * 6:(i + 1) * 6]
        if solution[1] < 0:  # Joint 2 (shoulder lift) must be negative
            q_target = solution
            break  
 
    if q_target is None:
        print("❌ No valid IK solution found where shoulder lift joint is negative.")
        return  
 
    print("_" * 50)
    print(f"✅ Selected Joint Configuration (Shoulder Lift Negative):")
    print("  ", np.round(q_target, 6))
    print("_" * 50 + "\n")
 
    # Comment out the following line to only print solutions without moving the robot
    # send_joint_trajectory(q_target)
 
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
 
if __name__ == "__main__":
    command = input("\n🔹 Enter movement command (up/down/left/right/forward/backward): ").strip().lower()
    move_robot(command)