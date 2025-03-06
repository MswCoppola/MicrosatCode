#!/usr/bin/env python
# make this automated, and get gregs coordinates to input for the arm and base. display all the values rotation rate etc etc
# main, runs all files of image and robot control in one file.
 
import rospy
import actionlib
import time
import subprocess  # For running take_picture.py
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from robotnik_navigation_msgs.msg import MoveAction, MoveGoal
import os
import ur_arm_cartesian as arm 
 
def move_arm_to_position(position):
    """
    Moves the robot arm to a given joint position.
    :param position: List of joint angles.
    """
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
    point.positions = position
    point.time_from_start = rospy.Duration(5)  # Move in 5 seconds
    trajectory.points.append(point)
    goal.trajectory = trajectory
 
    print(f"🚀 Moving arm to position: {position}")
    arm_client.send_goal(goal)
    arm_client.wait_for_result()
    print("✅ Arm movement complete.\n")
 
def move_arm_cartesian(x_move, y_move, z_move):
    try:
        arm.move_arm(x_move, y_move, z_move)
    except ValueError:
        print("❌ Invalid input. Please provide a valid movement.")

def move_base(distance_x):
    """
    Moves the base forward or backward by a specified distance in meters.
    :param distance_x: Distance in meters (+ forward, - backward)
    """
    goal = MoveGoal()
    goal.goal.x = distance_x
    goal.goal.y = 0.0
    goal.goal.theta = 0.0
 
    print(f"🚀 Moving base by {distance_x} meters...")
    base_client.send_goal(goal)
    base_client.wait_for_result()
    print("✅ Base movement complete.\n")
 
def capture_images(duration):
    """
    Runs the take_picture.py script for a specified duration.
    """
    print(f"📷 Running take_picture.py for {duration} seconds...")
    process = subprocess.Popen(["python", "/home/robot/catkin_ws/src/microsat_group_1/src/scripts/ur_picture.py"])  # Adjust path if needed
    time.sleep(duration)  # Let the script run for the given time
    process.terminate()  # Stop the process
    print("✅ Image capture complete.\n")