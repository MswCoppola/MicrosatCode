#!/usr/bin/env python
 
import rospy
import actionlib
import time
import subprocess  # For running take_picture.py
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from robotnik_navigation_msgs.msg import MoveAction, MoveGoal
 
# ROS Action Clients
arm_client = None
base_client = None
 
def move_arm(position):
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
 
def capture_images(duration=5):
    """
    Runs the take_picture.py script for a specified duration.
    """
    print(f"📷 Running take_picture.py for {duration} seconds...")
    process = subprocess.Popen(["python", "/home/robot/catkin_ws/src/microsat_group_1/src/scripts/take_picture.py"])  # Adjust path if needed
    time.sleep(duration)  # Let the script run for the given time
    process.terminate()  # Stop the process
    print("✅ Image capture complete.\n")
 
def main():
    global arm_client, base_client
 
    rospy.init_node("full_mission_script", anonymous=True)
 
    # Initialize action clients
    arm_client = actionlib.SimpleActionClient(
        "/robot/arm/scaled_pos_traj_controller/follow_joint_trajectory",
        FollowJointTrajectoryAction
    )
    base_client = actionlib.SimpleActionClient(
        "/robot/move", MoveAction
    )
 
    print("🟢 Waiting for action servers...")
    arm_client.wait_for_server()
    base_client.wait_for_server()
    print("✅ Action servers connected.\n")
 
    input("🔹 Press Enter to start mission...")
 
    # Step 1: Move arm to initial random position
    move_arm([-1.1572515964508057, -1, -1, -0.09999676168475347, 0.8887753486633301, -0.7499550024615687])
 
    input("🔹 Press Enter to move arm to observation position...")
 
    # Step 2: Move arm to observation position
    move_arm([-1.6573207378387451, -1.275725321178772, 6.29425039733178e-06, -0.04999335229907231, 1.6000003814697266, -0.7499907652484339])
 
    input("🔹 Press Enter to start image capture...")
 
    # Step 3: Run `take_picture.py` for 5 seconds
    capture_images(duration=5)
    
 
    input("🔹 Press Enter to move the base forward...")
 
    # Step 4: Move base forward by 1.2 meters
    move_base(1.2)
 
    input("🔹 Press Enter to move arm to final position...")
 
    # Step 5: Move arm to final target position
    move_arm([-1.0573879480361938, -1.8756677112975062, -1.2222920553028871e-05, -0.04997737825427251, 1.6000158786773682, -0.749979321156637])
 
    print("\n🎯 ✅ Target successfully acquired! Mission complete.")
 
if __name__ == "__main__":
    main()
