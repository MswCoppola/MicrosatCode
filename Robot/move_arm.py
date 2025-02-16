#!/usr/bin/env python

import rospy
from actionlib import SimpleActionClient
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

def callback(msg):
    # Log received joint states
    rospy.loginfo(msg)

def main():
    # Initialize the ROS node
    rospy.init_node('arm_control', anonymous=True)

    # Create a SimpleActionClient for the trajectory action
    arm_move = SimpleActionClient(
        '/robot/arm/scaled_pos_traj_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )

    # Wait for the action server to start
    rospy.loginfo("Waiting for the action server...")
    arm_move.wait_for_server()
    rospy.loginfo("Action server started.")

    # Subscribe to the joint states topic (using the correct message type)
    rospy.Subscriber('/robot/arm/joint_states', JointState, callback)

    # Define the trajectory goal
    mv_goal = FollowJointTrajectoryGoal()
    mv_goal.trajectory.joint_names = [
        'robot_arm_elbow_joint', # third
        'robot_arm_shoulder_lift_joint', # second
        'robot_arm_shoulder_pan_joint', # base
        'robot_arm_wrist_1_joint',
        'robot_arm_wrist_2_joint',
        'robot_arm_wrist_3_joint'
    ]

    # Create a single trajectory point
    point = JointTrajectoryPoint()
    point.positions = [1.4502070585833948, -1.8772961101927699, 0.02, -0.3, 0.22516180574893951, -1.7] # Relative positions
    point.velocities = [0.0] * 6
    point.time_from_start = rospy.Duration(7.0)

    # Add the point to the trajectory
    mv_goal.trajectory.points.append(point)

    # Send the trajectory goal
    rospy.loginfo("Sending trajectory goal...")
    arm_move.send_goal(mv_goal)

    # Wait for the result
    finished = arm_move.wait_for_result(rospy.Duration(50.0))

    if finished:
        state = arm_move.get_state()
        rospy.loginfo("Action finished with state: %s", arm_move.get_goal_status_text())
    else:
        rospy.loginfo("Action did not complete within the time limit.")

if __name__ == "__main__":
    main()