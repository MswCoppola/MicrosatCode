#!/usr/bin/env python

import rospy
import actionlib
from robotnik_navigation_msgs.msg import MoveAction, MoveGoal
from geometry_msgs.msg import Twist

def move_base():
    # Initialize the ROS node
    rospy.init_node("base_control")

    # Create an Action Client for sending movement goals
    client = actionlib.SimpleActionClient("/robot/move", MoveAction)

    rospy.loginfo("Waiting for the action server to start...")
    client.wait_for_server()
    rospy.loginfo("Action server started.")

    manuver_time = 7  # Time in seconds to complete the movement
    way_points = 1  # Number of movement cycles
    step = 1  # Initial movement step

    for _ in range(2 * way_points):  # Loop through movement steps
        goal = MoveGoal()

        if step == 1:
            # SET YOUR DESIRED GOAL POSITION HERE (Step 1)
            goal.goal.x = 0.0  # Move forward/backward (+/- X direction)
            goal.goal.y = -0.5  # Move left/right (+/- Y direction)
            goal.goal.theta = 0  # Rotation in radians (negative --> clockwise)

            # Set the robot's maximum velocity for smoother motion
            goal.maximum_velocity.linear.x = goal.goal.x / manuver_time
            goal.maximum_velocity.linear.y = goal.goal.y / manuver_time
            goal.maximum_velocity.linear.z = 0
            goal.maximum_velocity.angular.x = 0
            goal.maximum_velocity.angular.y = 0
            goal.maximum_velocity.angular.z = goal.goal.theta / manuver_time

            rospy.loginfo("Sending goal: Step 1 -> Moving robot")
            client.send_goal(goal)  # Send movement command to the robot
            finished = client.wait_for_result(rospy.Duration(manuver_time + 3))  # Wait for completion

            if finished:
                rospy.loginfo("Action finished: %s", client.get_state())
                step = 2  # Switch to next movement step
            else:
                rospy.logwarn("Action not finished, retrying Step 2")
                step = 2  # Move to Step 2

        elif step == 2:
            # SET YOUR DESIRED GOAL POSITION HERE (Step 2)
            goal.goal.x = 0  # Move forward/backward (+/- X direction)
            goal.goal.y = 0  # Move left/right (+/- Y direction)
            goal.goal.theta = 0.0  # Rotation in radians

            # Set the robot's maximum velocity
            goal.maximum_velocity.linear.x = goal.goal.x / manuver_time
            goal.maximum_velocity.linear.y = goal.goal.y / manuver_time
            goal.maximum_velocity.linear.z = 0
            goal.maximum_velocity.angular.x = 0
            goal.maximum_velocity.angular.y = 0
            goal.maximum_velocity.angular.z = goal.goal.theta / manuver_time

            rospy.loginfo("Sending goal: Step 2 -> Moving robot")
            client.send_goal(goal)  # Send movement command to the robot
            finished = client.wait_for_result(rospy.Duration(manuver_time + 3))  # Wait for completion

            if finished:
                rospy.loginfo("Action finished: %s", client.get_state())
                step = 1  # Switch back to Step 1
            else:
                rospy.logwarn("Action not finished, retrying Step 1")
                step = 1  # Move to Step 1

move_base()
