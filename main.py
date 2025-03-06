# Final main with Robot and Image Processing Integration, functionality should include (in order):

# 1. Initialize the robot arm and base
# 2. Wait for user input to start mission
# 3. Position robot arm for observation
# 4. Capture images for X seconds
# 5. Process captured images:
    # Initialize the satellite
    # Detect the rough center of the target
        # Move the camera until the center of the image roughly coincides with the center of the target

    # !!The following code should be looped until enough points are found to succesfully determine rotation axis)!!
    # Detect the corners of the current snapshot of the target
        # Pass corners along to the axis of rotation determination and the face detection

    # From this determine the axis of rotation
    # Link length of an edge in the image (pixel) space to the object (real life) size of the target to get a pixel->cm or mm conversion factor
    # Slightly move the camera to get a relative pixel movement for certain degree of rotation
        # Determine a pixel distance between the camera and the target in 3D space using trigonometry from this movement
            # Convert pixel distance to a real distance [m]
# 6. Adjust base position if target is too far
# 7. Move arm to grasp target
# 8. Mission complete notification


#!/usr/bin/env python

# Import necessary libraries
import rospy  # ROS Python client library
import actionlib  # ROS Action library for long-running tasks
import subprocess
import time
import cv2  # OpenCV for image processing
import numpy as np
import os
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from robotnik_navigation_msgs.msg import MoveAction, MoveGoal
from Robot import ur_functions as ur
from VisualDataProcessing import SatelliteDetector as SD

# Safety constants
max_safe_distance = 5  # meters
min_safe_distance = 0.3  # meters

# Declare global action clients
global arm_client, base_client

# Initialize ROS node
rospy.init_node("robot_image_integration", anonymous=True)

# Initialize action clients for robot arm and base control
arm_client = actionlib.SimpleActionClient(
    "/robot/arm/scaled_pos_traj_controller/follow_joint_trajectory",
    FollowJointTrajectoryAction
)
base_client = actionlib.SimpleActionClient(
    "/robot/move", MoveAction
)

# Wait for action servers to become available
print("Waiting for action servers...")
arm_client.wait_for_server()
base_client.wait_for_server()
print("Action servers connected.\n")

# Wait for user input to start mission
input("🔹 Press Enter to start mission...")

# Step 1: Position robot arm for observation
ur.move_arm_to_position([-1.6573207378387451, -1.275725321178772, 6.29425039733178e-06, 
                        -0.04999335229907231, 1.6000003814697266, -0.7499907652484339])

# Step 2: Capture images for X seconds
ur.capture_images(duration=10)

# Initialize satellite target with dimensions (WxDxH in mm)
current_satellite = SD.Satellite((100, 100, 500))
# Define kernel for image processing
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
it = 0
rot_det = False

# Step 3: Process captured images
imlst = []
all_point_dic = {}
all_point_lst = []
# Load all images from directory
for img in os.listdir(r"/home/robot/catkin_ws/src/microsat_group_1/src/images/"):
    imlst.append(os.path.join(r"/home/robot/catkin_ws/src/microsat_group_1/src/images/",img))

# Process each image until rotation is detected
while it < len(imlst) and rot_det is False:
    try:
        # Read and resize image
        imcol = cv2.imread(imlst[it])
        image_resized = cv2.resize(imcol, (900, 625))
    except:
        print(f"Unable to read image number {it}")
    it += 1
    
    try:
        # Detect satellite position and corners
        rel_cam_pos, rect = current_satellite.loc_on_screen(image_resized,0.10)
        process = current_satellite.current_corners(image_resized, kernel, rect)
        # Store detected corners
        all_point_dic[f"img_{it}_corners"] = process
        all_point_lst.append(np.array(process))
        # Calculate distance to target
        distance_vector = current_satellite.ranging(process)
    except:
        print("Unable to determine corners")

# Step 4: Adjust base position if target is too far (Assuming arm range is max 1 meter)
x,y,z = distance_vector
if x > max_safe_distance or x < min_safe_distance:
        raise Exception("Initial position outside safe operating range")
else:
    if x > 1:
        offset = x -1
        distance_vector -= offset
        ur.move_base(offset)

# Step 5: Move arm to grasp target
ur.move_arm_cartesian(x,y,z)

# Mission complete notification
print("\n Target successfully acquired! Mission complete.")