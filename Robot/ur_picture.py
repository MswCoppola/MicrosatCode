#!/usr/bin/env python
# takes pictures and saves them to a directory

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import os
import time  # Import time module for timestamping

# Initialize global variables
bridge = CvBridge()
process_flag = True
save_directory = "/home/robot/catkin_ws/src/microsat_group_1/src/images/"
last_capture_time = 0  # Store the timestamp of the last saved image
capture_interval = 0.2  # Time in seconds between captures

# Ensure the save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def image_callback(msg):
    """ Callback function for image topic, triggered when a new image is received. """
    global process_flag, last_capture_time  # Access global variables
    
    current_time = time.time()  # Get current time in seconds
    
    if process_flag and (current_time - last_capture_time >= capture_interval):
        try:
            rospy.loginfo("Image copy")

            # Convert the ROS image message to an OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")

            # Generate a timestamped filename
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            image_path = os.path.join(save_directory, f"target_{timestamp}.png")

            # Save the image using OpenCV
            success = cv2.imwrite(image_path, cv_image)

            if success:
                rospy.loginfo(f"Image saved to {image_path}")
                last_capture_time = current_time  # Update last capture time
            else:
                rospy.logerr("Failed to save image")
        
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge exception: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error: {e}")

def flag_listener_callback(msg):
    """ Callback function for the flag topic, updates the process_flag value. """
    global process_flag
    process_flag = msg.data
    rospy.loginfo(f"Processing flag set to: {process_flag}")

# Initialize the ROS node named "camera"
rospy.init_node("camera", anonymous=True)

# Subscribe to the image topic
rospy.Subscriber("/galaxy_camera/galaxy_camera/image_raw", Image, image_callback)

# Subscribe to the flag topic
rospy.Subscriber("image_flag", Bool, flag_listener_callback)

rospy.loginfo("Image acquisition node started.")

# Keep the node running
rospy.spin()
