#!/usr/bin/env python

import rospy  # Import ROS Python library for handling nodes and topics
import cv2  # OpenCV for image processing
from cv_bridge import CvBridge, CvBridgeError  # Converts ROS images to OpenCV format
from sensor_msgs.msg import Image  # Message type for images in ROS
from std_msgs.msg import Bool  # Message type for Boolean values in ROS
import os  # Library for handling file paths and directories

# Initialize global variables
bridge = CvBridge()  # Create a CvBridge instance for image conversion
process_flag = True  # Flag to determine whether images should be processed
image_count = 1  # Counter for saving images with unique filenames
save_directory = "/home/robot/catkin_ws/src/microsat_group_1/src/images/"  # Directory to save images

# Ensure the save directory exists, create it if it does not
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def image_callback(msg):
    """ Callback function for image topic, triggered when a new image is received. """
    global process_flag, image_count  # Access global variables
    if process_flag:  # Only process the image if the flag is set to True
        try:
            rospy.loginfo("Image copy")
            # Convert the ROS image message to an OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # Generate file path for saving the image
            image_path = os.path.join(save_directory, f"target_{image_count}.png")
            
            # Save the image using OpenCV
            success = cv2.imwrite(image_path, cv_image)
            
            if success:
                rospy.loginfo(f"Image saved to {image_path}")
                image_count += 1  # Increment image count for the next image
            else:
                rospy.logerr("Failed to save image")
            
            process_flag = True  # Ensure the flag remains True after processing
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge exception: {e}")  # Log CvBridge errors
        except Exception as e:
            rospy.logerr(f"Unexpected error: {e}")  # Catch any unexpected errors

def flag_listener_callback(msg):
    """ Callback function for the flag topic, updates the process_flag value. """
    global process_flag
    process_flag = msg.data  # Update processing flag based on received message
    rospy.loginfo(f"Processing flag set to: {process_flag}")

# Initialize the ROS node named "camera"
rospy.init_node("camera", anonymous=True)

# Subscribe to the image topic to receive images from the camera
rospy.Subscriber("/galaxy_camera/galaxy_camera/image_raw", Image, image_callback)

# Subscribe to the flag topic to control image processing state
rospy.Subscriber("image_flag", Bool, flag_listener_callback)

rospy.loginfo("Image acquisition node started.")

# Keep the node running and continuously listen for messages
rospy.spin()
