#!/usr/bin/env python
# Takes pictures and saves them to a directory

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
    """
    Callback function for the image topic, triggered when a new image is received.

    - Checks if image processing is enabled (`process_flag`).
    - Ensures a minimum time interval (`capture_interval`) has passed before saving.
    - Converts the ROS image message to an OpenCV image.
    - Saves the image with a timestamped filename.

    :param msg: ROS image message containing the captured frame.
    """
    global process_flag, last_capture_time  # Access global variables for processing control

    current_time = time.time()  # Get current system time in seconds

    # Proceed if processing is enabled and the time interval has passed
    if process_flag and (current_time - last_capture_time >= capture_interval):
        try:
            rospy.loginfo("Image copy")  # Log message indicating image processing has started

            # Convert the ROS image message to an OpenCV image format
            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")

            # Generate a filename with a timestamp for unique identification
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            image_path = os.path.join(save_directory, f"target_{timestamp}.png")

            # Save the image to the specified directory
            success = cv2.imwrite(image_path, cv_image)

            if success:
                rospy.loginfo(f"Image saved to {image_path}")  # Confirm successful save
                last_capture_time = current_time  # Update last capture time for interval control
            else:
                rospy.logerr("Failed to save image")  # Log error if the save operation fails

        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge exception: {e}")  # Log errors related to image conversion
        except Exception as e:
            rospy.logerr(f"Unexpected error: {e}")  # Catch and log any other unexpected errors


def flag_listener_callback(msg):
    """
    Callback function for the flag topic, triggered when a new message is received.

    - Updates the `process_flag` variable based on the received Boolean message.
    - Logs the updated flag status to indicate whether image processing is active.

    :param msg: ROS Bool message containing the new flag value.
    """
    global process_flag  # Access the global flag variable
    process_flag = msg.data  # Update the flag based on the received message
    rospy.loginfo(f"Processing flag set to: {process_flag}")  # Log the flag status


# Initialize the ROS node named "camera"
rospy.init_node("camera", anonymous=True)

# Subscribe to the image topic to receive image data from the camera
rospy.Subscriber("/galaxy_camera/galaxy_camera/image_raw", Image, image_callback)

# Subscribe to the flag topic to receive updates on whether images should be processed
rospy.Subscriber("image_flag", Bool, flag_listener_callback)

rospy.loginfo("Image acquisition node started.")  # Log that the node has started

# Keep the node running indefinitely to continuously process incoming messages
rospy.spin()
