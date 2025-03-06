import unittest
import rospy
import time
import os
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from ur_picture import image_callback, flag_listener_callback


class TestUrPicture(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.bridge = CvBridge()
        self.test_image = cv2.imread("test_image.png")  # Ensure test image is available
        self.test_msg = self.bridge.cv2_to_imgmsg(self.test_image, "rgb8")
        self.save_directory = "/tmp/test_images/"
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def test_image_callback(self):
        """Test if image_callback saves an image successfully."""
        global last_capture_time
        last_capture_time = 0  # Reset timing to ensure capture
        rospy.loginfo("Running image_callback test...")
        image_callback(self.test_msg)
        time.sleep(0.3)  # Allow some time for processing
        files = os.listdir(self.save_directory)
        self.assertTrue(len(files) > 0, "Image was not saved correctly.")

    def test_flag_listener_callback(self):
        """Test if flag_listener_callback correctly updates process_flag."""
        global process_flag
        initial_flag = process_flag
        msg = Bool()
        msg.data = not initial_flag  # Toggle flag
        flag_listener_callback(msg)
        self.assertEqual(process_flag, msg.data, "Process flag was not updated correctly.")


if __name__ == "__main__":
    unittest.main()