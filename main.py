# Current Main file, functionality should include (in order):

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
# Move the robot to the determined relative position in 3D space
# If touch to target is succesful stop operations



import cv2
import numpy as np
from matplotlib import pyplot as plt
from VisualDataProcessing import SatelliteDetector as SD


if __name__ == "__main__":
    current_satellite = SD.Satellite((100, 100, 500))       # This is the initialisation of our target, size WxDxH in mm
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imcol = cv2.imread(r"C:\Users\massi\Downloads\TryImage_block_blackEdge.jpeg")
    image_resized = cv2.resize(imcol, (900, 500))       # Resize the image to constant and processable dimensions
    rel_cam_pos, rect = current_satellite.loc_on_screen(image_resized, 0.1)  # Input(Resized colour image, fraction of maximum intensity which is considered !not background!)

    process = current_satellite.current_corners(image_resized, kernel, rect)     # Runs the entire corner detection, grouping etc, the output can be defined in the SatelliteDetector.py file