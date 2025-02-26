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
import os


if __name__ == "__main__":
    current_satellite = SD.Satellite((100, 100, 500))       # This is the initialisation of our target, size WxDxH in mm
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    it = 0
    rot_det = False

    """Test code for reading images"""
    imlst = []
    all_point_dic = {}
    for img in os.listdir(r"C:\Users\massi\Downloads\OneDrive_2025-02-26\target pics"):
        imlst.append(os.path.join(r"C:\Users\massi\Downloads\OneDrive_2025-02-26\target pics",img))
    while it < len(imlst) and rot_det is False:
        try:
            imcol = cv2.imread(imlst[it])
            image_resized = cv2.resize(imcol, (900, 500))  # Resize the image to constant and processable dimensions
        except:
            print(f"Unable to read image number {it}")
        it += 1
        try:
            rel_cam_pos, rect = current_satellite.loc_on_screen(image_resized,0.10)  # Input(Resized colour image, fraction of maximum intensity which is considered !not background!)
            process = current_satellite.current_corners(image_resized, kernel, rect)  # Runs the entire corner detection, grouping etc, the output can be defined in the SatelliteDetector.py file
            all_point_dic[f"img_{it}"] = process
        except:
            print("Unable to determine corners")
    print(all_point_dic)
    #----------Ruan code, use process output as input---------------

    #imcol = cv2.imread(r"C:\Users\massi\Downloads\TryImage_block_blackEdge.jpeg")
    #image_resized = cv2.resize(imcol, (900, 500))       # Resize the image to constant and processable dimensions
    #rel_cam_pos, rect = current_satellite.loc_on_screen(image_resized, 0.10)  # Input(Resized colour image, fraction of maximum intensity which is considered !not background!)

    #process = current_satellite.current_corners(image_resized, kernel, rect)     # Runs the entire corner detection, grouping etc, the output can be defined in the SatelliteDetector.py file