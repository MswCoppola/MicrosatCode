import cv2
import numpy as np
from VisualDataProcessing import DetectionProcesses as dp
from VisualDataProcessing.BaseFunctions import Locator as il
from matplotlib import pyplot as plt


class Satellite:

    def __init__(self, size):  # size input = (width, depth, height)
        self.width, self.depth, self.heigth = size
        self.screen_pos = []                                    # for saving the relative position on the image
        self.rot_ax = []                                        # for saving the rotation axis parameters later
        self.rel_pos = []                                       # for maintaining relative positions to the robot after ranging
        self.corner_lib = {}                                    # for maintaining the position of corners at any instance

    def image_extractor(self, unkown):  # ####################################### Can be used to take a snapshot
        return

    def corner_grouper(self, unkown):   # ####################################### Can be used to link corners across multiple image
        return

    def face_saving(self, unkown):   # ####################################### Gregs code for determining faces
        return

    def loc_on_screen(self, imcol, wind_sens= 0.05):  # uses a colour input image and block size for binning
        block = (imcol.shape[1] // 10, imcol.shape[0] // 10)  # Bin size for binning
        (xp, yp), (rel_x, rel_y), window = il.loc_on_screen(imcol, block, wind_sens)
        self.screen_pos = (rel_x, rel_y)
        return (rel_x, rel_y), window       # Outputs relative position of target on screen and the window which contains that target

    def current_corners(self, img, kernel, rect):
        try:
            # imcol_2 = cv2.filter2D(image_resized, -1, kernel)     # Currently not in use, sharpens the image (leads to noise)

            mask, processed_image = dp.background_remover(img, rect)
            image_processed = dp.canny_edge_detector(processed_image, 10000, 80000, False)      # Input=(colour image, corner sensitivity 1, corner sensitivity 2)
            edge_col = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2BGR)
            masked_img = cv2.bitwise_and(edge_col, edge_col, mask=mask)     # Subtracts the edges from the image to create clearly separated faces
            pic = dp.Corner_and_edge_outliner(masked_img, True)     # Determines and outlines the corners

            grpd_corn = dp.filter_close_points(pic[2], 10)      # Input=(array of points, max pixel distance to average 2 or more points)
            extr = mask.copy()

            for i in grpd_corn:         # Circles the corners in the image (visual assistance only)
                cv2.circle(extr, i, 4, (100, 100, 100), -1)


            plt.plot(), plt.imshow(extr, cmap="gray"), plt.title('Final cornered image')
            plt.show()
            plt.plot(), plt.imshow(image_processed), plt.title('Debugging image mask')
            plt.show()
        except:
            print("Was not able to process the image")
            return

        return pic      # Adjust this return to return the necessary outputs from each function

    def __str__(self):      # If you use print(satellite) this will be the printed statement
        print(f"Size of the satellite is {self.width}'W',{self.depth}'D'{self.heigth}'H' ")
        print(f"Position of the satellite on the screen relative to center is {self.screen_pos}")
        print(f"Rotation axis in (x,y),(x,y) screen coordinates and out of plane angle is {self.rot_ax} ")
        print(f"Current position (x,y,z) of the satellite relative to the camera is {self.rel_pos}")
        return "Current satellite parameters returned"

#Geometric_Three_Points----------
#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#from itertools import combinations
#import math
#import subprocess

# Example corner points------------------
#corner_points = [(0,0), (2.123,2.123), (1.5089479078638484,2.7336927792554375), (-0.6123724356957946,0.6123724356957946), (0.000000,-0.500000), (2.121320,1.623), (-0.6123724356957946,0.112372)]
#corner_points = [(0, 0), (1.5, 0), (0.333, -0.5),(1.11111, -0.5),(0.333, 3),(1.1111, 3)]


# Generate and plot quadrilaterals
valid_quads = generate_quadrilaterals(corner_points)
if valid_quads:
    print(f"Number of valid quadrilaterals found: {len(valid_quads)}")
    #plot_quadrilaterals(valid_quads, corner_points)
else:
    print("No valid quadrilaterals were found.")

#Cuboid_Detection-----------------------

#import matplotlib.pyplot as plt
#from itertools import combinations
#from shapely.geometry import Polygon, Point, LineString
#from Geometric_Three_Points import generate_quadrilaterals, corner_points as all_corners


# Use the quadrilaterals from Geometric_Three_Points
valid_quads = generate_quadrilaterals(all_corners)
print(f"✅ Generated {len(valid_quads)} valid quadrilaterals from Geometric_Three_Points.")

# Identify cuboid candidates
cuboid_candidates = identify_cuboids_from_faces(valid_quads, all_corners)
print(f"✅ Identified {len(cuboid_candidates)} possible cuboids from Cuboid_Detection.")

# Plot the identified cuboid candidates
if cuboid_candidates:
    plot_cuboid_candidates(cuboid_candidates)
else:
    print("❌ No valid cuboids found.")

#Face_Determination-----------------------

#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#import numpy as np
#from Cuboid_Detection import cuboid_candidates  # Import detected cuboid candidates


if cuboid_candidates:
    for cuboid in cuboid_candidates:
        classified_faces = classify_faces(cuboid["faces"])
        print("Face areas and aspect ratios:")
        for face_label, face in classified_faces.items():
            print(f"{face_label}: Area = {compute_face_area(face):.2f}, Aspect Ratio = {compute_aspect_ratio(face):.2f}")
        plot_classified_faces(classified_faces)
else:
    print("No valid cuboid detected.")

#Camera_Distance_Estimation------------------

#import numpy as np
#from shapely.geometry import LineString
#from Geometric_Three_Points import corner_points
#from Cuboid_Detection import cuboid_candidates
#import matplotlib.pyplot as plt
#import os


if __name__ == "__main__":
    main()

#Cuboid_Detection_Two-----------------------

#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon, LineString, Point
#from itertools import combinations
#from Geometric_Three_Points import generate_quadrilaterals, corner_points
#import numpy as np


valid_quads = generate_quadrilaterals(corner_points)
print(f"Loaded {len(valid_quads)} valid quadrilaterals from Geometric_Three_Points.")

# Step 1: Plot initial combinations
desired_combinations = plot_shared_edge_combinations(valid_quads)

# Step 2: Apply final filtering to select the best cuboid representation and save the plot
filter_final_combination(desired_combinations)

# Step 3: Show the saved final two-face cuboid plot
show_saved_two_face_cuboid()

# -------------------- End of Script --------------------

# Functions now comprehensively check for global edge overlaps and face intersections.
# Enhanced logic incorporates disjoint face validation and stricter edge uniqueness checks.
# The final two-face cuboid plot is saved and can be recalled using show_saved_two_face_cuboid().

#Face_Determination_Two----------------------------------

#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#import numpy as np
#from Geometric_Three_Points import generate_quadrilaterals, corner_points
#from Cuboid_Detection_Two import saved_final_two_face_cuboid  # Import saved plot from cuboid_determination_two

if saved_final_two_face_cuboid:
    extracted_faces = extract_faces_from_saved_plot(saved_final_two_face_cuboid)

    if not extracted_faces:
        print("No valid faces extracted from the saved cuboid figure. Trying quadrilaterals from Geometric_Three_Points.")
        extracted_faces = generate_quadrilaterals(corner_points)

    if extracted_faces:
        classified_faces = classify_faces(extracted_faces)
        if classified_faces:
            print("Face areas and aspect ratios:")
            for face_label, face in classified_faces.items():
                print(f"{face_label}: Area = {compute_face_area(face):.2f}, Aspect Ratio = {compute_aspect_ratio(face):.2f}")
            plot_classified_faces(classified_faces)
        else:
            print("Classification incomplete due to insufficient faces.")
    else:
        print("No valid quadrilateral faces available for classification.")
else:
    print("No saved two-face cuboid plot available to process.")

#Camera_Distance_Estimation_Two----------------------------

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from shapely.geometry import Polygon
#import numpy as np
#from Geometric_Three_Points import corner_points
#from Cuboid_Detection_Two import saved_final_two_face_cuboid  # Import saved plot from cuboid_determination_two
#from Face_Determination_Two import classified_faces  # Import face identification from face_determination_two
#import os

if __name__ == "__main__":
    main()


