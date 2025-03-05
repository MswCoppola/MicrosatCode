import cv2
import numpy as np
from VisualDataProcessing.BaseFunctions import Locator as il
from VisualDataProcessing import DetectionProcesses as dp
from matplotlib import pyplot as plt
from VisualDataProcessing import Corner_grouping as CornerGrouping
from VisualDataProcessing import Ellipse_Fitting as EllipseFitting


class Satellite:

    def __init__(self, size):  # size input = (width, depth, height)
        self.width, self.depth, self.heigth = size
        self.screen_pos = []                                    # for saving the relative position on the image
        self.rot_ax = []                                        # for saving the rotation axis parameters later
        self.rel_pos = []                                       # for maintaining relative positions to the robot after ranging
        self.corner_lib = {}                                    # for maintaining the position of corners at any instance

    def rotation_axis_determination(self, all_corners):   # ####################################### Can be used to link corners across multiple image
        matched_corner_lists = CornerGrouping.match_vertices_series(all_corners,2)
        print(matched_corner_lists[0])
        for i in range(0, len(matched_corner_lists[0])):

            self.corner_lib[f"corner_{i}"] = matched_corner_lists[0][i]
            plottable_array = np.array(matched_corner_lists[0][i])
            x, y = plottable_array.T
            xm, ym = np.mean(x), np.mean(y)
            std_x = np.std(x)
            std_y = np.std(y)
            for j in range(0, len(matched_corner_lists[0][i])-1):
                xj, yj = matched_corner_lists[0][i][j]
                if abs(xj - xm) >= 2*std_x or abs(yj - ym) >= 2*std_y:
                    del matched_corner_lists[0][i][j]
            plottable_array = np.array(matched_corner_lists[0][i])
            x, y = plottable_array.T
            plt.scatter(x, y)
            plt.scatter(xm, ym)
            plt.show()
        rot_ax = EllipseFitting.analyze_and_plot_ellipses(matched_corner_lists[0])
        self.rot_ax = rot_ax[0]
        return rot_ax[1]

    def ranging(self, pntl):   # ####################################### Gregs code for determining faces
        valid_quads = dp.generate_quadrilaterals(pntl)
        if len(pntl) == 7:
            cuboid_candidates = dp.identify_cuboids_from_faces(valid_quads, pntl)
            Distance = dp.Camera_Distance_Estimation(pntl, cuboid_candidates)
            return Distance
        elif len(pntl) == 6:
            desired_comb = dp.plot_shared_edge_combinations(valid_quads)
            coboid_candidates = dp.filter_final_combination(desired_comb)
            print(f"coboid_candidates are {coboid_candidates}")
            Distance = dp.Camera_Distance_Estimation_Two(pntl, coboid_candidates)
            return Distance
        else:
            return None

    def loc_on_screen(self, imcol, wind_sens= 0.05):  # uses a colour input image and block size for binning
        block = (imcol.shape[1] // 15, imcol.shape[0] // 15)  # Bin size for binning
        (xp, yp), (rel_x, rel_y), window = il.loc_on_screen(imcol, block, wind_sens)
        self.screen_pos = (rel_x, rel_y)
        return (rel_x, rel_y), window       # Outputs relative position of target on screen and the window which contains that target

    def current_corners(self, img, kernel, rect):
        # imcol_2 = cv2.filter2D(image_resized, -1, kernel)     # Currently not in use, sharpens the image (leads to noise)

        mask, processed_image = dp.background_remover(img, rect)
        image_processed = dp.canny_edge_detector(processed_image, 200, 1000, False)      # Input=(colour image, corner sensitivity 1, corner sensitivity 2)
        print(image_processed)
        edge_col = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2BGR)

        masked_img = cv2.bitwise_and(edge_col, edge_col, mask=mask)     # Subtracts the edges from the image to create clearly separated faces

        pic = dp.Corner_and_edge_outliner(masked_img, True)     # Determines and outlines the corners
        print(pic)

        grpd_corn = dp.filter_close_points(pic[2], 15)      # Input=(array of points, max pixel distance to average 2 or more points)
        extr = mask.copy()
        extr_img = img.copy()

        for i in grpd_corn:         # Circles the corners in the image (visual assistance only)
            cv2.circle(extr_img, i, 4, (255, 255, 255), -1)
        #except:
        #    print("Was not able to process the image")
        #    return
        return grpd_corn      # Adjust this return to return the necessary outputs from each function

    def __str__(self):      # If you use print(satellite) this will be the printed statement
        print(f"Size of the satellite is {self.width}'W',{self.depth}'D'{self.heigth}'H' ")
        print(f"Position of the satellite on the screen relative to center is {self.screen_pos}")
        print(f"Rotation axis in (x,y),(x,y) screen coordinates and out of plane angle is {self.rot_ax} ")
        print(f"Current position (x,y,z) of the satellite relative to the camera is {self.rel_pos}")
        return "Current satellite parameters returned"




