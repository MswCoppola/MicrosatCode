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

    def image_extractor(self, unkown):  # #######################################
        return

    def corner_grouper(self, unkown):   # #######################################
        return

    def face_saving(self, unkown):   # #######################################
        return

    def loc_on_screen(self, imcol, block, wind_sens= 0.05):  # uses a colour input image and block size for binning
        (xp, yp), (rel_x, rel_y), window = il.loc_on_screen(imcol, block, wind_sens)
        self.screen_pos = (rel_x, rel_y)
        return (rel_x, rel_y), window

    def current_corners(self, img, kernel):
        try:
            image_resized = cv2.resize(img, (900, 500))
            imcol_2 = cv2.filter2D(image_resized, -1, kernel)
            block = (100, 100)
            rel_cam_pos, rect = self.loc_on_screen(image_resized, block)

            mask, processed_image = dp.background_remover(image_resized, rect)
            maskcol = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 100

            image_processed = dp.canny_edge_detector(processed_image, 8000, 10000)
            edge_col = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2BGR)

            masked_img = cv2.bitwise_and(edge_col, edge_col, mask=mask)

            pic = dp.Corner_and_edge_outliner(maskcol, True)
            print(f"The following points are corners {pic[1]}")

            plt.plot(), plt.imshow(pic[0], cmap="gray"), plt.title('Final cornered image')
            plt.show()
            plt.plot(), plt.imshow(image_processed), plt.title('Debugging image mask')
            plt.show()
        except:
            print("Was not able to process the image")
            return

        return pic

    def __str__(self):
        print(f"Size of the satellite is {self.width}'W',{self.depth}'D'{self.heigth}'H' ")
        print(f"Position of the satellite on the screen relative to center is {self.screen_pos}")
        print(f"Rotation axis in (x,y),(x,y) screen coordinates and out of plane angle is {self.rot_ax} ")
        print(f"Current position (x,y,z) of the satellite relative to the camera is {self.rel_pos}")
        return "Current satellite parameters returned"



