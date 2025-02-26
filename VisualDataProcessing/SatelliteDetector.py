import cv2
from VisualDataProcessing.BaseFunctions import Locator as il, DetectionProcesses as dp
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
        valid_quads = dp.generate_quadrilaterals(unkown)
        cuboid_candidates = dp.identify_cuboids_from_faces(valid_quads, unkown)
        return cuboid_candidates

    def loc_on_screen(self, imcol, wind_sens= 0.05):  # uses a colour input image and block size for binning
        block = (imcol.shape[1] // 15, imcol.shape[0] // 15)  # Bin size for binning
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
            extr_img = img.copy()

            for i in grpd_corn:         # Circles the corners in the image (visual assistance only)
                cv2.circle(extr_img, i, 4, (255, 255, 255), -1)
            plt.plot(), plt.imshow(extr_img), plt.title('Debugging image mask')
            plt.show()
        except:
            print("Was not able to process the image")
            return

        return grpd_corn      # Adjust this return to return the necessary outputs from each function

    def __str__(self):      # If you use print(satellite) this will be the printed statement
        print(f"Size of the satellite is {self.width}'W',{self.depth}'D'{self.heigth}'H' ")
        print(f"Position of the satellite on the screen relative to center is {self.screen_pos}")
        print(f"Rotation axis in (x,y),(x,y) screen coordinates and out of plane angle is {self.rot_ax} ")
        print(f"Current position (x,y,z) of the satellite relative to the camera is {self.rel_pos}")
        return "Current satellite parameters returned"



