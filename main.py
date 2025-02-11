# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2
import numpy as np
from matplotlib import pyplot as plt
from VisualDataProcessing import SatelliteDetector as SD


if __name__ == "__main__":
    current_satellite = SD.Satellite((100, 100, 500))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    imcol = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\ModelDavinciMicrosatDahlengCam.jpeg")
    process = current_satellite.current_corners(imcol, kernel)