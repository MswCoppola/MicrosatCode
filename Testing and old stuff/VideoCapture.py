# Importing all necessary libraries
import cv2
import os
import time

# Read the video from specified path
cam = cv2.VideoCapture(r"C:\Users\massi\Downloads\VideoCap.mp4")

try:

    # creating a folder named data
    if not os.path.exists('../camcapdata'):
        os.makedirs('../camcapdata')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of camcapdata')

# frame
currentframe = 0

while (True):
    time.sleep(5) # take schreenshot every 5 seconds
    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './camcapdata/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break