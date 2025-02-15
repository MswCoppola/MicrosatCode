import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the list of all files and directories
path = r"/camcapdata"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)
nb = 0
for i in dir_list:
    nb += 1
    img = cv2.imread(path + f"\{i}")
    imsl = img[250:400, 100:350]
    name = f"SLFrame{nb}.jpg"
    cv2.imwrite(name, imsl)
    plt.plot(), plt.imshow(imsl, cmap="gray"), plt.title(f'Imgray after contour')
    plt.show()
