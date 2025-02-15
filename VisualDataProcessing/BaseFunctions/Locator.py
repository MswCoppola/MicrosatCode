import numpy as np
import matplotlib.pyplot as plt
import cv2

def loc_on_screen(imcol, block, wind_sens):  # uses a colour input image, block size for binning and a intensity sensitivity for windowing
    y, x, z = imcol.shape
    print(y,x,z)
    amt_x = x // block[0]  # define blocks for binning image intensities
    amt_y = y // block[1]
    int_img = np.zeros((y, x))
    for i in range(0, amt_y):       # Loop through all the bins collecting all the light intensities to determine target position on image
        for j in range(0, amt_x):
            strt_x, nd_x = i * block[1], (i + 1) * block[1]
            strt_y, nd_y = j * block[0], (j + 1) * block[0]
            intensity = np.mean(imcol[strt_x:nd_x, strt_y:nd_y])  # bin image intensity into blocks for faster processing
            int_img[strt_x:nd_x, strt_y:nd_y] = intensity
    int_high = np.where(int_img >= 0.8 * int_img.max())     # Used for determining target location
    int_med = np.where(int_img >= wind_sens * int_img.max())        # Used to determine target possible window for the background remover
    xp, yp = int(np.mean(int_high[1]))+1, int(np.mean(int_high[0]))+1
    cv2.circle(int_img, (xp, yp), 10, (255, 255, 255), -1)      # Circle the rough center on the image
    rel_x, rel_y = xp - x // 2, yp - y // 2  # detect the rough center of the satellite relative to image size

    plt.plot(), plt.imshow(int_img, cmap="gray"), plt.title('Debugging image for the loc_on_screen Function')
    plt.show()

    x_len = np.max(int_med[1]) - np.min(int_med[1]) + 1     # Window size calculation
    y_len = np.max(int_med[0]) - np.min(int_med[0]) + 1
    window = (np.min(int_med[1]), np.min(int_med[0]), x_len, y_len)  # window for the background removal process
    print(f"window is {window}")
    return (xp, yp), (rel_x, rel_y), window     # Output= (absolute target center on image, target center relative to image center, window which is considered not background)

def exact_center(imcol): # Input should be a masked image of the satellite without background
    return

def centering_operation():
    return  # Should return a remaining difference between the exact center of the image and the center of the

def scanning_operation():
    return  # Should not return anything, the output is in ROS operations turning the camera until a high intensity is located