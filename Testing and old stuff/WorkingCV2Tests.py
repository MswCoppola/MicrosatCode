# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math



def All_non_zero(gray_image, threshold, replacement, hightresh=False):
    out = gray_image.copy()
    # YOUR CODE HERE
    for y in range(gray_image.shape[0]):
        for x in range(gray_image.shape[1]):
            if gray_image[y, x] <= threshold:
                out[y, x] = replacement
            elif hightresh is True:
                print("f")
                out[y, x] = 200
            else:
                out[y, x] = gray_image[y, x]
    return out

def process(img, can1, can2, double =False):
    plt.plot(), plt.imshow(img), plt.title('image for mapping')
    plt.show()

    img_canny = cv.Canny(img, can1, can2, np.array(img.shape), 7)
    kernel = np.ones((3, 3))
    img_dilate = cv.dilate(img_canny, kernel, iterations=1)

    if double is True:
        img_canny2 = cv.Canny(img_dilate, can1, can2, np.array(img.shape), 5)
        corners = cv.cornerHarris(img_canny2, 4, 7, 0.02)
        plt.plot(), plt.imshow(img_canny), plt.title('canny image')
        plt.show()
    else:
        corners = cv.cornerHarris(img_dilate,4,7,0.02)
        plt.plot(), plt.imshow(img_canny), plt.title('canny image')
        plt.show()


    plt.plot(), plt.imshow(corners), plt.title('corner weights')
    plt.show()
    b = np.argwhere(corners >= 0.8*corners.max())

    #Used for debugging
    """
    print(f"corners = {corners}")
    print(f"corner shape = {corners.shape}")
    print(f"b = {b}")
    print(f"max corner = {corners[b]}")
    """

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = (imgray//2  + img_dilate*100)
    return out

def backremover(img, rect, thrs= False):
    assert img is not None, "file could not be read, check with os.path.exists()"
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(imgray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    if thrs is True:
        ret, thresh = cv2.threshold(imgray, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imguse = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv.grabCut(imguse, mask, rect, bgdModel, fgdModel, 7, cv.GC_INIT_WITH_RECT)
    else:
        cv.grabCut(img,mask,rect,bgdModel,fgdModel, 7,cv.GC_INIT_WITH_RECT)
    print("still functioning")
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_new = img*mask2[:,:,np.newaxis]

    # internal debugging
    #plt.plot(), plt.imshow(thresh, cmap="gray"), plt.title('Internal Debugging image')
    #plt.show()

    return mask2, img_new

#img = cv.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\Sat_80.png")
#image_resized = cv.resize(img, (900, 500))
#process_image = image_resized
#equ = cv.equalizeHist(cv.cvtColor(process_image, cv.COLOR_BGR2GRAY))
#equ_col = cv.cvtColor(equ, cv.COLOR_GRAY2BGR)
#plt.plot(), plt.imshow(equ, cmap="gray"), plt.title('contrasted image')
#plt.show()

#rect = (process_image.shape[1]//3, 1, process_image.shape[1]//3, process_image.shape[0]-2)
#mask, processed_image = backremover(process_image, rect)
#plt.plot(), plt.imshow(mask, cmap="gray"), plt.title('Mask')
#plt.show()

#masked_img = cv.bitwise_and(equ_col,equ_col,mask = mask)

#plt.plot(), plt.imshow(masked_img, cmap="gray"), plt.title('Masked image')
#plt.show()

def Outliner(imcol):

    #convert image to gray image
    imgray = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)//2

    #threshold the image and draw the contours
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #save the largest contours (size by area)
    cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 200:
            cnts.append(cnt)

            #debugging
            #print(area)

    contoured = imgray
    plt.plot(), plt.imshow(thresh, cmap="gray"), plt.title('Threshold')
    plt.show()
    plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Imgray before contour')
    plt.show()

    for i in range(0, len(cnts)):
        c = cnts[i]
        #determine and draw the largest contour
        #contrs = max(contours, key=cv2.contourArea)

        contoured = cv2.drawContours(imgray, c, -1, (255,255,255), 8)
        plt.plot(), plt.imshow(contoured, cmap="gray"), plt.title(f'Imgray after contour {i}')
        plt.show()

        #determine all edge points based on contour extremes
        leftmost = tuple(c[c[:, :, 0].argmin()][0])
        rightmost = tuple(c[c[:, :, 0].argmax()][0])
        topmost = tuple(c[c[:, :, 1].argmin()][0])
        bottommost = tuple(c[c[:, :, 1].argmax()][0])

        pa = np.where(c[:, 0, 1] == c[:, 0, 1].min())[0]
        pa_ = np.where(c[pa, 0] == c[pa,0][:, 0].max())[0][0]
        pap = c[pa,0][pa_]
        pb = np.where(c[:, 0, 1] == c[:, 0, 1].max())[0]
        pb_ = np.where(c[pb, 0] == c[pb, 0][:, 0].min())[0][0]
        pbp = c[pb, 0][pb_]
        pc = np.where(c[:, 0, 0] == c[:, 0, 0].min())[0]
        pc_ = np.where(c[pc, 0] == c[pc, 0][:, 1].max())[0][0]
        pcp = c[pc, 0][pc_]
        pd = np.where(c[:, 0, 0] == c[:, 0, 0].max())[0]
        pd_ = np.where(c[pd, 0] == c[pd, 0][:, 1].min())[0][0]
        pdp = c[pd,0][pd_]
        print("yes it is working")

        pnts = np.array([pap, pbp, pcp, pdp])
        #pnts = np.array([leftmost, rightmost, topmost, bottommost])

        print(f"points for corners are {pnts}")
        for i in range(0, len(pnts)):
            cv2.circle(contoured,pnts[i], 10, (255,255,255), -1)
    plt.plot(), plt.imshow(contoured, cmap="gray"), plt.title('Final with corners')
    plt.show()
    return contoured

imcol = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\OptimalSatV2.png")
image_resized = cv.resize(imcol, (900, 500))

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/2
imcol2 = cv2.filter2D(image_resized, -1, kernel)

equ = cv.equalizeHist(cv.cvtColor(imcol2, cv.COLOR_BGR2GRAY))
equ_col = cv.cvtColor(equ, cv.COLOR_GRAY2BGR)

rect = (image_resized.shape[1]//4, 1, image_resized.shape[1]//2, image_resized.shape[0]-2)
#rect = (1, 1, image_resized.shape[1]-2, image_resized.shape[0]-2)
print(f"rect is {rect}")
mask, processed_image = backremover(imcol2, rect, False)
print("process is running 1")

image_processed = process(processed_image, 8000, 10000, False)
edge_col = cv.cvtColor(image_processed, cv.COLOR_GRAY2BGR)

impic = (edge_col*image_resized)

pic = Outliner(imcol2)

plt.plot(), plt.imshow(mask, cmap="gray"), plt.title('Debugging image')
plt.show()
