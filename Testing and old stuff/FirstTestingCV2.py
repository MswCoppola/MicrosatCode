import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def All_non_zero(gray_image, threshold, replacement, hightresh=False):
    out = gray_image.copy()
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

def canny_edge_detector(img, can1, can2, double =False): #can1,  can2 are the hysteria thresholds
    img_canny = cv2.Canny(img, can1, can2, np.array(img.shape), 7)   # creating the canny edge image for processing
    kernel = np.ones((3, 3))                                         # defining kernel for dilation and erosion
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)

    if double is True: # Creating inset face edges for each face, an alternative method for forcing faces to be split
        img_dilate2 = cv2.dilate(img_canny, kernel, iterations=3)
        img_canny2 = cv2.Canny(img_dilate2, can1, can2, np.array(img.shape), 5)
        corners = cv2.cornerHarris(img_canny2, 4, 7, 0.02)
    else: # Just calculate corners
        corners = cv2.cornerHarris(img_canny,14,7,0.04)

    b = np.argwhere(corners >= 0.5 * corners.max())  # finding the maximum corners

    """ Debuging images"""
    #plt.plot(), plt.imshow(img), plt.title('image for mapping')
    #plt.show()
    #plt.plot(), plt.imshow(img_canny), plt.title(f'canny image for double is {double}')
    #plt.show()
    #plt.plot(), plt.imshow(corners), plt.title('corner weights')
    #plt.show()

    """ Debugging for large corners"""
    #print(f"b = {b}")
    #print(f"b-1 = {b[:,::-1]}")
    #print(f"corners = {corners}")
    #print(f"corner shape = {corners.shape}")
    #print(f"max corner = {corners[b[:,::-1]]}")

    inverted_mask = cv2.bitwise_not(img_canny)
    return inverted_mask

def background_remover(img, rect, thrs= False):                # Provide a rectangle section for possible sat position and define used backremover method
    assert img is not None, "file could not be read, check with os.path.exists()"
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # Convert image to grayscale

    # create arrays for grabcut alghorithm
    mask = np.zeros(imgray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply grabcut alghorithm to remove background
    if thrs is True:                                    # Using thresholds first to limit contamination
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imguse = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.grabCut(imguse, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)
    else:   # Not using thresholds
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel, 7,cv2.GC_INIT_WITH_RECT)

    # Create mask and image with removed background
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_new = img*mask2[:,:,np.newaxis]

    """ Debugging images"""
    plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Internal Debugging image from backremover')
    plt.show()

    return mask2, img_new

def Corner_and_edge_outliner(imcol, aprx = True):

    #convert image to gray image
    imgray = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)//2
    kernel = np.ones((3, 3))
    img_erode = cv2.erode(imgray, kernel, iterations=2)             #TODO determine if errosion is necessary for face splitting

    #threshold the image and draw the contours
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contoured = imgray

    #save the largest contours (size by area)
    cnts = []
    approximations = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        per = 0.02*perimeter
        approx = cv2.approxPolyDP(cnt, per, True)
        print(f"current values for approx = {approx}")
        approximations.append(approx)
        area = cv2.contourArea(cnt)
        # method if aprx is set as true, aprx defines if edges are to be interpolated or directly measured
        if aprx is True:
            for point in approx:
                x, y = point[0]
                cv2.circle(contoured, (x, y), 5, (255, 255, 255), -1)

            # drawing skewed rectangle
            cv2.drawContours(contoured, [approx], -1, (255, 255, 255))
        if area >= 200:
            cnts.append(cnt)

    """ Debugging images area"""
    #plt.plot(), plt.imshow(contoured), plt.title(f'Imgray after approx')
    #plt.show()
    #plt.plot(), plt.imshow(thresh, cmap="gray"), plt.title('Threshold')
    #plt.show()
    #plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Imgray before contour')
    #plt.show()

    if aprx is False:
        for i in range(0, len(cnts)):
            c = cnts[i]
            #determine and draw the largest contour
            #contrs = max(contours, key=cv2.contourArea)

            contoured = cv2.drawContours(imgray, c, -1, (255,255,255), 2)
            plt.plot(), plt.imshow(contoured), plt.title(f'Imgray after contour {i}')
            plt.show()

            #determine all edge points based on contour extremes
            leftmost = tuple(c[c[:, :, 0].argmin()][0])
            rightmost = tuple(c[c[:, :, 0].argmax()][0])
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            bottommost = tuple(c[c[:, :, 1].argmax()][0])

            # alternative points determination
            # TODO find most effective method
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
            print("yes it is working") #Debugging

            pnts = np.array([pap, pbp, pcp, pdp])
            #pnts = np.array([leftmost, rightmost, topmost, bottommost])

            #print(f"points for corners are {pnts}") #Debugging
            for i in range(0, len(pnts)):
                cv2.circle(contoured,pnts[i], 10, (255,255,255), -1)

    """ Debugging image area"""
    #plt.plot(), plt.imshow(contoured, cmap="gray"), plt.title('Final with corners')
    #plt.show()

    if aprx is True:
        return contoured, approximations
    return contoured, pnts

def Rough_loc(imcol, block):
    y, x, z = imcol.shape
    amt_x = x//block[0]
    amt_y = y//block[1]
    int_img = np.zeros((y, x))
    for i in range(0, amt_y):
        for j in range(0, amt_x):
            strt_x, nd_x = i*block[1], (i+1)*block[1]
            strt_y, nd_y = j*block[0], (j+1)*block[0]
            intensity = np.mean(imcol[strt_x:nd_x, strt_y:nd_y])
            int_img[strt_x:nd_x, strt_y:nd_y] = intensity
    int_high = np.where(int_img >= 0.8*int_img.max())
    int_med = np.where(int_img >= 0.2 * int_img.max())
    xp, yp = int(np.mean(int_high[1])), int(np.mean(int_high[0]))
    cv2.circle(int_img, (xp, yp), 10, (255, 255, 255), -1)
    print(f"int_med[0] {int_med[0]}")
    print(f"int_med[1] {int_med[1]}")
    x_len = np.max(int_med[1]) - np.min(int_med[1]) + 1
    print(x_len)
    rel_x, rel_y = xp - x//2, yp - y//2

    plt.plot(), plt.imshow(int_img, cmap="gray"), plt.title('Debugging image')
    plt.show()
    return (xp, yp), (rel_x, rel_y)

#imcol = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\SLFrame1.jpg")
imcol = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\ModelDavinciMicrosatDahlengCam.jpeg")
image_resized = cv.resize(imcol, (900, 500))

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
imcol_2 = cv2.filter2D(image_resized, -1, kernel)

equ = cv.equalizeHist(cv.cvtColor(imcol_2, cv.COLOR_BGR2GRAY))
equ_col = cv.cvtColor(equ, cv.COLOR_GRAY2BGR)

rect = (image_resized.shape[1]//4, 1, image_resized.shape[1]//2, image_resized.shape[0]-2)
#rect = (1, 1, image_resized.shape[1]-2, image_resized.shape[0]-2)

""
mask, processed_image = background_remover(image_resized, rect, False)
maskcol = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)*100
print("process is running 1")

image_processed = canny_edge_detector(processed_image, 8000, 10000, False)
edge_col = cv.cvtColor(image_processed, cv.COLOR_GRAY2BGR)

#impic = (edge_col*image_resized)

masked_img = cv.bitwise_and(edge_col,edge_col,mask = mask)

pic = Corner_and_edge_outliner(maskcol, True)
print(f"The following points are corners {pic[1]}")
""

location = Rough_loc(image_resized, (100, 100))
print(f"Current output of Rough_loc = {location}")

#plt.plot(), plt.imshow(pic[0], cmap="gray"), plt.title('Final cornered image')
#plt.show()
plt.plot(), plt.imshow(image_processed), plt.title('Debugging image mask')
plt.show()