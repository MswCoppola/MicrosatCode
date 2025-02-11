import cv2
import numpy as np
from matplotlib import pyplot as plt


def canny_edge_detector(img, can1, can2): #can1,  can2 are the hysteria thresholds
    img_canny = cv2.Canny(img, can1, can2, np.array(img.shape), 7)   # creating the canny edge image for processing
    kernel = np.ones((3, 3))                                         # defining kernel for dilation and erosion
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    corners = cv2.cornerHarris(img_canny,14,7,0.04)

    b = np.argwhere(corners >= 0.5 * corners.max())  # finding the maximum corners

    """ Debuging images"""
    #plt.plot(), plt.imshow(img), plt.title('image for mapping')
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

def background_remover(img, rect):                # Provide a rectangle section for possible sat position and define used backremover method
    assert img is not None, "file could not be read, check with os.path.exists()"
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # Convert image to grayscale

    # create arrays for grabcut alghorithm
    mask = np.zeros(imgray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply grabcut alghorithm to remove background
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

def face_grouping_detection(imcol, pnts):
    return  # output should be a dictionary containing the face as key and corresponding points as output

def ellipse_fit_axis(imcol, pnts):
    return  # output should be 2 points in 2d space and an out of plane angle

def range_detection(imcol, size, edge): # input is the processed image, size of the satellite and a collection of the detected edges
    return
