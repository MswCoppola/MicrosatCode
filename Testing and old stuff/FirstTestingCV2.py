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
    im_gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img, can1, can2, np.array(img.shape), 7)   # creating the canny edge image for processing
    kernel = np.ones((3, 3))                                         # defining kernel for dilation and erosion
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)

    if double is True: # Creating inset face edges for each face, an alternative method for forcing faces to be split
        img_dilate2 = cv2.dilate(img_canny, kernel, iterations=3)
        img_canny2 = cv2.Canny(img_dilate2, can1, can2, np.array(img.shape), 5)
        corners = cv2.cornerHarris(img_canny2, 4, 7, 0.02)
    else: # Just calculate corners
        corners = cv2.cornerHarris(im_gr,3,3,0.2)

    b = np.argwhere(corners >= 0.5 * corners.max())  # finding the maximum corners

    """ Debuging images"""
    # plt.plot(), plt.imshow(img), plt.title('image for mapping')
    # plt.show()
    plt.plot(), plt.imshow(img_canny), plt.title(f'canny image for double is {double}')
    plt.show()
    # plt.plot(), plt.imshow(corners), plt.title('corner weights')
    # plt.show()

    """ Debugging for large corners"""
    #print(f"b = {b}")
    #print(f"b-1 = {b[:,::-1]}")
    #print(f"corners = {corners}")
    #print(f"corner shape = {corners.shape}")
    #print(f"max corner = {corners[b[:,::-1]]}")

    inverted_mask = cv2.bitwise_not(img_dilate)
    return inverted_mask

def background_remover(img, rect, thrs= False):                # Provide a rectangle section for possible sat position and define used backremover method
    assert img is not None, "file could not be read, check with os.path.exists()"
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # Convert image to grayscale

    # create arrays for grabcut alghorithm
    mask = np.zeros(imgray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    print("we are getting to here 1")

    # Apply grabcut alghorithm to remove background
    if thrs is True:                                    # Using thresholds first to limit contamination
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        imguse = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.grabCut(imguse, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)
    else:   # Not using thresholds
        print("we are getting to here 2")
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel, 7,cv2.GC_INIT_WITH_RECT)
        print("we are getting to here 3")

    # Create mask and image with removed background
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_new = img*mask2[:,:,np.newaxis]

    """ Debugging images"""
    # plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Internal Debugging image from backremover')
    # plt.show()

    return mask2, img_new

def Corner_and_edge_outliner(imcol, aprx = True):

    #convert image to gray image
    imgray = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3))
    img_erode = cv2.erode(imgray, kernel, iterations=1)             #TODO determine if errosion is necessary for face splitting

    #threshold the image and draw the contours
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contoured = imgray
    all_corners =[]
    #save the largest contours (size by area)
    cnts = []
    approximations = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 100:
            # method if aprx is set as true, aprx defines if edges are to be interpolated or directly measured
            if aprx is True:
                perimeter = cv2.arcLength(cnt, True)
                it = 10
                per = 0.02 * perimeter
                approx = cv2.approxPolyDP(cnt, per, True)
                arr = approx.reshape(-1, approx.shape[-1])
                print(f"current values for approx = {arr}")
                """while len(approx) > 4 and it!=0:
                    print(f"current iteration = {it} and current #pnts = {len(approx)}")
                    per = 1.1*per
                    approx = cv2.approxPolyDP(cnt, per, True)
                    it -= 1"""
                for point in approx:
                    x, y = point[0]
                    all_corners.append(point[0])
                    approximations.append(arr)
                    cv2.circle(contoured, (x, y), 5, (255, 255, 255), -1)
            else:
                cnts.append(cnt)
            # drawing skewed rectangle
                cv2.drawContours(contoured, [approx], -1, (255, 255, 255))

    """ Debugging images area"""
    #plt.plot(), plt.imshow(contoured), plt.title(f'Imgray after approx')
    #plt.show()
    plt.plot(), plt.imshow(imcol, cmap="gray"), plt.title('Threshold')
    plt.show()
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
        return contoured, approximations, all_corners
    return contoured, pnts

def Rough_loc(imcol, block, wind_sens=0.05):
    y, x, z = imcol.shape
    print(y,x,z)
    amt_x = x // block[0]  # define blocks for binning image intensities
    amt_y = y // block[1]
    int_img = np.zeros((y, x))
    for i in range(0, amt_y):
        for j in range(0, amt_x):
            strt_x, nd_x = i * block[1], (i + 1) * block[1]
            strt_y, nd_y = j * block[0], (j + 1) * block[0]
            intensity = np.mean(imcol[strt_x:nd_x, strt_y:nd_y])  # bin image intensity into blocks for faster processing
            int_img[strt_x:nd_x, strt_y:nd_y] = intensity
    int_high = np.where(int_img >= 0.8 * int_img.max())
    int_med = np.where(int_img >= wind_sens * int_img.max())
    xp, yp = int(np.mean(int_high[1]))+1, int(np.mean(int_high[0]))+1
    cv2.circle(int_img, (xp, yp), 10, (255, 255, 255), -1)
    rel_x, rel_y = xp - x // 2, yp - y // 2  # detect the rough center of the satellite relative to image size

    # plt.plot(), plt.imshow(int_img, cmap="gray"), plt.title('Debugging image for the loc_on_screen Function')
    # plt.show()

    x_len = np.max(int_med[1]) - np.min(int_med[1]) + 1
    y_len = np.max(int_med[0]) - np.min(int_med[0]) + 1
    window = (np.min(int_med[1]), np.min(int_med[0]), x_len, y_len)  # window for the background removal process
    print(f"window is {window}")
    return (xp, yp), (rel_x, rel_y), window

import math

def filter_close_points(points, threshold):
    """
    Filters a list of points (x, y) and averages points closer than the threshold.

    Args:
      points: A list of tuples representing points (x, y).
      threshold: The maximum distance between points to be averaged.

    Returns:
      A new list containing the averaged points.
    """
    filtered_points = []
    i = 0
    while i < len(points):
        # Initialize a list to store points to be averaged
        average_group = [points[i]]
        j = i + 1
        while j < len(points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance < threshold:
                average_group.append(points[j])
                # Remove the point from the original list to avoid duplicates
                del points[j]
            else:
                j += 1
        # Calculate the average of the points in the group
        x_avg = int(sum([x for x, _ in average_group]) / len(average_group))
        y_avg = int(sum([y for _, y in average_group]) / len(average_group))
        filtered_points.append((x_avg, y_avg))
        i += 1
    return filtered_points





def multi_corner_grouper(im1, im2):  # #######################################
    return


#imcol = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\SLFrame1.jpg")
imcol = cv2.imread(r"C:\Users\massi\Downloads\TryImage_block_blackEdge.jpeg")
#imcol2 = cv2.imread(r"C:\Users\massi\OneDrive\Afbeeldingen\Microsat_pictures\OptSatNew_2.png")

image_resized = cv.resize(imcol, (900, 500))
#image_resized2 = cv.resize(imcol2, (900, 500))

location = Rough_loc(image_resized, (900//10, 500//10), 0.1)
#location2 = Rough_loc(image_resized2, (900//6, 500//6))

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
imcol_2 = cv2.filter2D(image_resized, -1, kernel)

equ = cv.equalizeHist(cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY))
equ_col = cv.cvtColor(equ, cv.COLOR_GRAY2BGR)

rect = location[2]
#rect2 = location2[2]
#rect = (1, 1, image_resized.shape[1]-2, image_resized.shape[0]-2)

plt.plot(), plt.imshow(image_resized, cmap="gray"), plt.title('Primary target image')
plt.show()

print("process is running 1")
mask, processed_image = background_remover(image_resized, rect, False)
maskcol = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)*100
#mask2, processed_image2 = background_remover(image_resized2, rect2, False)
#maskcol2 = cv.cvtColor(mask2, cv.COLOR_GRAY2BGR)*100

print("process is running 2")

image_processed = canny_edge_detector(processed_image, 10000, 80000, False)
edge_col = cv.cvtColor(image_processed, cv.COLOR_GRAY2BGR)
#image_processed2 = canny_edge_detector(processed_image2, 3000, 10000, False)
#edge_col2 = cv.cvtColor(image_processed2, cv.COLOR_GRAY2BGR)

#impic = (edge_col*image_resized)

masked_img = cv.bitwise_and(edge_col,maskcol,mask = mask)
#masked_img2 = cv.bitwise_and(edge_col2,maskcol2,mask = mask2)

pic = Corner_and_edge_outliner(masked_img, True)
#pic2 = Corner_and_edge_outliner(masked_img2, True)
print(f"The following points are corners {pic[2]}")
""


grpd_corn = filter_close_points(pic[2], 10)
print(f"grouped corners are {grpd_corn}")
extr = mask.copy()
for i in grpd_corn:
    cv2.circle(extr, i, 4, (100, 100, 100), -1)


plt.plot(), plt.imshow(image_resized, cmap="gray"), plt.title('Primary target image')
plt.show()
plt.plot(), plt.imshow(pic[0], cmap="gray"), plt.title('Final cornered image 2')
plt.show()
plt.plot(), plt.imshow(extr, cmap="gray"), plt.title('Debugging image final')
plt.show()


"""# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(image_resized, None)
kp2, des2 = orb.detectAndCompute(image_resized2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
print(des1)
# Match descriptors

matches = bf.match(des1, des2)
distance_threshold = 20
# Apply the ratio test and store unique matches
good_matches = [match for match in matches if match.distance < distance_threshold]


# Draw matches
result_img = cv2.drawMatches(image_resized, kp1, image_resized2, kp2, good_matches, None)

cv2.imshow('Good Matches', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(image_resized, None)
keypoints2, descriptors2 = orb.detectAndCompute(image_resized2, None)

# Use BFMatcher to match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on their distance
matches = sorted(matches, key=lambda x: x.distance)
distance_threshold = 20  # Adjust this value based on your needs

# Filter matches based on distance
good_matches = [match for match in matches if match.distance < distance_threshold]

# Extract pixel locations of matched keypoints
matched_keypoints1 = [keypoints1[m.queryIdx].pt for m in good_matches]
matched_keypoints2 = [keypoints2[m.trainIdx].pt for m in good_matches]

# Print pixel locations
for i, (pt1, pt2) in enumerate(zip(matched_keypoints1, matched_keypoints2)):
    print(f"Match {i}: Image 1 - {pt1}, Image 2 - {pt2}")

result_img = cv2.drawMatches(image_resized, keypoints1, image_resized2, keypoints2, good_matches, None)
image_with_keypoints = cv2.drawKeypoints(image_resized, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Keypoints', image_with_keypoints)

cv2.imshow('Good Matches', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""