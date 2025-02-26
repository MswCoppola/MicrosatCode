import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from itertools import combinations
from shapely.geometry import Polygon, Point, LineString
from VisualDataProcessing.BaseFunctions.Geometric_Three_Points import generate_quadrilaterals, corner_points as all_corners

def canny_edge_detector(img, can1, can2, double =False): #can1,  can2 are the hysteria thresholds
    img_canny = cv2.Canny(img, can1, can2, np.array(img.shape), 7)   # creating the canny edge image for processing
    kernel = np.ones((3, 3))                                         # defining kernel for dilation and erosion
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    """ Debuging images"""
    # plt.plot(), plt.imshow(img), plt.title('image for mapping')
    # plt.show()
    #plt.plot(), plt.imshow(img_canny), plt.title(f'canny image for double is {double}')
    #plt.show()

    inverted_mask = cv2.bitwise_not(img_dilate)     # Invert the edge intensity to accomodate removal of edges from the image in later stage
    return inverted_mask

def background_remover(img, rect):                # Provide a rectangle section for possible sat position and define used backremover method
    assert img is not None, "file could not be read, check with os.path.exists()"
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # Convert image to grayscale

    # create arrays for grabcut alghorithm
    mask = np.zeros(imgray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply grabcut alghorithm to remove background
    #cv2.grabCut(img,mask,rect,bgdModel,fgdModel, 7,cv2.GC_INIT_WITH_RECT)
    ret, thresh = cv2.threshold(imgray, 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imguse = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.grabCut(imguse, mask, rect, bgdModel, fgdModel, 7, cv2.GC_INIT_WITH_RECT)

    # Create mask and image with removed background
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_new = img*mask2[:,:,np.newaxis]

    """ Debugging images"""
    #plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Internal Debugging image from backremover')
    #plt.show()

    return mask2, img_new

def Corner_and_edge_outliner(imcol, aprx = True):   # aprx=True determines if it approximates the contour by the best points (corners) or if it just finds all contour defining points

    #convert image to gray image
    imgray = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)

    #threshold the image and draw the contours
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contoured = imcol
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

                # The following code can be used if more than 4 points are found for a face to try and limit the points to 4
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
                cv2.drawContours(contoured, [approx], -1, (255, 255, 255))
            else:
                cnts.append(cnt)
            # drawing The contour
                cv2.drawContours(contoured, [approx], -1, (255, 255, 255))

    """ Debugging images area"""
    #plt.plot(), plt.imshow(contoured), plt.title(f'Imgray after approx')
    #plt.show()
    #plt.plot(), plt.imshow(imcol, cmap="gray"), plt.title('Threshold')
    #plt.show()
    #plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Imgray before contour')
    #plt.show()
    if aprx is False:
        return contoured, cnts
    return contoured, approximations, all_corners

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

def face_grouping_detection(imcol, pnts):
    return  # output should be a dictionary containing the face as key and corresponding points as output

def ellipse_fit_axis(imcol, pnts):
    return  # output should be 2 points in 2d space and an out of plane angle

def range_detection(imcol, size, edge): # input is the processed image, size of the satellite and a collection of the detected edges
    return

def all_points_used(face_combination, all_corners):
    """Check if all given corner points are used in the selected faces."""
    used_points = set()
    for face in face_combination:
        used_points.update(face)
    return set(all_corners) == used_points

def no_points_within_faces(face_combination, all_corners):
    """Ensure no unused points are inside any of the selected faces."""
    polygons = [Polygon(face) for face in face_combination]
    for point in all_corners:
        if any(polygon.contains(Point(point)) for polygon in polygons):
            return False  # A point lies inside one of the faces
    return True

def faces_are_disjoint_except_shared_edges(face_combination):
    """Ensure faces are disjoint except where they share edges."""
    polygons = [Polygon(face) for face in face_combination]
    for i, poly1 in enumerate(polygons):
        for j, poly2 in enumerate(polygons):
            if i >= j:
                continue
            intersection = poly1.intersection(poly2)
            if not (intersection.is_empty or isinstance(intersection, LineString)):
                return False  # Polygons overlap beyond shared edges
    return True

def identify_cuboids_from_faces(valid_quads, all_corners):
    """Identify possible cuboids from the given faces with specific constraints."""
    cuboid_candidates = []

    # Iterate over all combinations of three faces
    for face_combination in combinations(valid_quads, 3):
        if (all_points_used(face_combination, all_corners) and
            no_points_within_faces(face_combination, all_corners) and
            faces_are_disjoint_except_shared_edges(face_combination)):
            cuboid_candidates.append({
                "faces": face_combination,
            })
    return cuboid_candidates

# -------------------- Main Execution --------------------

"""# Use the quadrilaterals from Geometric_Three_Points
valid_quads = generate_quadrilaterals(all_corners)
print(f"✅ Generated {len(valid_quads)} valid quadrilaterals from Geometric_Three_Points.")

# Identify cuboid candidates
cuboid_candidates = identify_cuboids_from_faces(valid_quads, all_corners)
print(cuboid_candidates)
print(f"✅ Identified {len(cuboid_candidates)} possible cuboids.")"""

