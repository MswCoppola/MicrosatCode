import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from itertools import combinations
import math
import subprocess

def canny_edge_detector(img, can1, can2, double =False): #can1,  can2 are the hysteria thresholds
    im_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img, can1, can2, np.array(img.shape), 7)   # creating the canny edge image for processing
    kernel = np.ones((3, 3))                                         # defining kernel for dilation and erosion
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
    corners = cv2.cornerHarris(im_gr,3,3,0.2)

    b = np.argwhere(corners >= 0.5 * corners.max())  # finding the maximum corners (doesn't fully work, only in specific situations)

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
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel, 7,cv2.GC_INIT_WITH_RECT)

    # Create mask and image with removed background
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_new = img*mask2[:,:,np.newaxis]

    """ Debugging images"""
    plt.plot(), plt.imshow(imgray, cmap="gray"), plt.title('Internal Debugging image from backremover')
    plt.show()

    return mask2, img_new

def Corner_and_edge_outliner(imcol, aprx = True):   # aprx=True determines if it approximates the contour by the best points (corners) or if it just finds all contour defining points

    #convert image to gray image
    imgray = cv2.cvtColor(imcol, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3))
    img_erode = cv2.erode(imgray, kernel, iterations=1)             #TODO determine if errosion is necessary for face splitting (right now it seems like not)

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
            else:
                cnts.append(cnt)
            # drawing The contour
                cv2.drawContours(contoured, [approx], -1, (255, 255, 255))

    """ Debugging images area"""
    #plt.plot(), plt.imshow(contoured), plt.title(f'Imgray after approx')
    #plt.show()
    plt.plot(), plt.imshow(imcol, cmap="gray"), plt.title('Threshold')
    plt.show()
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

#--------------- Gregoire code --------------------
#Geometric_Three_Points

def sort_clockwise(points):
    """Sort points in clockwise order around the centroid."""
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

def cross_product(o, a, b):
    """Calculate the cross product of vectors OA and OB.
    A positive value means counter-clockwise, negative means clockwise.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def is_convex(points):
    """Check if the quadrilateral is convex using the cross product."""
    n = len(points)
    if n != 4:
        return False
    signs = []
    for i in range(n):
        o, a, b = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cp = cross_product(o, a, b)
        signs.append(cp)
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

def is_valid_quadrilateral(points):
    """Check if the points form a valid, convex quadrilateral."""
    try:
        polygon = Polygon(points)
        return polygon.is_valid and polygon.area > 0 and is_convex(points)
    except Exception as e:
        print(f"Error validating quadrilateral: {e}")
        return False

def generate_quadrilaterals(corner_points):
    """Generate valid quadrilaterals using clockwise-sorted points and convexity check."""
    valid_quadrilaterals = []
    for quad in combinations(corner_points, 4):
        quad_sorted = sort_clockwise(list(quad))
        if is_valid_quadrilateral(quad_sorted):
            valid_quadrilaterals.append(quad_sorted)
    return valid_quadrilaterals

def plot_quadrilaterals(valid_quads, corner_points):
    """Plot valid quadrilaterals separately."""
    for idx, quad in enumerate(valid_quads):
        plt.figure(figsize=(8, 8))
        x, y = zip(*quad)
        plt.plot(x + (x[0],), y + (y[0],), color='blue', marker='o', linestyle='-', linewidth=2,
                 label=f'Face {idx + 1}')
        plt.scatter(*zip(*corner_points), color='red', label='Corner Points')
        plt.title(f'Face {idx + 1} Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        #plt.show()
#Cuboid_Detection----------------------------
#import matplotlib.pyplot as plt
#from itertools import combinations
#from shapely.geometry import Polygon, Point, LineString
#from Geometric_Three_Points import generate_quadrilaterals, corner_points as all_corners
# -------------------- Cuboid Validation Functions --------------------

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

# -------------------- Plotting Functions --------------------

def plot_cuboid_candidates(cuboid_candidates):
    """Plot the identified cuboid candidates."""
    for idx, cuboid in enumerate(cuboid_candidates):
        plt.figure(figsize=(10, 10))

        # Plot each face in the cuboid
        for face in cuboid["faces"]:
            x, y = zip(*face)
            plt.plot(x + (x[0],), y + (y[0],), linestyle='-', linewidth=2, label=f'Face {cuboid["faces"].index(face) + 1}')

        plt.title(f'Cuboid Candidate {idx + 1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()


#Face_Determination--------------
#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#import numpy as np
#from Cuboid_Detection import cuboid_candidates  # Import detected cuboid candidates

def compute_face_area(face):
    """Compute the area of a quadrilateral face."""
    return Polygon(face).area

def compute_aspect_ratio(face):
    """Compute the aspect ratio of a quadrilateral face based on side lengths."""
    side_lengths = []
    for i in range(len(face)):
        p1 = np.array(face[i])
        p2 = np.array(face[(i + 1) % len(face)])
        side_length = np.linalg.norm(p2 - p1)
        side_lengths.append(side_length)

    side_lengths.sort()
    aspect_ratio = side_lengths[-1] / side_lengths[0] if side_lengths[0] != 0 else float('inf')
    return aspect_ratio

def classify_faces(cuboid_faces):
    """Classify the faces as Square, Rectangle 1, and Rectangle 2 considering aspect ratio."""
    face_data = [
        (face, compute_face_area(face), compute_aspect_ratio(face)) for face in cuboid_faces
    ]

    # Determine the square based on aspect ratio closest to 1
    face_data.sort(key=lambda x: (abs(x[2] - 1), x[1]))  # Prioritize aspect ratio, then area

    classified_faces = {
        "Square": face_data[0][0],
        "Rectangle 1": face_data[1][0],
        "Rectangle 2": face_data[2][0]
    }

    return classified_faces

def plot_classified_faces(classified_faces):
    """Plot the identified cuboid faces with correct classification and labels."""
    plt.figure(figsize=(10, 10))
    face_colors = {
        "Square": 'blue',
        "Rectangle 1": 'green',
        "Rectangle 2": 'orange'
    }

    for face_type, face in classified_faces.items():
        x, y = zip(*face)
        plt.plot(x + (x[0],), y + (y[0],), linestyle='-', linewidth=2, label=face_type,
                 color=face_colors[face_type])
        # Add label at centroid
        centroid_x = sum(x) / len(x)
        centroid_y = sum(y) / len(y)
        plt.text(centroid_x, centroid_y, face_type, fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title('Cuboid Face Classification with Labels (Aspect Ratio Adjusted)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

#Camera_Distance_Estimation-----------
#import numpy as np
#from shapely.geometry import LineString
#from Geometric_Three_Points import corner_points
#from Cuboid_Detection import cuboid_candidates
#import matplotlib.pyplot as plt
#import os

# Suppress threading warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Constants
PIXEL_SIZE_CM = 0.0008  # 8 microns per pixel (comment: Adjusted pixel size assumption)
FOCAL_LENGTH_CM = 0.5  # Camera focal length in cm
REAL_SHORT_CM = 3      # Real short edge length in cm
REAL_LONG_CM = 9       # Real long edge length in cm
EXPECTED_ASPECT_RATIO = REAL_LONG_CM / REAL_SHORT_CM  # Expected aspect ratio = 3.0


def calculate_center_of_mass(corner_points):
    """Calculate the center of mass of the cuboid in meters, assuming uniform density."""
    corner_points = np.array(corner_points)
    center_of_mass_xy = np.mean(corner_points, axis=0) * (PIXEL_SIZE_CM / 100)  # Convert cm to meters
    return center_of_mass_xy


def calculate_distance(short_avg, long_avg):
    """Estimate camera distance using the pinhole camera model with aspect ratio adjustments."""
    short_adjustment = REAL_SHORT_CM / short_avg if short_avg else 1
    long_adjustment = REAL_LONG_CM / long_avg if long_avg else 1
    short_distance = (FOCAL_LENGTH_CM * short_adjustment) / 100 if short_avg else None
    long_distance = (FOCAL_LENGTH_CM * long_adjustment) / 100 if long_avg else None
    distances = [d for d in [short_distance, long_distance] if d]
    avg_distance = np.mean(distances) if distances else None  # Convert cm to meters
    return avg_distance


def compute_camera_vector(center_of_mass, distance):
    """Compute the camera vector to the center of mass in meters (X, Y, Z)."""
    x, y = center_of_mass  # X and Y now in meters
    z = distance  # Z-axis distance in meters
    camera_vector = np.array([x, y, z])
    return camera_vector


def plot_2d_cuboid_with_center(corner_points, center_of_mass, faces):
    """Plot the 2D cuboid with its edges and center of mass (uses pre-validated cuboid faces)."""
    corner_points = np.array(corner_points) * (PIXEL_SIZE_CM / 100)  # Convert cm to meters
    plt.figure(figsize=(8, 8))

    # Plot cuboid faces
    for idx, face in enumerate(faces):
        x, y = zip(*np.array(face) * (PIXEL_SIZE_CM / 100))
        plt.plot(list(x) + [x[0]], list(y) + [y[0]], label=f'Face {idx + 1}')

    # Plot center of mass
    plt.scatter(center_of_mass[0], center_of_mass[1], c='red', label='Center of Mass', zorder=5)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('2D Projection of Cuboid with Center of Mass (Corrected)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_3d_camera_position(camera_vector, center_of_mass):
    """Plot the camera position relative to the cuboid center of mass in 3D (meters)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([center_of_mass[0]], [center_of_mass[1]], [0], c='blue', label='Cuboid Center of Mass')
    ax.scatter([camera_vector[0]], [camera_vector[1]], [camera_vector[2]], c='red', label='Camera Position')

    ax.plot([center_of_mass[0], camera_vector[0]],
            [center_of_mass[1], camera_vector[1]],
            [0, camera_vector[2]],
            c='green', linestyle='--', label='Vector to Camera')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    plt.title('Camera Position Relative to Cuboid Center of Mass (Meters)')
    plt.show()


def Camera_Distance_Estimation():
    """Estimate camera position vector from cuboid center of mass with X, Y in meters."""
    if cuboid_candidates:
        faces = cuboid_candidates[0]["faces"]
        center_of_mass = calculate_center_of_mass(corner_points)
        print(f"📍 Center of Mass (X, Y in meters): {center_of_mass}")

        # 2D Plot with center of mass (correctly referencing the cuboid faces)
        plot_2d_cuboid_with_center(corner_points, center_of_mass, faces)

        # Approximate observed edge lengths in pixels (using validated cuboid data)
        edges = []
        for face in faces:
            for i in range(len(face)):
                p1, p2 = np.array(face[i]), np.array(face[(i + 1) % len(face)])
                edge_length = np.linalg.norm(p2 - p1) * PIXEL_SIZE_CM
                edges.append(edge_length)

        short_edges = sorted(edges)[:4]  # Shortest 4 as short edges
        long_edges = sorted(edges)[-4:]  # Longest 4 as long edges

        short_avg = np.mean(short_edges)
        long_avg = np.mean(long_edges)

        distance = calculate_distance(short_avg, long_avg)
        if distance:
            print(f"✅ Estimated camera distance (Z-axis in meters): {distance:.5f} meters")
            camera_vector = compute_camera_vector(center_of_mass, distance)
            print(f"🎯 Camera position vector (X, Y, Z in meters): {camera_vector}")
            plot_3d_camera_position(camera_vector, center_of_mass)
        else:
            print("❌ Distance estimation failed.")
    else:
        print("❌ No valid cuboid detected.")
#Cuboid_Detection_Two------------------------
#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon, LineString, Point
#from itertools import combinations
#from Geometric_Three_Points import generate_quadrilaterals, corner_points
#import numpy as np

# -------------------- Global Variable for Final Plot --------------------
saved_final_two_face_cuboid = None  # Global variable to store the final two-face cuboid plot figure

# -------------------- Geometry Functions --------------------

def get_edges(face, tol=1e-3):
    """Return a globally unique list of edges represented as sorted tuples with relaxed tolerance."""
    def round_point(p):
        return tuple(np.round(p, decimals=3))

    edges = [tuple(sorted([round_point(face[i]), round_point(face[(i + 1) % 4])] )) for i in range(4)]
    return list(set(edges))  # Ensure globally unique edges

def edges_overlap(edge1, edge2, tol=1e-3):
    """Check if two edges overlap beyond sharing a single point, using LineString intersection."""
    line1 = LineString(edge1)
    line2 = LineString(edge2)
    intersection = line1.intersection(line2)
    return intersection.length > tol and not intersection.equals(line1) and not intersection.equals(line2)

def edges_exactly_shared(edge1, edge2, tol=1e-3):
    """Check if two edges are exactly the same, considering direction."""
    return (np.allclose(edge1[0], edge2[0], atol=tol) and np.allclose(edge1[1], edge2[1], atol=tol)) or \
           (np.allclose(edge1[0], edge2[1], atol=tol) and np.allclose(edge1[1], edge2[0], atol=tol))

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

def no_global_edge_overlap(all_faces, tol=1e-3):
    """Ensure no non-shared edges overlap globally across all quadrilaterals."""
    global_edges = []
    for face in all_faces:
        global_edges.extend(get_edges(face, tol))

    for i, e1 in enumerate(global_edges):
        for j, e2 in enumerate(global_edges):
            if i < j and not edges_exactly_shared(e1, e2, tol) and edges_overlap(e1, e2, tol):
                return False
    return True

def valid_shared_edge_combination(face1, face2, tol=1e-3):
    """Check if two quadrilaterals share exactly one edge and have no overlapping edges globally."""
    edges1 = get_edges(face1, tol)
    edges2 = get_edges(face2, tol)
    shared_edges = [e1 for e1 in edges1 if any(edges_exactly_shared(e1, e2, tol) for e2 in edges2)]

    if len(shared_edges) == 1:
        if no_global_edge_overlap([face1, face2], tol) and faces_are_disjoint_except_shared_edges([face1, face2]):
            return True
    return False

def plot_shared_edge_combinations(valid_quads):
    """Plot each combination of quadrilaterals that share a valid edge without any overlapping edges globally."""
    combination_index = 1
    found_combinations = False
    filtered_combinations = []
    for i, face1 in enumerate(valid_quads):
        for j, face2 in enumerate(valid_quads):
            if i < j and valid_shared_edge_combination(face1, face2, tol=1e-2):
                found_combinations = True
                filtered_combinations.append((face1, face2))
                fig, ax = plt.subplots(figsize=(8, 8))
                x1, y1 = zip(*face1)
                ax.plot(x1 + (x1[0],), y1 + (y1[0],), color='blue', marker='o', linestyle='-', label=f'Quadrilateral {i + 1}')

                x2, y2 = zip(*face2)
                ax.plot(x2 + (x2[0],), y2 + (y2[0],), color='green', marker='o', linestyle='-', label=f'Quadrilateral {j + 1}')

                shared_edge = get_edges(face1, tol=1e-2)
                for e1 in shared_edge:
                    if any(edges_exactly_shared(e1, e2, tol=1e-2) for e2 in get_edges(face2, tol=1e-2)):
                        x_edge, y_edge = zip(*list(e1))
                        ax.plot(x_edge, y_edge, color='red', linestyle='-', linewidth=3, label='Shared Edge')

                ax.set_title(f'Shared Edge (Strictly Non-Overlapping Edges) Combination {combination_index}: Q{i + 1} & Q{j + 1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True)
                plt.show()
                combination_index += 1

    if not found_combinations:
        print("No valid quadrilateral combinations found with strictly non-overlapping edges.")
    return filtered_combinations

def filter_final_combination(filtered_combinations):
    """Filter the previously plotted combinations to select the best cuboid representation and save the plot."""
    global saved_final_two_face_cuboid

    if filtered_combinations:
        print(f"\nApplying final filtering to {len(filtered_combinations)} combination(s)...")
        best_face1, best_face2 = filtered_combinations[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        x1, y1 = zip(*best_face1)
        ax.plot(x1 + (x1[0],), y1 + (y1[0],), color='blue', marker='o', linestyle='-', label='Final Quadrilateral 1')

        x2, y2 = zip(*best_face2)
        ax.plot(x2 + (x2[0],), y2 + (y2[0],), color='green', marker='o', linestyle='-', label='Final Quadrilateral 2')

        shared_edge = get_edges(best_face1)
        for e1 in shared_edge:
            if any(edges_exactly_shared(e1, e2) for e2 in get_edges(best_face2)):
                x_edge, y_edge = zip(*list(e1))
                ax.plot(x_edge, y_edge, color='red', linestyle='-', linewidth=3, label='Shared Edge')

        ax.set_title('Final Filtered Two-Face Cuboid Representation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)

        saved_final_two_face_cuboid = fig  # Save the final two-face cuboid plot figure
        plt.show()
    else:
        print("No valid cuboid representation found after final filtering.")

def show_saved_two_face_cuboid():
    """Display the saved final two-face cuboid plot if available."""
    if saved_final_two_face_cuboid:
        saved_final_two_face_cuboid.show()
    else:
        print("⚠️  No saved two-face cuboid plot available to display.")

#Face_Determination_Two----------------------------
#import matplotlib.pyplot as plt
#from shapely.geometry import Polygon
#import numpy as np
#from Geometric_Three_Points import generate_quadrilaterals, corner_points
#from Cuboid_Detection_Two import saved_final_two_face_cuboid  # Import saved plot from cuboid_determination_two

def compute_face_area(face):
    """Compute the area of a quadrilateral face."""
    return Polygon(face).area

def compute_aspect_ratio(face):
    """Compute the aspect ratio of a quadrilateral face based on side lengths."""
    side_lengths = []
    for i in range(len(face)):
        p1 = np.array(face[i])
        p2 = np.array(face[(i + 1) % len(face)])
        side_length = np.linalg.norm(p2 - p1)
        side_lengths.append(side_length)

    if not side_lengths:
        return float('inf')

    side_lengths.sort()
    aspect_ratio = side_lengths[-1] / side_lengths[0] if side_lengths[0] != 0 else float('inf')
    return aspect_ratio

def classify_faces(cuboid_faces):
    """Classify the faces as Square, Rectangle 1, and Rectangle 2 considering aspect ratio."""
    face_data = [
        (face, compute_face_area(face), compute_aspect_ratio(face)) for face in cuboid_faces
    ]

    if len(face_data) < 3:
        print(f"Warning: Only {len(face_data)} faces found. Adjusting classification accordingly.")
        classified_faces = {}
        if face_data:
            face_data.sort(key=lambda x: (abs(x[2] - 1), x[1]))
            labels = ["Square", "Rectangle 1", "Rectangle 2"]
            for i, data in enumerate(face_data):
                classified_faces[labels[i]] = data[0]
        return classified_faces

    # Determine the square based on aspect ratio closest to 1
    face_data.sort(key=lambda x: (abs(x[2] - 1), x[1]))  # Prioritize aspect ratio, then area

    classified_faces = {
        "Square": face_data[0][0],
        "Rectangle 1": face_data[1][0],
        "Rectangle 2": face_data[2][0]
    }

    return classified_faces

def plot_classified_faces(classified_faces):
    """Plot the identified cuboid faces with correct classification and labels."""
    if not classified_faces:
        print("No faces available for plotting.")
        return

    plt.figure(figsize=(10, 10))
    face_colors = {
        "Square": 'blue',
        "Rectangle 1": 'green',
        "Rectangle 2": 'orange'
    }

    for face_type, face in classified_faces.items():
        x, y = zip(*face)
        plt.plot(x + (x[0],), y + (y[0],), linestyle='-', linewidth=2, label=face_type,
                 color=face_colors.get(face_type, 'gray'))
        centroid_x = sum(x) / len(x)
        centroid_y = sum(y) / len(y)
        plt.text(centroid_x, centroid_y, face_type, fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title('Cuboid Face Classification from Final Two-Face Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def extract_faces_from_saved_plot(saved_final_two_face_cuboid):
    """Extract faces from the saved final two-face cuboid plot."""
    faces = []
    for ax in saved_final_two_face_cuboid.axes:
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) >= 4:
                face = list(zip(x_data[:-1], y_data[:-1]))
                if len(face) == 4:
                    faces.append(face)
    return faces

#Camera_Distance_Estimation_Two-------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon
import numpy as np
from Geometric_Three_Points import corner_points
from Cuboid_Detection_Two import saved_final_two_face_cuboid  # Import saved plot from cuboid_determination_two
from Face_Determination_Two import classified_faces  # Import face identification from face_determination_two
import os

# Suppress threading warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Constants
PIXEL_SIZE_CM = 0.0008  # 8 microns per pixel
FOCAL_LENGTH_CM = 0.5   # Camera focal length in cm
REAL_SHORT_CM = 3       # Real short edge length in cm
REAL_LONG_CM = 9        # Real long edge length in cm
EXPECTED_ASPECT_RATIO = REAL_LONG_CM / REAL_SHORT_CM  # Expected aspect ratio = 3.0

def calculate_center_of_mass(corner_points):
    """Calculate the center of mass of the cuboid in meters."""
    corner_points = np.array(corner_points)
    center_of_mass_xy = np.mean(corner_points, axis=0) * (PIXEL_SIZE_CM / 100)
    return center_of_mass_xy

def calculate_distance(short_avg, long_avg):
    """Estimate camera distance using the pinhole camera model."""
    short_adjustment = REAL_SHORT_CM / short_avg if short_avg else 1
    long_adjustment = REAL_LONG_CM / long_avg if long_avg else 1
    short_distance = (FOCAL_LENGTH_CM * short_adjustment) / 100 if short_avg else None
    long_distance = (FOCAL_LENGTH_CM * long_adjustment) / 100 if long_avg else None
    distances = [d for d in [short_distance, long_distance] if d]
    avg_distance = np.mean(distances) if distances else None
    return avg_distance

def compute_camera_vector(center_of_mass, distance):
    """Compute the camera vector to the center of mass in meters (X, Y, Z)."""
    x, y = center_of_mass
    z = distance
    camera_vector = np.array([x, y, z])
    return camera_vector

def plot_3d_camera_position(camera_vector, center_of_mass):
    """Plot the camera position relative to the cuboid center of mass in 3D (meters) with vector annotation."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter([center_of_mass[0]], [center_of_mass[1]], [0], c='blue', label='Cuboid Center of Mass')
    ax.scatter([camera_vector[0]], [camera_vector[1]], [camera_vector[2]], c='red', label='Camera Position')

    # Plot vector
    ax.plot([center_of_mass[0], camera_vector[0]],
            [center_of_mass[1], camera_vector[1]],
            [0, camera_vector[2]],
            c='green', linestyle='--', label=f'Vector to Camera: {camera_vector}')

    # Annotations
    ax.text(camera_vector[0], camera_vector[1], camera_vector[2],
            f"Camera\n({camera_vector[0]:.2e}, {camera_vector[1]:.2e}, {camera_vector[2]:.2f} m)",
            color='red')
    ax.text(center_of_mass[0], center_of_mass[1], 0,
            f"Center\n({center_of_mass[0]:.2e}, {center_of_mass[1]:.2e}, 0 m)",
            color='blue')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    plt.title('Camera Position and Vector Relative to Cuboid Center of Mass (Meters)')
    plt.show()

def Camera_Distance_Estimation_Two(corn_list):
    """Estimate camera position vector and plot it in 3D using pre-identified faces."""
    if saved_final_two_face_cuboid and classified_faces:
        center_of_mass = calculate_center_of_mass(corn_list)
        print(f"📍 Center of Mass (X, Y in meters): {center_of_mass}")

        edges = []
        for face_label, face in classified_faces.items():
            for i in range(len(face)):
                p1, p2 = np.array(face[i]), np.array(face[(i + 1) % len(face)])
                edge_length = np.linalg.norm(p2 - p1) * PIXEL_SIZE_CM
                edges.append(edge_length)

        short_edges = sorted(edges)[:4]
        long_edges = sorted(edges)[-4:]
        short_avg = np.mean(short_edges)
        long_avg = np.mean(long_edges)

        distance = calculate_distance(short_avg, long_avg)
        if distance:
            print(f"✅ Estimated camera distance (Z-axis in meters): {distance:.5f} meters")
            camera_vector = compute_camera_vector(center_of_mass, distance)
            print(f"🎯 Camera position vector (X, Y, Z in meters): {camera_vector}")
            plot_3d_camera_position(camera_vector, center_of_mass)
        else:
            print("❌ Distance estimation failed.")
    else:
        print("❌ Required data not available: Either no saved cuboid plot or face identification missing.")
