import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from itertools import combinations
from shapely.geometry import Polygon, Point, LineString
from scipy.optimize import linear_sum_assignment
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
                # print(f"current values for approx = {arr}")

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

def filter_close_points(points, threshold= 15):
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


def analyze_and_plot_ellipses(ellipse_data):
    """
    Given multiple sets of points defining ellipses, this function:
    1. Computes and plots ellipses from the given points.
    2. Finds the centers and eccentricities of each ellipse.
    3. Calculates the best-fit line through the ellipse centers.
    4. Computes the average eccentricity and viewing angle θ (arcsin(e_avg)).
    5. If e_avg ≥ 0.9, draws a perpendicular line through the midpoint of centers.
    6. Plots everything: ellipses, centers, best-fit line, and perpendicular line if applicable.

    Returns:
    - (m, c): Slope and intercept of the best-fit line (y = mx + c).
    - theta: Viewing angle in degrees.
    """

    rot_ax_found = False

    def fit_ellipse(points):
        """Fit an ellipse to a given set of points using least squares."""
        A = np.array([[x ** 2, x * y, y ** 2, x, y, 1] for x, y in points])
        _, _, Vt = np.linalg.svd(A)
        return Vt[-1, :]  # Last row of Vt corresponds to the best-fit solution

    def ellipse_center(coeffs):
        """Calculate the center (x0, y0) of the ellipse."""
        A, B, C, D, E, _ = coeffs
        denominator = (4 * A * C - B ** 2)
        return None if denominator == 0 else ((B * E - 2 * C * D) / denominator, (B * D - 2 * A * E) / denominator)

    def ellipse_axes(coeffs):
        """Compute semi-major axis (a), semi-minor axis (b), and eccentricity (e)."""
        A, B, C, _, _, _ = coeffs
        matrix = np.array([[A, B / 2], [B / 2, C]])
        eigenvalues, _ = np.linalg.eig(matrix)
        if np.any(eigenvalues <= 0): return None, None, None  # Invalid ellipse
        a, b = np.sqrt(1 / np.min(eigenvalues)), np.sqrt(1 / np.max(eigenvalues))
        return a, b, np.sqrt(1 - (b ** 2 / a ** 2))  # Eccentricity

    def fit_best_fit_line(centers):
        """Perform least squares regression for best-fit line y = mx + c through centers."""
        x_vals, y_vals = zip(*centers)
        A = np.vstack([x_vals, np.ones(len(x_vals))]).T
        return np.linalg.lstsq(A, y_vals, rcond=None)[0]  # Returns (m, c)

    # Store ellipse centers and eccentricities
    centers = []
    eccentricities = []

    # Plot setup
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for idx, points in enumerate(ellipse_data):
        coeffs = fit_ellipse(points)
        center = ellipse_center(coeffs)
        if center:
            print(f"Ellipse {idx + 1} Center: {center}")
            centers.append(center)

        a, b, e = ellipse_axes(coeffs)
        if a and b:
            print(f"Ellipse {idx + 1} Semi-Major Axis (a): {a:.4f}, Semi-Minor Axis (b): {b:.4f}")
            print(f"Ellipse {idx + 1} Eccentricity: {e:.4f}")
            eccentricities.append(e)

        # Generate ellipse contour for plotting
        x_vals = np.linspace(min(p[0] for p in points) - 3, max(p[0] for p in points) + 3, 400)
        y_vals = np.linspace(min(p[1] for p in points) - 3, max(p[1] for p in points) + 3, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = sum(coeffs[i] * term for i, term in enumerate([X ** 2, X * Y, Y ** 2, X, Y, np.ones_like(X)]))

        if Z.min() < 0 < Z.max():
            plt.contour(X, Y, Z, levels=[0], colors=colors[idx % len(colors)], linewidths=2)

        # Plot original points
        px, py = zip(*points)
        plt.scatter(px, py, color=colors[idx % len(colors)], label=f"Ellipse {idx + 1} Points", zorder=3)

        # Plot ellipse center
        if center:
            plt.scatter(*center, color=colors[idx % len(colors)], marker='x', s=100, label=f"Ellipse {idx + 1} Center")

    # Compute best-fit line
    if len(centers) > 1 and eccentricities:
        m, c = fit_best_fit_line(centers)
        x_min, x_max = min(x for x, _ in centers) - 2, max(x for x, _ in centers) + 2
        x_line = np.linspace(x_min, x_max, 100)
        y_line = m * x_line + c
        plt.plot(x_line, y_line, 'k--', label="Best-Fit Line (Centers)", linewidth=2)
        print(f"Best-Fit Line Equation: y = {m:.4f}x + {c:.4f}")

        # Compute average eccentricity and viewing angle
        e_avg = np.mean(eccentricities)
        theta = np.arcsin(e_avg) * (180 / np.pi)  # Convert radians to degrees
        print(f"Average Eccentricity: {e_avg:.4f}")
        print(f"Viewing Angle (θ) = arcsin(e_avg): {theta:.4f}°")

        # If e_avg is high, draw a perpendicular line through the midpoint of the centers
        if e_avg >= 0.9:
            x_mid, y_mid = np.mean(centers, axis=0)  # Compute midpoint of centers
            perp_slope = -1 / m if m != 0 else np.inf  # Perpendicular slope

            # Compute line points
            if perp_slope == np.inf:
                x_perp = np.array([x_mid, x_mid])
                y_perp = np.array([y_mid - 5, y_mid + 5])
            else:
                x_perp = np.linspace(x_mid - 5, x_mid + 5, 100)
                y_perp = perp_slope * (x_perp - x_mid) + y_mid

            plt.plot(x_perp, y_perp, 'b-', linewidth=2, label="Perpendicular Line")
            print(f"Perpendicular Line through ({x_mid:.2f}, {y_mid:.2f})")

    else:
        m, c, theta = None, None, None

    # Plot settings
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Ellipses with Centers, Eccentricity & Best-Fit Line")
    plt.grid()
    plt.show()

    return [(m, c), theta], rot_ax_found


def range_detection(imcol, size, edge): # input is the processed image, size of the satellite and a collection of the detected edges
    return


def match_vertices_series(series, threshold_factor=1.7):
    """
    Matches corresponding vertices across multiple 2D projections and returns the series of coordinates for each point.

    Parameters:
        series (list of np.ndarray): List of 2D point sets, each representing a frame.
        threshold_factor (float): Multiplier for the mean distance threshold.

    Returns:
        list: List of coordinate series for each tracked point across frames.
    """
    all_matches = []
    point_tracks = {i: [tuple(series[0][i])] for i in range(len(series[0]))}  # Initialize tracking

    for i in range(len(series) - 1):
        points1, points2 = series[i], series[i + 1]

        # Compute pairwise Euclidean distance matrix
        cost_matrix = np.linalg.norm(points1[:, np.newaxis, :] - points2[np.newaxis, :, :], axis=2)

        # Solve assignment problem using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(row_ind, col_ind))

        # Compute distances of assigned pairs
        match_distances = [cost_matrix[i1, i2] for i1, i2 in matched_pairs]

        # Compute threshold
        distance_threshold = threshold_factor * np.mean(match_distances)

        # Filter out matches exceeding the threshold
        filtered_matches = [(i1, i2) for (i1, i2) in matched_pairs if cost_matrix[i1, i2] <= distance_threshold]

        all_matches.append(filtered_matches)

        # Update point tracks
        new_tracks = {}
        for i1, i2 in filtered_matches:
            if i1 in point_tracks:
                point_tracks[i1].append(tuple(points2[i2]))
                new_tracks[i2] = point_tracks[i1]  # Assign continuation to new index
            else:
                point_tracks[i2] = [tuple(points2[i2])]

        point_tracks = new_tracks  # Update tracking for next iteration
    return list(point_tracks.values()), all_matches


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

