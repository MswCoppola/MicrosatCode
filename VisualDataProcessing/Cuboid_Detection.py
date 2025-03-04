import matplotlib.pyplot as plt
from itertools import combinations
from shapely.geometry import Polygon, Point, LineString
#from VisualDataProcessing.BaseFunctions.Geometric_Three_Points import generate_quadrilaterals, corner_points as all_corners

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

