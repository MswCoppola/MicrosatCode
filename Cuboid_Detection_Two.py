import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from itertools import combinations
import math

# -------------------- Geometry Functions --------------------

def sort_clockwise(points):
    """Sort points in clockwise order around the centroid."""
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

def is_rectangle(points, tolerance=1e-2):
    """Check if a quadrilateral is a rectangle using dot product for right angles."""
    def dot(a, b, c):
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        return ab[0] * bc[0] + ab[1] * bc[1]

    return all(abs(dot(points[i], points[(i + 1) % 4], points[(i + 2) % 4])) < tolerance for i in range(4))

def shared_edge(face1, face2):
    """Check if two quadrilaterals share exactly one full edge."""
    def get_edges(face):
        return [{face[i], face[(i + 1) % 4]} for i in range(4)]

    edges1 = get_edges(face1)
    edges2 = get_edges(face2)
    shared = sum(1 for e1 in edges1 for e2 in edges2 if e1 == e2)
    return shared == 1

def edges_do_not_cross(face1, face2):
    """Ensure edges do not cross unless they coincide."""
    def get_edges(face):
        return [LineString([face[i], face[(i + 1) % 4]]) for i in range(4)]

    edges1 = get_edges(face1)
    edges2 = get_edges(face2)

    for e1 in edges1:
        for e2 in edges2:
            if e1.crosses(e2):
                return False
    return True

# -------------------- Cuboid Identification --------------------

def identify_two_face_cuboids(valid_quads):
    """Identify combinations of two quadrilaterals forming a valid two-face cuboid."""
    cuboid_candidates = []
    for face1, face2 in combinations(valid_quads, 2):
        if shared_edge(face1, face2) and edges_do_not_cross(face1, face2):
            if is_rectangle(face1) and is_rectangle(face2):
                cuboid_candidates.append((face1, face2))
    return cuboid_candidates

# -------------------- Plotting --------------------

def plot_two_face_cuboids(cuboid_candidates):
    """Plot all identified two-face cuboids."""
    for idx, (face1, face2) in enumerate(cuboid_candidates):
        plt.figure(figsize=(8, 8))
        for face, color in zip([face1, face2], ['blue', 'orange']):
            x, y = zip(*face)
            plt.plot(x + (x[0],), y + (y[0],), color=color, marker='o', linestyle='-', linewidth=2)
        plt.title(f'Two-Face Cuboid Candidate {idx + 1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

# -------------------- Main Execution --------------------

corner_points = [(0, 0), (3, 0), (0.333, -1), (2.6666, -1), (0.333, 3), (2.6666, 3)]

valid_quads = []
for quad in combinations(corner_points, 4):
    sorted_quad = sort_clockwise(list(quad))
    polygon = Polygon(sorted_quad)
    if polygon.is_valid and polygon.area > 0:
        valid_quads.append(sorted_quad)

print(f"✅ Generated {len(valid_quads)} valid quadrilaterals from Geometric_Three_Points.")

cuboid_candidates = identify_two_face_cuboids(valid_quads)
print(f"✅ Identified {len(cuboid_candidates)} possible two-face cuboids.")

if cuboid_candidates:
    plot_two_face_cuboids(cuboid_candidates)
else:
    print("❌ No valid two-face cuboids found.")
