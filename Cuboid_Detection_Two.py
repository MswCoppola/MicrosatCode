import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from itertools import combinations
from Geometric_Three_Points import generate_quadrilaterals, corner_points
import numpy as np

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

# -------------------- Main Execution --------------------

valid_quads = generate_quadrilaterals(corner_points)
print(f"Loaded {len(valid_quads)} valid quadrilaterals from Geometric_Three_Points.")

# Step 1: Plot initial combinations
desired_combinations = plot_shared_edge_combinations(valid_quads)

# Step 2: Apply final filtering to select the best cuboid representation and save the plot
filter_final_combination(desired_combinations)

# Step 3: Show the saved final two-face cuboid plot
show_saved_two_face_cuboid()

# -------------------- End of Script --------------------

# Functions now comprehensively check for global edge overlaps and face intersections.
# Enhanced logic incorporates disjoint face validation and stricter edge uniqueness checks.
# The final two-face cuboid plot is saved and can be recalled using show_saved_two_face_cuboid().
