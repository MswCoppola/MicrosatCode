import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
from Cuboid_Detection import cuboid_candidates  # Import detected cuboid candidates

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

# -------------------- Main Execution --------------------

if cuboid_candidates:
    for cuboid in cuboid_candidates:
        classified_faces = classify_faces(cuboid["faces"])
        print("Face areas and aspect ratios:")
        for face_label, face in classified_faces.items():
            print(f"{face_label}: Area = {compute_face_area(face):.2f}, Aspect Ratio = {compute_aspect_ratio(face):.2f}")
        plot_classified_faces(classified_faces)
else:
    print("No valid cuboid detected.")
