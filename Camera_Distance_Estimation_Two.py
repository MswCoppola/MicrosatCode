import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
from Geometric_Three_Points import generate_quadrilaterals, corner_points
from Cuboid_Detection_Two import saved_final_two_face_cuboid  # Import saved plot from cuboid_determination_two
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

def main():
    """Estimate camera position vector from cuboid center of mass with X, Y in meters."""
    if saved_final_two_face_cuboid:
        extracted_faces = extract_faces_from_saved_plot(saved_final_two_face_cuboid)
        center_of_mass = calculate_center_of_mass(corner_points)
        print(f"📍 Center of Mass (X, Y in meters): {center_of_mass}")

        edges = []
        for face in extracted_faces:
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
        else:
            print("❌ Distance estimation failed.")
    else:
        print("❌ No saved two-face cuboid plot available to process.")

if __name__ == "__main__":
    main()
