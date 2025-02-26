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

def main():
    """Estimate camera position vector and plot it in 3D using pre-identified faces."""
    if saved_final_two_face_cuboid and classified_faces:
        center_of_mass = calculate_center_of_mass(corner_points)
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

if __name__ == "__main__":
    main()
