import numpy as np
from shapely.geometry import LineString
from Geometric_Three_Points import corner_points
from Cuboid_Detection import cuboid_candidates
import matplotlib.pyplot as plt
import os

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


def main():
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


if __name__ == "__main__":
    main()
