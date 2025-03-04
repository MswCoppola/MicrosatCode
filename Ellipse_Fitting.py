import numpy as np
import matplotlib.pyplot as plt


def analyze_and_plot_ellipses(ellipse_data):
    """
    Given multiple sets of points defining ellipses, this function:
    1. Computes and plots ellipses from the given points.
    2. Finds the centers and eccentricities of each ellipse.
    3. Calculates the best-fit line through the ellipse centers.
    4. Computes the average eccentricity and viewing angle θ (arcsin(e_avg)).
    5. Plots everything: ellipses, centers, and the best-fit line.

    Returns:
    - (m, c): Slope and intercept of the best-fit line (y = mx + c).
    - theta: Viewing angle in degrees.
    """

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
        e = np.sqrt(1 - (b ** 2 / a ** 2))
        return a, b, e if e < 1 else (None, None, None)  # Filter out invalid ellipses

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
        a, b, e = ellipse_axes(coeffs)

        if center and a and b:
            print(f"Ellipse {idx + 1} Center: {center}")
            print(f"Ellipse {idx + 1} Semi-Major Axis (a): {a:.4f}, Semi-Minor Axis (b): {b:.4f}")
            print(f"Ellipse {idx + 1} Eccentricity: {e:.4f}")
            centers.append(center)
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
    else:
        m, c, theta = None, None, None

    # Plot settings
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Ellipses with Centers, Eccentricity & Best-Fit Line")
    plt.grid()
    plt.show()

    return (m, c), theta

"""
# Example usage
ellipse_data = [
    [(536, 270), (467, 270), (419, 273), (393, 271), (389, 269), (410, 270), (426, 271)],  # Ellipse 1
    [(573, 293), (429, 301), (423, 298), (385, 295), (367, 289), (381, 289), (409, 280)],  # Ellipse 2
    [(591, 412), (616, 420), (601, 424), (615, 433), (561, 408), (556, 412), (526, 409), (515, 408)],  # Ellipse 3
]
"""

best_fit_line, theta = analyze_and_plot_ellipses(ellipse_data)
print(f"Best-Fit Line: y = {best_fit_line[0]:.4f}x + {best_fit_line[1]:.4f}")
print(f"Viewing Angle (θ) = {theta:.4f}°")
