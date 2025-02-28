import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from itertools import combinations
import math

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

# Example corner points
corner_points = [(0,0), (2.123,2.123), (1.5089479078638484,2.7336927792554375), (-0.6123724356957946,0.6123724356957946), (0.000000,-0.500000), (2.121320,1.623), (-0.6123724356957946,0.112372)]

