# filename: shortest_path_cone_improved.py
import numpy as np

def shortest_path_on_cone(point1, point2, apex_z=1.0):
    """
    Calculate the shortest path length between two points on the conical surface
    defined by z = apex_z - sqrt(x^2 + y^2).

    Parameters:
    - point1, point2: numpy arrays or lists with (x, y, z) coordinates.
    - apex_z: z-coordinate of the cone apex (default 1.0).

    Returns:
    - Length of the shortest path along the cone surface.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Check that points lie on the cone surface (within a tolerance)
    def on_cone(p):
        r = np.sqrt(p[0]**2 + p[1]**2)
        return np.isclose(p[2], apex_z - r, atol=1e-8)

    if not (on_cone(point1) and on_cone(point2)):
        raise ValueError("One or both points do not lie on the conical surface.")

    # Radius of base circle (where z=0)
    base_radius = apex_z  # since z = apex_z - r, at z=0 => r=apex_z

    # Slant height of the cone (distance from apex to base edge)
    s = np.sqrt(apex_z**2 + base_radius**2)  # sqrt((apex_z - 0)^2 + (0 - base_radius)^2)

    # Sector angle of the unfolded cone (arc length = base circumference)
    theta = 2 * np.pi * base_radius / s  # sector angle in radians

    # Compute angles of points around the cone axis (z-axis)
    angle1 = np.arctan2(point1[1], point1[0])
    angle2 = np.arctan2(point2[1], point2[0])

    # Difference in angle on the base circle
    delta_angle = np.abs(angle2 - angle1)
    # Ensure the angle is between 0 and pi (shortest arc)
    if delta_angle > np.pi:
        delta_angle = 2 * np.pi - delta_angle

    # Scale the angle to the sector angle
    alpha = delta_angle * (theta / (2 * np.pi))

    # Length of the shortest path (chord length on sector circle)
    L = 2 * s * np.sin(alpha / 2)

    return L

# Example usage with given points
p1 = (0, -1, 0)
p2 = (0, 1, 0)

length = shortest_path_on_cone(p1, p2)
print(f"Shortest path length on the conical surface: {length:.4f} units")

# Expected output: approximately 2.53 units
