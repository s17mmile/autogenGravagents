# filename: calculate_angle_phi.py
import numpy as np

def calculate_angle_between_vectors(a, b, degrees=True):
    """
    Calculate the angle between two vectors a and b.

    Parameters:
    a (array-like): First vector.
    b (array-like): Second vector.
    degrees (bool): If True, return angle in degrees; otherwise in radians.

    Returns:
    float: Angle between vectors in specified units.
    """
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("One or both vectors have zero magnitude, angle is undefined.")
    dot_product = np.dot(a, b)
    cos_phi = dot_product / (magnitude_a * magnitude_b)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)  # Numerical safety
    phi_rad = np.arccos(cos_phi)
    if degrees:
        return np.degrees(phi_rad)
    else:
        return phi_rad

if __name__ == "__main__":
    vec_a = np.array([3.0, -4.0, 0.0])
    vec_b = np.array([-2.0, 0.0, 3.0])
    angle_phi = calculate_angle_between_vectors(vec_a, vec_b)
    with open('angle_phi_result.txt', 'w') as f:
        f.write(f'Angle phi between vectors a and b: {angle_phi:.4f} degrees\n')
