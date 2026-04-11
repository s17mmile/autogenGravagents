import math
from typing import Dict, Tuple


def compute_a_cm_angles(F1: float, F2: float, F3: float,
                        theta1_deg: float, theta2_deg: float, theta3_deg: float,
                        m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> Dict[str, float]:
    """
    Compute center-of-mass acceleration for three external forces given their directions
    as angles (in degrees) from the +x axis.

    Returns a dictionary with:
      - F_net_x, F_net_y: net external force components (N)
      - a_x, a_y: center-of-mass acceleration components (m/s^2)
      - a_magnitude: |a_CM| (m/s^2)
      - a_direction_deg: direction of a_CM in degrees from +x
      - M_total: total mass (kg)
    """
    M_total = m1 + m2 + m3
    if M_total <= 0:
        raise ValueError("Total mass must be positive.")

    # Components of the external forces
    F_net_x = F1 * math.cos(math.radians(theta1_deg)) + \
              F2 * math.cos(math.radians(theta2_deg)) + \
              F3 * math.cos(math.radians(theta3_deg))

    F_net_y = F1 * math.sin(math.radians(theta1_deg)) + \
              F2 * math.sin(math.radians(theta2_deg)) + \
              F3 * math.sin(math.radians(theta3_deg))

    a_x = F_net_x / M_total
    a_y = F_net_y / M_total
    a_magnitude = math.hypot(a_x, a_y)
    a_direction_deg = math.degrees(math.atan2(a_y, a_x))

    return {
        'F_net_x': F_net_x,
        'F_net_y': F_net_y,
        'a_x': a_x,
        'a_y': a_y,
        'a_magnitude': a_magnitude,
        'a_direction_deg': a_direction_deg,
        'M_total': M_total
    }


def compute_a_cm_vectors(F1: float, F2: float, F3: float,
                         u1: Tuple[float, float], u2: Tuple[float, float], u3: Tuple[float, float],
                         m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> Dict[str, float]:
    """
    Compute center-of-mass acceleration given three unit vectors u1, u2, u3 describing the force directions.
    Each ui should be a 2-tuple (ux, uy) with ux^2 + uy^2 == 1 (unit vector).

    Returns a dictionary with the same keys as compute_a_cm_angles.
    """
    M_total = m1 + m2 + m3
    if M_total <= 0:
        raise ValueError("Total mass must be positive.")

    F_net_x = F1 * u1[0] + F2 * u2[0] + F3 * u3[0]
    F_net_y = F1 * u1[1] + F2 * u2[1] + F3 * u3[1]

    a_x = F_net_x / M_total
    a_y = F_net_y / M_total
    a_magnitude = math.hypot(a_x, a_y)
    a_direction_deg = math.degrees(math.atan2(a_y, a_x))

    return {
        'F_net_x': F_net_x,
        'F_net_y': F_net_y,
        'a_x': a_x,
        'a_y': a_y,
        'a_magnitude': a_magnitude,
        'a_direction_deg': a_direction_deg,
        'M_total': M_total
    }

if __name__ == "__main__":
    # Illustrative example using angles (the values match the earlier discussion)
    F1, F2, F3 = 6.0, 12.0, 14.0
    theta1_deg, theta2_deg, theta3_deg = 0.0, 90.0, 180.0

    res_angles = compute_a_cm_angles(F1, F2, F3, theta1_deg, theta2_deg, theta3_deg,
                                    m1=1.0, m2=1.0, m3=1.0)
    print("Using angles:")
    print("F_net_x = {:.6f} N".format(res_angles["F_net_x"]))
    print("F_net_y = {:.6f} N".format(res_angles["F_net_y"]))
    print("a_CM_x = {:.6f} m/s^2".format(res_angles["a_x"]))
    print("a_CM_y = {:.6f} m/s^2".format(res_angles["a_y"]))
    print("|a_CM| = {:.6f} m/s^2".format(res_angles["a_magnitude"]))
    print("theta_CM = {:.6f} degrees".format(res_angles["a_direction_deg"]))

    # Illustrative example using unit vectors
    u1, u2, u3 = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)
    res_vectors = compute_a_cm_vectors(F1, F2, F3, u1, u2, u3, m1=1.0, m2=1.0, m3=1.0)
    print("
Using unit vectors:")
    print("F_net_x = {:.6f} N".format(res_vectors["F_net_x"]))
    print("F_net_y = {:.6f} N".format(res_vectors["F_net_y"]))
    print("a_CM_x = {:.6f} m/s^2".format(res_vectors["a_x"]))
    print("a_CM_y = {:.6f} m/s^2".format(res_vectors["a_y"]))
    print("|a_CM| = {:.6f} m/s^2".format(res_vectors["a_magnitude"]))
    print("theta_CM = {:.6f} degrees".format(res_vectors["a_direction_deg"]))