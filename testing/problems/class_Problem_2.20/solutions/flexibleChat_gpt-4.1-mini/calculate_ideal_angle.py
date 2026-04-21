# filename: calculate_ideal_angle.py
import math

def calculate_ideal_angle(range_m, muzzle_velocity, gravity=9.81):
    """Calculate the two possible angles of elevation (in degrees) to hit a target at a given range with no air resistance."""
    sin_2theta = (gravity * range_m) / (muzzle_velocity ** 2)
    if sin_2theta > 1 or sin_2theta < 0:
        raise ValueError("Target out of range for given muzzle velocity.")
    two_theta = math.asin(sin_2theta)
    theta_deg = math.degrees(two_theta / 2)
    return theta_deg, 90 - theta_deg

if __name__ == "__main__":
    range_target = 1000  # meters
    muzzle_velocity = 140  # m/s
    angle1, angle2 = calculate_ideal_angle(range_target, muzzle_velocity)
    print(f"Ideal projectile motion angles to hit {range_target} m target with muzzle velocity {muzzle_velocity} m/s are: {angle1:.2f} degrees and {angle2:.2f} degrees.")
