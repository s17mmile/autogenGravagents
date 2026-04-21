# filename: calculate_projectile_range.py
import math

def calculate_range(v0, theta_degrees):
    if v0 <= 0:
        raise ValueError("Muzzle velocity must be positive.")
    if not (0 < theta_degrees < 90):
        raise ValueError("Angle of elevation must be between 0 and 90 degrees.")
    g = 9.81  # acceleration due to gravity in m/s^2
    theta_radians = math.radians(theta_degrees)
    range_ = (v0 ** 2) * math.sin(2 * theta_radians) / g
    return range_

if __name__ == '__main__':
    # Given values
    v0 = 1450  # muzzle velocity in m/s
    theta = 55  # angle of elevation in degrees

    predicted_range = calculate_range(v0, theta)
    print(f"Predicted range of the projectile: {predicted_range:.2f} meters")
