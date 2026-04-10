import math

def calculate_angle(muzzle_velocity, range_target):
    g = 9.81  # acceleration due to gravity in m/s^2
    # Calculate the sine of double the angle
    sin_2theta = (range_target * g) / (muzzle_velocity ** 2)
    # Ensure sin_2theta is within valid range for arcsin
    if -1 <= sin_2theta <= 1:
        angle_radians = math.asin(sin_2theta) / 2
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees
    else:
        return None  # Invalid angle due to range exceeding limits

# Constants
muzzle_velocity = 140  # in m/s
range_target = 1000     # in meters

# Calculate the angle
angle = calculate_angle(muzzle_velocity, range_target)

# Output the result
if angle is not None:
    print(f'The angle required to hit the target without retardation is approximately {angle:.2f} degrees.')
else:
    print('The target is out of range for the given muzzle velocity.')