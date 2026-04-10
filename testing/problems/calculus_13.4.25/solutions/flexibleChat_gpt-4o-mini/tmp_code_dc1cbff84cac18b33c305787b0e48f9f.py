import math

def calculate_initial_speed(range_distance, angle_degrees, gravity):
    """
    Calculate the initial speed of a projectile given its range, launch angle, and gravitational acceleration.
    
    Parameters:
    range_distance (float): The horizontal distance the projectile travels (in meters).
    angle_degrees (float): The launch angle (in degrees).
    gravity (float): The acceleration due to gravity (in m/s^2).
    
    Returns:
    float: The initial speed of the projectile (in m/s).
    """
    angle_radians = math.radians(angle_degrees)
    sin_2theta = math.sin(2 * angle_radians)
    if sin_2theta == 0:
        raise ValueError('Angle must be between 0 and 90 degrees for projectile motion.')
    initial_speed_squared = (range_distance * gravity) / sin_2theta
    initial_speed = math.sqrt(initial_speed_squared)
    return initial_speed

# Constants
range_distance = 90  # in meters
angle_degrees = 45  # in degrees
gravity = 9.81  # in m/s^2

# Calculate initial speed
initial_speed = calculate_initial_speed(range_distance, angle_degrees, gravity)
print(f'The initial speed of the ball is approximately {initial_speed:.2f} m/s')