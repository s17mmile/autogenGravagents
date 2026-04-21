# filename: calculate_initial_speed.py
import math

def calculate_initial_speed(range_meters, angle_degrees, gravity=9.8):
    """
    Calculate the initial speed of a projectile given the range, launch angle, and gravity.
    Parameters:
        range_meters (float): The horizontal distance the projectile travels (meters).
        angle_degrees (float): The launch angle in degrees (0 < angle < 90).
        gravity (float): Acceleration due to gravity (default 9.8 m/s^2).
    Returns:
        float: The initial speed in meters per second.
    Raises:
        ValueError: If inputs are out of expected ranges.
    """
    if not (0 < angle_degrees < 90):
        raise ValueError("Angle must be between 0 and 90 degrees (exclusive).")
    if range_meters <= 0:
        raise ValueError("Range must be a positive value.")
    if gravity <= 0:
        raise ValueError("Gravity must be a positive value.")

    angle_radians = math.radians(angle_degrees)
    sin_2theta = math.sin(2 * angle_radians)
    if sin_2theta == 0:
        raise ValueError("Sin of double the angle is zero, invalid for calculation.")

    initial_speed = math.sqrt(range_meters * gravity / sin_2theta)
    return initial_speed

# Given values
range_distance = 90  # meters
launch_angle = 45    # degrees

initial_speed = calculate_initial_speed(range_distance, launch_angle)
print(f"Initial speed of the ball: {initial_speed:.2f} m/s")
