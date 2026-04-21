# filename: skier_velocity_calculation.py
import math

def calculate_skier_velocity(incline_angle_deg, distance, mu_k):
    """
    Calculate the velocity of a skier at the bottom of an inclined hill considering kinetic friction.
    Parameters:
        incline_angle_deg (float): Incline angle in degrees.
        distance (float): Distance traveled down the hill in meters.
        mu_k (float): Coefficient of kinetic friction.
    Returns:
        float: Velocity at the bottom of the hill in m/s.
    Raises:
        ValueError: If distance or mu_k is negative, or if net acceleration is non-positive.
    """
    if distance < 0:
        raise ValueError("Distance must be non-negative.")
    if mu_k < 0:
        raise ValueError("Coefficient of kinetic friction must be non-negative.")

    g = 9.81  # acceleration due to gravity in m/s^2
    theta = math.radians(incline_angle_deg)  # convert angle to radians

    # Calculate net acceleration down the slope
    a = g * (math.sin(theta) - mu_k * math.cos(theta))

    if a <= 0:
        raise ValueError("Net acceleration is zero or negative; the skier will not accelerate down the hill.")

    u = 0  # initial velocity
    v = math.sqrt(u**2 + 2 * a * distance)
    return v

# Given values
incline_angle_deg = 17  # degrees
distance = 100  # meters
mu_k = 0.18  # coefficient of kinetic friction

velocity_bottom = calculate_skier_velocity(incline_angle_deg, distance, mu_k)

# Save the result to a text file
with open('skier_velocity.txt', 'w') as f:
    f.write(f'Velocity at the bottom of the hill: {velocity_bottom:.2f} m/s\n')
