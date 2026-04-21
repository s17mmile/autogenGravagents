# filename: coriolis_deflection_refined.py
import math

def coriolis_deflection(latitude_deg, elevation_deg, speed):
    """
    Calculate the lateral deflection of a projectile due to the Coriolis effect.

    Parameters:
    latitude_deg (float): Latitude in degrees (positive for North, negative for South).
    elevation_deg (float): Elevation angle in degrees (0 to 90).
    speed (float): Initial speed in m/s (positive).

    Returns:
    tuple: (deflection_magnitude_in_meters, direction_string)
    """
    # Input validation
    if not (-90 <= latitude_deg <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (0 <= elevation_deg <= 90):
        raise ValueError("Elevation angle must be between 0 and 90 degrees.")
    if speed <= 0:
        raise ValueError("Speed must be positive.")

    # Constants
    g = 9.81  # gravity in m/s^2
    omega = 7.2921e-5  # Earth's angular velocity in rad/s

    # Convert degrees to radians
    latitude = math.radians(latitude_deg)
    elevation = math.radians(elevation_deg)

    # Initial velocity components
    v0 = speed
    v_vertical = v0 * math.sin(elevation)
    v_horizontal = v0 * math.cos(elevation)

    # Time of flight (ignoring air resistance)
    t_flight = 2 * v_vertical / g

    # Coriolis parameter f = 2 * omega * sin(latitude)
    # Note: sin(latitude) is negative in southern hemisphere, which affects deflection direction
    f = 2 * omega * math.sin(latitude)

    # Calculate lateral deflection magnitude (meters)
    lateral_deflection = abs((1/3) * f * v_horizontal * t_flight**2)

    # Determine deflection direction
    # In southern hemisphere (latitude < 0), deflection is to the west
    # In northern hemisphere (latitude > 0), deflection is to the east
    if latitude_deg < 0:
        direction = 'west'
    elif latitude_deg > 0:
        direction = 'east'
    else:
        direction = 'none (equator)'

    return lateral_deflection, direction

# Given values
latitude_deg = -50  # 50 degrees South
elevation_deg = 37
speed = 800  # m/s

# Calculate deflection
deflection_magnitude, deflection_direction = coriolis_deflection(latitude_deg, elevation_deg, speed)

# Output the result
print(f'Lateral deflection due to Coriolis effect: {deflection_magnitude:.2f} meters to the {deflection_direction}')
