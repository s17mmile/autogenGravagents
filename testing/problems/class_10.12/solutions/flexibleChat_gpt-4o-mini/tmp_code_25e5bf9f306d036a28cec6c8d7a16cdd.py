import math

# Constants
EARTH_RADIUS = 6371000  # Radius of Earth in meters
ANGULAR_VELOCITY = 7.272e-5  # Angular velocity of Earth in radians per second
GRAVITY = 9.80665  # Standard gravity in m/s^2


def calculate_deviation(latitude):
    """
    Calculate the maximum angular deviation of a plumb line from the true vertical
    at a given latitude.
    
    Parameters:
    latitude (float): Latitude in degrees.
    
    Returns:
    float: Angular deviation in seconds of arc.
    """
    # Convert latitude to radians
    lambda_rad = math.radians(latitude)
    
    # Calculate the effective gravitational force denominator
    denominator = GRAVITY - EARTH_RADIUS * ANGULAR_VELOCITY**2 * (math.cos(lambda_rad)**2)
    
    # Check for division by zero
    if denominator == 0:
        raise ValueError('Denominator for effective gravity calculation is zero.')
    
    # Calculate the angular deviation epsilon in radians
    epsilon_rad = (EARTH_RADIUS * ANGULAR_VELOCITY**2 * math.sin(lambda_rad) * math.cos(lambda_rad)) / denominator
    
    # Convert epsilon from radians to seconds of arc
    epsilon_seconds_of_arc = epsilon_rad * (180 * 3600 / math.pi)
    
    return epsilon_seconds_of_arc

# Example usage
latitude = 45  # Latitude in degrees
try:
    deviation = calculate_deviation(latitude)
    print(f'Maximum angular deviation in seconds of arc: {deviation}')
except ValueError as e:
    print(e)