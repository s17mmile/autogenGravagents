import math

# Constants
v_d1 = 14.44  # Final speed of deuteron in km/s (first set)
v_n1 = 5.18   # Final speed of neutron in km/s (first set)
v_d2 = 5.12   # Final speed of deuteron in km/s (second set)
v_n2 = 19.79  # Final speed of neutron in km/s (second set)

# Initial speed of deuteron
initial_deuteron_speed = 14.9  # Initial speed of deuteron in km/s

# Function to calculate maximum scattering angle

def calculate_max_angle(v_d, v_n, initial_speed, psi_deg):
    """
    Calculate the maximum scattering angle for a deuteron after an elastic collision.
    
    Parameters:
    v_d (float): Final speed of the deuteron in km/s.
    v_n (float): Final speed of the neutron in km/s.
    initial_speed (float): Initial speed of the deuteron in km/s.
    psi_deg (float): Scattering angle in degrees.
    
    Returns:
    float: Maximum scattering angle in degrees or None if invalid.
    """
    psi_rad = math.radians(psi_deg)  # Convert angle to radians
    sin_phi = (2 * v_d * math.sin(psi_rad)) / v_n
    
    # Check if sin_phi is within valid range
    if sin_phi < -1 or sin_phi > 1:
        return None  # or handle it as needed
    
    # Calculate phi in radians
    phi_rad = math.asin(sin_phi)
    # Convert phi to degrees
    phi_deg = math.degrees(phi_rad)
    return phi_deg

# Calculate maximum scattering angles for both sets of speeds
max_angle1 = calculate_max_angle(v_d1, v_n1, initial_deuteron_speed, 10)
max_angle2 = calculate_max_angle(v_d2, v_n2, initial_deuteron_speed, 10)

# Output results
print(f'Maximum scattering angle for first set: {max_angle1:.2f} degrees')
print(f'Maximum scattering angle for second set: {max_angle2:.2f} degrees')