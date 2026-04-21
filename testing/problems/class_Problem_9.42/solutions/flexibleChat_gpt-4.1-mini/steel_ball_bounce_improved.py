# filename: steel_ball_bounce_improved.py
import math

def calculate_bounce_velocity(v, theta_deg, e):
    """
    Calculate the rebound velocity magnitude and angle of a steel ball bouncing off a smooth, heavy steel plate.
    
    Parameters:
    v (float): Initial velocity magnitude (m/s), must be >= 0
    theta_deg (float): Angle of incidence from the normal (degrees), between 0 and 90
    e (float): Coefficient of restitution, between 0 and 1
    
    Returns:
    tuple: (velocity_after (float), angle_after_deg (float))
        velocity_after: Magnitude of velocity after bounce (m/s)
        angle_after_deg: Angle from the normal after bounce (degrees)
    """
    # Input validation
    if v < 0:
        raise ValueError('Initial velocity must be non-negative')
    if not (0 <= theta_deg <= 90):
        raise ValueError('Angle of incidence must be between 0 and 90 degrees')
    if not (0 <= e <= 1):
        raise ValueError('Coefficient of restitution must be between 0 and 1')

    # Convert angle from degrees to radians
    theta = math.radians(theta_deg)
    
    # Decompose velocity into normal and tangential components
    v_n = v * math.cos(theta)
    v_t = v * math.sin(theta)
    
    # Calculate normal component after bounce using coefficient of restitution
    v_n_after = -e * v_n
    
    # Tangential component remains unchanged (smooth plate)
    v_t_after = v_t
    
    # Calculate resultant velocity magnitude after bounce
    v_after = math.sqrt(v_n_after**2 + v_t_after**2)
    
    # Calculate rebound angle from the normal in degrees using atan2 for robustness
    phi_rad = math.atan2(abs(v_t_after), abs(v_n_after))
    phi_deg = math.degrees(phi_rad)
    
    return v_after, phi_deg

# Example usage with given values
initial_velocity = 5  # m/s
angle_of_incidence = 30  # degrees
coefficient_of_restitution = 0.8

velocity_after, angle_after = calculate_bounce_velocity(initial_velocity, angle_of_incidence, coefficient_of_restitution)

# Save results to a file
with open('bounce_results.txt', 'w') as f:
    f.write(f'Rebound velocity magnitude: {velocity_after:.2f} m/s\n')
    f.write(f'Rebound angle from normal: {angle_after:.2f} degrees\n')
