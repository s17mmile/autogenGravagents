import math

def calculate_terminal_velocity(m, g, C_d, rho, diameter):
    """
    Calculate the terminal velocity of an object falling under gravity with air resistance.
    Parameters:
    m (float): Mass of the object in kg.
    g (float): Gravitational acceleration in m/s^2.
    C_d (float): Drag coefficient.
    rho (float): Air density in kg/m^3.
    diameter (float): Diameter of the object in meters.
    Returns:
    float: Terminal velocity in m/s.
    """
    # Cross-sectional area in m^2 (assuming spherical shape)
    A = math.pi * (diameter / 2) ** 2
    # Calculate terminal velocity using the quadratic model
    v_t = math.sqrt((2 * m * g) / (C_d * rho * A))
    return v_t

# Constants
mass = 0.5  # mass of the potato in kg

# Gravitational acceleration in m/s^2
g = 9.81

# Drag coefficient for a potato (approximate)
C_d = 0.6

# Air density in kg/m^3
rho = 1.225

# Diameter of the potato in meters
diameter = 0.1

# Calculate terminal velocity
terminal_velocity = calculate_terminal_velocity(mass, g, C_d, rho, diameter)

# Output the terminal velocity
print(f'The terminal velocity of the potato is {terminal_velocity:.2f} m/s')