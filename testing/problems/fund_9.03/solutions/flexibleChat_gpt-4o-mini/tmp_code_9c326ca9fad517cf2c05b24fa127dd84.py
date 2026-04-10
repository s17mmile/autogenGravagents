def calculate_acceleration(m1, m2, m3, F_net):
    """
    Calculate the acceleration of the center of mass (COM) for a system of particles.
    Parameters:
    m1 (float): Mass of the first particle in kg.
    m2 (float): Mass of the second particle in kg.
    m3 (float): Mass of the third particle in kg.
    F_net (float): Net force acting on the system in Newtons.
    Returns:
    float: Acceleration of the center of mass in m/s^2.
    """
    # Calculate total mass
    M = m1 + m2 + m3
    # Calculate acceleration of the center of mass
    a_COM = F_net / M
    return a_COM

# Constants for the masses of the particles in kg
m1 = 1.67e-27  # mass of proton in kg
m2 = 1.68e-27  # mass of neutron in kg
m3 = 9.11e-31   # mass of electron in kg

# Given net force
F_net = 32.0  # in Newtons

# Calculate acceleration
acceleration = calculate_acceleration(m1, m2, m3, F_net)

# Print the result
print(f'The acceleration of the center of mass is approximately {acceleration:.2e} m/s^2')