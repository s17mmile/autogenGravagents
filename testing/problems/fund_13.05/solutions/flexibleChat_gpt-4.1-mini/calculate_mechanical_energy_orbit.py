# filename: calculate_mechanical_energy_orbit.py

def calculate_mechanical_energy(mass_ball, altitude_km):
    """Calculate the mechanical energy of an object in circular orbit around Earth."""
    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M = 5.972e24     # mass of Earth in kg
    R = 6.371e6      # radius of Earth in meters

    # Validate inputs
    if mass_ball <= 0 or altitude_km < 0:
        raise ValueError("Mass must be positive and altitude cannot be negative.")

    # Convert altitude from km to meters
    h = altitude_km * 1000

    # Calculate orbital radius
    r = R + h

    # Calculate mechanical energy
    E = - (G * M * mass_ball) / (2 * r)

    return E

# Given values
mass_ball = 7.20  # kg
altitude_km = 350  # km

energy = calculate_mechanical_energy(mass_ball, altitude_km)
print(f"Mechanical energy of the bowling ball in orbit: {energy:.2f} J")
