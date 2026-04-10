import math

def calculate_charge(M_earth, M_moon, R):
    # Constants
    gravitational_constant = 6.674 * 10**-11  # Gravitational constant (N m^2/kg^2)
    coulombs_constant = 8.9875 * 10**9  # Coulomb's constant (N m^2/C^2)

    # Step 1: Calculate gravitational force
    F_g = gravitational_constant * (M_earth * M_moon) / R**2

    # Step 2: Calculate charge required to neutralize gravitational force
    if R == 0:
        raise ValueError('Distance R cannot be zero.')  # Error handling for zero distance
    q_squared = (F_g * R**2) / coulombs_constant
    q = math.sqrt(q_squared)

    return q

# Constants for Earth and Moon
M_earth = 5.972 * 10**24  # Mass of Earth (kg)
M_moon = 7.342 * 10**22  # Mass of Moon (kg)
R = 3.844 * 10**8  # Distance between Earth and Moon (m)

# Calculate the charge required
charge_required = calculate_charge(M_earth, M_moon, R)
print(f'The charge required on both Earth and Moon is approximately {charge_required:.2e} C')