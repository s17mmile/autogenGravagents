# filename: calculate_neutralizing_charge.py
import math

def calculate_neutralizing_charge() -> float:
    """Calculate the equal positive charge needed on Earth and Moon to neutralize gravitational attraction.

    Returns:
        float: Charge in Coulombs
    """
    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    k = 8.9875517923e9  # Coulomb's constant in N m^2 C^-2
    M_e = 5.972e24  # mass of Earth in kg
    M_m = 7.348e22  # mass of Moon in kg

    # Calculate charge q such that electrostatic force equals gravitational force
    q = math.sqrt((G * M_e * M_m) / k)
    return q

if __name__ == '__main__':
    charge = calculate_neutralizing_charge()
    print(f'Charge needed on each body to neutralize gravitational attraction: {charge:.3e} Coulombs')
