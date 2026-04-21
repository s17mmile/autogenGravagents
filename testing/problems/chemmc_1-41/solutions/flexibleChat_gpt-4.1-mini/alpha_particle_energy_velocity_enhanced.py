# filename: alpha_particle_energy_velocity_enhanced.py

from math import sqrt
from scipy.constants import elementary_charge


def calculate_energy(potential_difference_volts: float, charge_coulombs: float) -> float:
    """Calculate the energy gained by a charged particle falling through a potential difference.

    Args:
        potential_difference_volts (float): Potential difference in volts.
        charge_coulombs (float): Charge of the particle in coulombs.

    Returns:
        float: Energy gained in joules.
    """
    if potential_difference_volts <= 0:
        raise ValueError("Potential difference must be positive.")
    if charge_coulombs <= 0:
        raise ValueError("Charge must be positive.")
    return charge_coulombs * potential_difference_volts


def calculate_velocity(energy_joules: float, mass_kg: float) -> float:
    """Calculate velocity of a particle given kinetic energy and mass.

    Args:
        energy_joules (float): Kinetic energy in joules.
        mass_kg (float): Mass of the particle in kilograms.

    Returns:
        float: Velocity in meters per second.
    """
    if energy_joules <= 0:
        raise ValueError("Energy must be positive.")
    if mass_kg <= 0:
        raise ValueError("Mass must be positive.")
    return sqrt(2 * energy_joules / mass_kg)


def main():
    potential_difference = 4.0  # volts
    mass_alpha = 6.64e-27  # kg
    charge_alpha = 2 * elementary_charge  # coulombs

    energy = calculate_energy(potential_difference, charge_alpha)
    velocity = calculate_velocity(energy, mass_alpha)

    print(f"Energy gained by alpha particle: {energy:.3e} J")
    print(f"Velocity of alpha particle: {velocity:.3e} m/s")


if __name__ == '__main__':
    main()
