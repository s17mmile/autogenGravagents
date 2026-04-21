# filename: vibrational_frequency_cl35.py
import math

# Function to calculate fundamental vibrational frequency
def calculate_vibrational_frequency(force_constant, atomic_mass_u):
    """
    Calculate the fundamental vibrational frequency of a diatomic molecule.
    :param force_constant: Force constant k in N/m
    :param atomic_mass_u: Atomic mass in atomic mass units (u)
    :return: Vibrational frequency in Hz
    """
    # Conversion factor from atomic mass units to kilograms
    u_to_kg = 1.66054e-27

    # Calculate mass of one atom in kg
    mass_kg = atomic_mass_u * u_to_kg

    # Calculate reduced mass (kg) for two identical atoms
    reduced_mass = mass_kg / 2

    # Calculate vibrational frequency (Hz)
    vibrational_frequency = (1 / (2 * math.pi)) * math.sqrt(force_constant / reduced_mass)

    return vibrational_frequency

# Given values
force_constant = 319  # N/m
atomic_mass_u = 35  # atomic mass units for Cl-35

# Calculate frequency
frequency = calculate_vibrational_frequency(force_constant, atomic_mass_u)

print(f"Fundamental vibrational frequency: {frequency:.2e} Hz")
