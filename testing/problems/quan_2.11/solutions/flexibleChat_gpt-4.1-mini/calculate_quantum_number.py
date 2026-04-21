# filename: calculate_quantum_number.py
import math
from scipy.constants import h

def calculate_quantum_number(mass_kg, speed_m_per_s, box_length_m):
    """Calculate the quantum number n for a particle in a 1D box.

    Args:
        mass_kg (float): Mass of the particle in kilograms (must be positive).
        speed_m_per_s (float): Speed of the particle in meters per second (must be positive).
        box_length_m (float): Length of the box in meters (must be positive).

    Returns:
        float: Quantum number n.
    """
    if mass_kg <= 0 or speed_m_per_s <= 0 or box_length_m <= 0:
        raise ValueError("Mass, speed, and box length must be positive values.")

    n = (math.sqrt(2) * mass_kg * box_length_m * speed_m_per_s) / h
    return n

# Given values
mass = 1.0e-3  # 1.0 g in kg
speed = 1.0e-2  # 1.0 cm/s in m/s
box_length = 1.0e-2  # 1.0 cm in m

quantum_number = calculate_quantum_number(mass, speed, box_length)
print(f"Quantum number n: {quantum_number:.3e}")
