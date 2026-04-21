# filename: calculate_dipole_potential_energy.py
import numpy as np

def calculate_potential_energy(p_vector, E_vector):
    """Calculate the potential energy of an electric dipole in an electric field.

    Parameters:
    p_vector (np.array): Dipole moment vector in C*m
    E_vector (np.array): Electric field vector in N/C

    Returns:
    float: Potential energy in Joules
    """
    if p_vector.shape != E_vector.shape:
        raise ValueError("Dipole moment and electric field vectors must have the same dimensions.")
    return -np.dot(p_vector, E_vector)

# Given values
p_magnitude = 1.24e-30  # C*m
p_vector = np.array([3.00, 4.00]) * p_magnitude  # Dipole moment vector in C*m
E_vector = np.array([4000, 0])  # Electric field vector in N/C

# Calculate potential energy
U = calculate_potential_energy(p_vector, E_vector)

print(f"Potential energy of the electric dipole: {U:.3e} J")
