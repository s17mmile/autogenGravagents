# filename: hcl_rotational_levels_improved.py
import numpy as np

# Constants
I = 2.6422e-47  # Moment of inertia in kg m^2
h = 6.62607015e-34  # Planck constant in J s
hbar = h / (2 * np.pi)  # Reduced Planck constant in J s
c = 299792458  # Speed of light in m/s


def rotational_energy(J, I):
    """Calculate rotational energy for quantum number J and moment of inertia I."""
    return (hbar**2 / (2 * I)) * J * (J + 1)  # Energy in Joules


def degeneracy(J):
    """Calculate degeneracy of rotational level J."""
    return 2 * J + 1


def energy_to_wavenumber(E):
    """Convert energy in Joules to wavenumber in cm^-1."""
    return E / (h * c * 100)  # 100 converts m^-1 to cm^-1


# Calculate energies and degeneracies for J=0 to 3
J_levels = np.array([0, 1, 2, 3])
energies = np.array([rotational_energy(J, I) for J in J_levels])
degeneracies = np.array([degeneracy(J) for J in J_levels])

# Calculate transition frequency J=0 to J=1
transition_freq = (energies[1] - energies[0]) / h  # in Hz

# Print results
for J, E, g in zip(J_levels, energies, degeneracies):
    wavenumber = energy_to_wavenumber(E)
    print(f"J={J}: Energy = {E:.3e} J, Degeneracy = {g}, Wavenumber = {wavenumber:.3f} cm^-1")

print(f"\nTransition frequency (J=0 to J=1): {transition_freq:.3e} Hz")

# Return values for further use if needed
energies, degeneracies, transition_freq
