# filename: finite_proton_size_energy_shift.py
import scipy.constants as const

"""
Calculate the first-order perturbation energy shift in the hydrogen atom ground state
caused by the finite size of the proton, modeled as a uniformly charged sphere.
"""

# Proton radius in cm and conversion to meters
R_cm = 1e-13
R = R_cm * 1e-2  # cm to m

# Physical constants
 e = const.e  # elementary charge (C)
 epsilon_0 = const.epsilon_0  # vacuum permittivity (F/m)
 a0 = const.bohr_radius  # Bohr radius (m)

# Calculate energy shift (Joules)
Delta_E_Joule = (3 * e**2 * R**2) / (10 * const.pi * epsilon_0 * a0**3)

# Convert energy shift to electronvolts
Delta_E_eV = Delta_E_Joule / e

# Print the result
print(f"Energy shift due to finite proton size: {Delta_E_eV:.3e} eV")
