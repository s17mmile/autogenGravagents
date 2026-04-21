# filename: calculate_hartree_energy_ev.py
import scipy.constants as const

# Define constants
m_e = const.m_e  # electron mass in kg

e = const.e  # elementary charge in coulombs

epsilon_0 = const.epsilon_0  # vacuum permittivity in F/m

hbar = const.hbar  # reduced Planck constant in J*s

# Calculate Hartree energy in joules
E_h_joules = (m_e * e**4) / (16 * (const.pi**2) * epsilon_0**2 * hbar**2)

# Convert Hartree energy to electron volts
E_h_eV = E_h_joules / const.e

print(f"Hartree energy in eV: {E_h_eV}")

# Return the value for further use if needed
E_h_eV