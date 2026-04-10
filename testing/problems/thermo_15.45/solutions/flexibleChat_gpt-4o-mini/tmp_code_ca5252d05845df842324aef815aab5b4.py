import numpy as np

# Constants
h = 6.626e-34  # Planck's constant (J·s)
R = 8.314  # Gas constant (J/(mol·K))
T = 298  # Temperature in Kelvin

# Given values
vibrational_frequency_cm = 560  # cm^-1
rotational_constant_cm = 0.244  # cm^-1
bond_length = 1.99e-10  # Bond length in meters (assumed for Cl2)

# Convert vibrational frequency to Hz
vibrational_frequency_Hz = vibrational_frequency_cm * 100 * 3.00e10

# Calculate reduced mass (mu) for Cl2
m_Cl = 35 * 1.66e-27  # Mass of Cl in kg
mu = m_Cl / 2  # Reduced mass for Cl2

# Calculate moment of inertia (I)
I = mu * (bond_length ** 2)

# Convert rotational constant to SI units
B = rotational_constant_cm * 100 * h / (8 * np.pi**2 * I)

# Calculate zero-point energy (ZPE)
E_ZPE = 0.5 * h * vibrational_frequency_Hz

# Calculate rotational energy contributions
E_rot = sum(B * J * (J + 1) for J in range(10))  # Summing over J=0 to J=9

# Calculate entropy contributions
S_vib = R * (1 + (h * vibrational_frequency_Hz) / (R * T))
S_rot = R * (1 + np.log(2 * np.pi * I * R * T / (h**2)))  # Simplified rotational entropy
S_total = S_vib + S_rot

# Calculate Gibbs energy
G = E_ZPE + E_rot - T * S_total

# Print the result with units
print(f'Molar standard Gibbs energy G: {G:.3e} J/mol')