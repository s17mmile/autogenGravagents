import numpy as np

# Constants
R = 8.314  # J/(mol·K)
T = 298  # K
D = 70400  # J/mol (dissociation energy)

# Given values
vibrational_frequency = 159  # cm^-1
B = 0.155  # cm^-1
h = 6.626e-34  # Planck's constant in J·s
c = 2.998e10  # speed of light in cm/s
k = 1.381e-23  # Boltzmann's constant in J/K

# Convert vibrational frequency to J
vibrational_energy = h * vibrational_frequency * c * 100  # Convert cm^-1 to J

# Calculate the vibrational contribution to entropy for Na2
S_vib_Na2 = R * (h * vibrational_frequency / (k * T)) * (np.exp(h * vibrational_frequency / (k * T)) / (np.exp(h * vibrational_frequency / (k * T)) - 1)**2)

# Entropy of Na2 (simplified)
S_Na2 = R * (5/2) + S_vib_Na2  # Including vibrational contribution

# Entropy of Na (considering degeneracy)
S_Na = R * (5/2) + R * np.log(2)  # Degeneracy factor included

# Calculate ΔS
Delta_S = 2 * S_Na - S_Na2

# Calculate ΔG using ΔG = ΔH - TΔS
Delta_G = D - T * Delta_S

# Calculate equilibrium constant K
K = np.exp(-Delta_G / (R * T))

# Output the equilibrium constant in scientific notation
print(f'Equilibrium constant K: {K:.2e}')