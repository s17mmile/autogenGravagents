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

# Step 1: Calculate the vibrational contribution to entropy for Na2
vibrational_energy = h * vibrational_frequency * c * 100  # Convert cm^-1 to J
S_vib_Na2 = R * (h * vibrational_frequency / (k * T)) * (np.exp(h * vibrational_frequency / (k * T)) / (np.exp(h * vibrational_frequency / (k * T)) - 1)**2)

# Step 2: Calculate the rotational contribution to entropy for Na2
S_rot_Na2 = R * np.log(T / B)

# Total entropy for Na2
S_Na2 = S_vib_Na2 + S_rot_Na2

# Step 3: Calculate the entropy for sodium atoms (Na)
S_Na = R * (5/2) + R * np.log(2)  # Including degeneracy factor

# Step 4: Calculate ΔS
Delta_S = 2 * S_Na - S_Na2

# Step 5: Calculate ΔG using ΔG = ΔH - TΔS
Delta_G = D - T * Delta_S

# Step 6: Calculate equilibrium constant K
K = np.exp(-Delta_G / (R * T))

# Output the equilibrium constant in scientific notation
print(f'Equilibrium constant K: {K:.2e}')