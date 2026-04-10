import numpy as np

# Constants
R = 8.314  # J/(mol·K)

# Given data
wavenumber = 450.0  # cm^-1
T = 298  # K
h = 6.626e-34  # J·s
k = 1.381e-23  # J/K

# Step 1: Convert wavenumber to frequency
c = 2.998e10  # cm/s
nu = wavenumber * c  # Hz

# Step 2: Calculate h * nu / (k * T)
hnut = (h * nu) / (k * T)

# Step 3: Calculate vibrational entropy contribution
S_vib = R * (hnut) * (1 / (np.exp(hnut) - 1) + np.log(1 - np.exp(-hnut)))

# Step 4: Electronic contribution calculation
Delta_E1 = 1.609 * 1.602e-19  # J
Delta_E2 = 1.702 * 1.602e-19  # J

# Calculate populations of excited states
p_1 = np.exp(-Delta_E1 / (k * T))
p_2 = np.exp(-Delta_E2 / (k * T))

# Ground state population
p_0 = 1.0

# Step 5: Calculate electronic entropy contribution
# Adding a small epsilon to avoid log(0)
epsilon = 1e-10
S_elec = -R * (p_0 * np.log(p_0) + (p_1 + epsilon) * np.log(p_1 + epsilon) + (p_2 + epsilon) * np.log(p_2 + epsilon))

# Step 6: Total entropy
S_total = S_vib + S_elec

# Print the result with context
print(f'Standard molar entropy of F2- at 298 K: {S_total:.2f} J/(mol·K)')