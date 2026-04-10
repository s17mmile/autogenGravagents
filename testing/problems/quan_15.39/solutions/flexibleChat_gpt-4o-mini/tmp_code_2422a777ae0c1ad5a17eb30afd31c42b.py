import numpy as np

# Constants
h = 6.626e-34  # Planck's constant in J·s
k = 1.381e-23  # Boltzmann's constant in J/K
R = 8.314      # Ideal gas constant in J/(mol·K)

# Given values
wavenumber = 900  # in cm^-1
T = 298          # Temperature in K

# Step 1: Convert wavenumber to frequency
c = 3.00e10  # Speed of light in cm/s
nu_s = wavenumber * c  # Frequency in Hz

# Step 2: Calculate theta_s
theta_s = (h * nu_s) / k

# Step 3: Calculate U_{m, vib}
# Check for numerical stability in the denominator
exp_term = np.exp(theta_s / T) - 1
if np.abs(exp_term) < 1e-10:  # Avoid division by a very small number
    U_m_vib = R * T * (theta_s / T)  # Use approximation for small theta_s/T
else:
    U_m_vib = R * (theta_s / exp_term)

# Print the result with units
print(f'U_m_vib: {U_m_vib:.2f} J/mol')