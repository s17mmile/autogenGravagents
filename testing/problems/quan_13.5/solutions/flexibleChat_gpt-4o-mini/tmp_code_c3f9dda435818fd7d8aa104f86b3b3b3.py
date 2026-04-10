import numpy as np

# Constants
D0 = 2.4287  # eV
nu_e_cm = 1405.65  # cm^-1
nu_ex_e_cm = 23.20  # cm^-1
h = 4.135667696e-15  # eV*s (Planck's constant)

# Speed of light in cm/s
c = 2.99792458e10  # cm/s

# Convert nu_e from cm^-1 to energy in eV
nu_e_energy = h * c * nu_e_cm

# Calculate D_e using the relationship D_e = D_0 + h * nu_e * x_e
D_e = D0 + nu_e_energy + h * nu_ex_e_cm

# Output the result
print(f'D_e = {D_e} eV')