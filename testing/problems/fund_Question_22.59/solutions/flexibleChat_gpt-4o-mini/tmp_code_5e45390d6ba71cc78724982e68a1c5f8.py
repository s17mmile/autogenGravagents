import numpy as np

# Constants
E = 46.0  # Electric field in N/C
p = 3.02e-25  # Dipole moment in C*m
theta_i_deg = 64  # Initial angle in degrees
theta_f_deg = 180  # Final angle in degrees

# Convert angles from degrees to radians
theta_i = np.radians(theta_i_deg)
theta_f = np.radians(theta_f_deg)

# Calculate initial potential energy U_i
U_i = -p * E * np.cos(theta_i)

# Calculate final potential energy U_f
U_f = -p * E * np.cos(theta_f)

# Calculate work done W
W = U_f - U_i

# Output the result
print(f'The work required to turn the dipole is {W:.2e} J')