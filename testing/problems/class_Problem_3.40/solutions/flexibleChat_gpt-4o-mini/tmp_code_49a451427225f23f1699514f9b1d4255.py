import math

# Given data
m = 1000  # mass of the automobile in kg
k = 981000  # effective spring constant in N/m
b = 2000  # damping coefficient in Ns/m

# Calculate natural frequency
omega_n = math.sqrt(k / m)

# Previously calculated values
F_0 = 490.5  # peak driving force in N
v = 20 * 1000 / 3600  # speed in m/s
lambda_ = 0.2  # wavelength in meters
omega = (v / lambda_) * (2 * math.pi)  # angular frequency in rad/s

# Refine amplitude calculation
A = (F_0 / m) / math.sqrt((omega_n**2 - omega**2)**2 + (b/m * omega)**2)

# Output the results
omega_n, A