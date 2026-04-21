# filename: rotational_partition_function_nof.py
import math

# Given rotational constants in cm^-1
A = 3.1752
B = 0.3951
C = 0.3505

# Temperature in Kelvin
T = 25 + 273.15  # 298.15 K

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant in J/K
h = 6.62607015e-34  # Planck constant in J*s
c = 2.99792458e10   # Speed of light in cm/s

# Calculate the term (k_B * T / (h * c))^(3/2)
term = (k_B * T / (h * c)) ** 1.5

# Calculate sqrt(pi / (A * B * C))
sqrt_term = math.sqrt(math.pi / (A * B * C))

# Rotational partition function
Q_rot = sqrt_term * term

print(f"Rotational partition function Q_rot at {T} K: {Q_rot:.4e}")
