# filename: calculate_force_constant_HI.py

import math

# Constants
c = 3.00e10  # speed of light in cm/s
wavenumber = 2309  # in cm^-1

# Atomic masses in atomic mass units (u)
mass_H = 1.0078
mass_I = 126.90447

# Atomic mass unit in kg
u_to_kg = 1.66054e-27

# Calculate reduced mass in atomic mass units
reduced_mass_u = (mass_H * mass_I) / (mass_H + mass_I)

# Convert reduced mass to kg
reduced_mass_kg = reduced_mass_u * u_to_kg

# Convert wavenumber to frequency in Hz
frequency = wavenumber * c  # Hz

# Calculate force constant k in N/m using the formula k = (2 * pi * frequency)^2 * reduced_mass
k = (2 * math.pi * frequency)**2 * reduced_mass_kg

print(f"Force constant k of H127I: {k:.2f} N/m")
