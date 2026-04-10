import math

# Constants
wavenumber = 2309  # in cm^-1
speed_of_light = 3.00e10  # in cm/s
mass_H = 1.00784  # in atomic mass units (u)
mass_I = 126.90447  # in atomic mass units (u)

# Step 1: Convert wavenumber to frequency (Hz)
f = wavenumber * speed_of_light  # in Hz

# Step 2: Convert masses from atomic mass units to kg (1 u = 1.66053906660e-27 kg)
mass_H_kg = mass_H * 1.66053906660e-27  # in kg
mass_I_kg = mass_I * 1.66053906660e-27  # in kg

# Calculate reduced mass
reduced_mass = (mass_H_kg * mass_I_kg) / (mass_H_kg + mass_I_kg)  # in kg

# Step 3: Calculate the force constant (k)
k = (2 * math.pi * f) ** 2 * reduced_mass  # in N/m

# Output the force constant
print(f'The force constant k for H^{127}I is: {k} N/m')