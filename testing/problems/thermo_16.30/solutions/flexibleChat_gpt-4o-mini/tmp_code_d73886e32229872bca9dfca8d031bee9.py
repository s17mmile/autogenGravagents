import math

# Constants
P = 101325  # Pressure in Pa (1 atm)
R = 8.314  # Ideal gas constant in J/(mol K)
T = 298  # Temperature in K
M = 0.04401  # Molar mass of CO2 in kg/mol
Avogadro_number = 6.022e23  # molecules/mol

# Step 1: Calculate number density (n)
n = P / (R * T)  # in mol/m^3
n_molecules = n * Avogadro_number  # in molecules/m^3

# Step 2: Calculate average speed (v_bar)
v_bar = math.sqrt((8 * R * T) / (math.pi * M))  # in m/s

# Step 3: Collision cross-section (sigma)
sigma = 0.4e-10  # in m^2 (typical value for diatomic gases)

# Step 4: Calculate collisional frequency (Z)
Z = (1/4) * n_molecules * sigma * v_bar  # in collisions/s

# Print the result with units
print(f'Total collisional frequency for CO2 at 1 atm and 298 K: {Z:.2e} collisions/s')