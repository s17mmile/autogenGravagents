import math

# Constants
molar_mass_cesium = 132.91  # g/mol
mass_loss_mg = 385  # mg
hole_diameter_mm = 0.50  # mm
hole_diameter_m = hole_diameter_mm / 1000  # convert to meters
R = 8.314  # J/(mol*K)
T_C = 500  # Celsius
T_K = T_C + 273.15  # convert to Kelvin

# Step 1: Convert mass loss to grams
mass_loss_grams = mass_loss_mg / 1000  # convert mg to g

# Step 2: Calculate number of moles lost
n = mass_loss_grams / molar_mass_cesium

# Step 3: Calculate area of the hole
A = math.pi * (hole_diameter_m / 2) ** 2

# Step 4: Calculate average velocity of gas molecules
M = molar_mass_cesium / 1000  # convert g/mol to kg/mol
v = math.sqrt((2 * R * T_K) / M)

# Step 5: Calculate volume of gas that effuses through the hole in 100 seconds
V = A * v * 100  # volume in m^3

# Step 6: Calculate vapor pressure using ideal gas law
P = (n * R * T_K) / V  # pressure in Pascals
P_kPa = P / 1000  # convert to kPa

# Output the vapor pressure
print(f'Vapor pressure of liquid caesium at 500 K: {P_kPa:.2f} kPa')