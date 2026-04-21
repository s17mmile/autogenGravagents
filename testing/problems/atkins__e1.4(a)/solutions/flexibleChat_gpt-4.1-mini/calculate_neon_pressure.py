# filename: calculate_neon_pressure.py

# Given data
mass_milligrams = 255  # mass in milligrams
volume_dm3 = 3.00  # volume in decimeters cubed
temperature_K = 122  # temperature in kelvin

# Constants
molar_mass_neon_g_per_mol = 20.18  # g/mol
R = 8.314  # J/(mol*K)

# Convert units
mass_grams = mass_milligrams * 1e-3  # mg to g
volume_m3 = volume_dm3 * 1e-3  # dm^3 to m^3

# Calculate number of moles
number_of_moles = mass_grams / molar_mass_neon_g_per_mol

# Calculate pressure using ideal gas law P = nRT / V
pressure_pascals = (number_of_moles * R * temperature_K) / volume_m3

# Convert pressure to atmospheres
pressure_atm = pressure_pascals / 101325

# Output results
print(f"Pressure of neon gas sample:")
print(f"{pressure_pascals:.3f} pascals (Pa)")
print(f"{pressure_atm:.3f} atmospheres (atm)")
