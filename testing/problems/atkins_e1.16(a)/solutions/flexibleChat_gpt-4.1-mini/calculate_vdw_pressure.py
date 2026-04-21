# filename: calculate_vdw_pressure.py

# Constants and given data
mass_N2_kg = 92.4  # mass of nitrogen in kg
molar_mass_N2_g_per_mol = 28.0134  # molar mass of nitrogen in g/mol
V_m3 = 1.000  # volume in cubic meters
T_K = 500  # temperature in Kelvin

# van der Waals constants for nitrogen given in dm^6 atm mol^-2 and dm^3 mol^-1
# a = 1.352 dm^6 atm mol^-2
# b = 0.0387 dm^3 mol^-1

a_dm6_atm_per_mol2 = 1.352
b_dm3_per_mol = 0.0387

# Convert van der Waals constants to m^3 and atm units
# 1 dm^3 = 1e-3 m^3
# Therefore, a in m^6 atm mol^-2 = a_dm6 * (1e-3)^2 = a_dm6 * 1e-6
# b in m^3 mol^-1 = b_dm3 * 1e-3

a_m6_atm_per_mol2 = a_dm6_atm_per_mol2 * 1e-6
b_m3_per_mol = b_dm3_per_mol * 1e-3

# Calculate number of moles n
# mass in g = mass in kg * 1000
mass_N2_g = mass_N2_kg * 1000
n_mol = mass_N2_g / molar_mass_N2_g_per_mol

# Ideal gas constant R in atm m^3 / mol K
R_atm_m3_per_mol_K = 0.082057 * 1e-3  # more precise value

# Check denominator to avoid division by zero or negative volume
denominator = V_m3 - n_mol * b_m3_per_mol
if denominator <= 0:
    raise ValueError(f"Invalid volume after accounting for b: V - n*b = {denominator} <= 0")

# Apply van der Waals equation to solve for pressure P
# (P + a n^2 / V^2)(V - n b) = n R T
# => P = (n R T) / (V - n b) - a n^2 / V^2

P_atm = (n_mol * R_atm_m3_per_mol_K * T_K) / denominator - a_m6_atm_per_mol2 * n_mol**2 / V_m3**2

# Output the pressure
print(f"Approximate pressure of nitrogen at {T_K} K: {P_atm:.2f} atm")

# The code calculates and prints the pressure in atm at the working temperature using the van der Waals equation.