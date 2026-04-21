# filename: calculate_compression_factor.py

# Given data
p_atm = 327.6  # pressure in atm
T_K = 776.4    # temperature in K
rho_kg_m3 = 133.2  # mass density in kg/m^3

# Constants
M_g_mol = 18.02  # molar mass in g/mol
M_kg_mol = M_g_mol / 1000  # molar mass in kg/mol

# Universal gas constant in dm^3 atm K^-1 mol^-1
R = 0.082057

# Convert mass density to molar density (mol/dm^3)
# 1 m^3 = 1000 dm^3
molar_density_mol_dm3 = (rho_kg_m3 / M_kg_mol) / 1000  # mol/dm^3

# Calculate molar volume (dm^3/mol)
V_m = 1 / molar_density_mol_dm3

# Calculate ideal gas molar volume (dm^3/mol)
V_m_ideal = (R * T_K) / p_atm

# Calculate compression factor Z
Z = V_m / V_m_ideal

print(f"Compression factor Z = {Z:.4f}")

Z
