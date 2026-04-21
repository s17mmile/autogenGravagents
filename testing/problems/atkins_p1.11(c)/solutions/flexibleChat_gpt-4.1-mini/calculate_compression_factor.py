# filename: calculate_compression_factor.py

# Given constants
p_atm = 327.6  # pressure in atm
T_K = 776.4    # temperature in K
rho_kg_m3 = 133.2  # density in kg/m^3
# Critical constants (not used in calculation)
T_c = 647.4    # critical temperature in K
p_c = 218.3    # critical pressure in atm

a_dm6_atm_mol2 = 5.464  # van der Waals constant a in dm^6 atm mol^-2
b_dm3_mol = 0.03049     # van der Waals constant b in dm^3 mol^-1
M_g_mol = 18.02         # molar mass in g/mol

# Constants
R_atm_dm3_mol_K = 0.082057366080960  # Ideal gas constant in atm dm^3 mol^-1 K^-1

# Convert molar mass to kg/mol
M_kg_mol = M_g_mol / 1000.0

# Convert density from kg/m^3 to g/dm^3 (1 kg/m^3 = 1 g/dm^3)
rho_g_dm3 = rho_kg_m3  # same numeric value

# Calculate molar volume Vm in dm^3/mol
Vm_dm3_mol = M_g_mol / rho_g_dm3  # g/mol divided by g/dm^3 = dm^3/mol

# Calculate B (second virial coefficient) in dm^3/mol
B_dm3_mol = b_dm3_mol - (a_dm6_atm_mol2 / (R_atm_dm3_mol_K * T_K))

# Calculate compression factor Z using virial expansion approximation: Z = 1 + (B * p) / (R * T)
Z = 1 + (B_dm3_mol * p_atm) / (R_atm_dm3_mol_K * T_K)

# Output the result
print(f"Compression factor Z from virial expansion: {Z:.5f}")
