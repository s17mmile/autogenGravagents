# filename: calculate_oxygen_and_hemoglobin.py

# Constants
P_atm = 1.00  # Pressure in atm
V_liters = 0.18  # Volume of O2 in liters
R_L_atm_per_mol_K = 0.082057  # Ideal gas constant in L atm / (mol K)
T_K = 298  # Temperature in Kelvin

# Calculate moles of O2 using the ideal gas law: n = PV / RT
moles_O2 = (P_atm * V_liters) / (R_L_atm_per_mol_K * T_K)

# Avogadro's number (molecules per mole)
avogadro_number = 6.02214076e23  # molecules/mol

# Calculate total number of O2 molecules
num_O2_molecules = moles_O2 * avogadro_number

# Each hemoglobin molecule binds 4 O2 molecules
O2_binding_sites_per_hemoglobin = 4

# Calculate number of hemoglobin molecules required
num_hemoglobin_molecules = num_O2_molecules / O2_binding_sites_per_hemoglobin

# Prepare results with units and formatted output
result = {
    'moles_O2_per_liter_blood': moles_O2,  # mol
    'hemoglobin_molecules_required': num_hemoglobin_molecules  # molecules
}

# Optional: formatted print output for clarity
print(f"Moles of O2 per liter of blood: {moles_O2:.6e} mol")
print(f"Number of hemoglobin molecules required: {num_hemoglobin_molecules:.6e} molecules")

result
