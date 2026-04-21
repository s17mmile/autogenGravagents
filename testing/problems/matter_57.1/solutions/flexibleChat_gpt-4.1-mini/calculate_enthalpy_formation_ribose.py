# filename: calculate_enthalpy_formation_ribose.py

# Constants
R = 8.314  # J/(mol K), gas constant
T = 298.15  # K, assumed temperature for calculations

# Given data
mass_ribose = 0.727  # g
mass_benzoic_acid = 0.825  # g
delta_T_ribose = 0.910  # K
delta_T_benzoic = 1.940  # K

# Benzoic acid data
M_benzoic_acid = 122.12  # g/mol
delta_U_benzoic = -3251e3  # J/mol (internal energy of combustion)

# Standard enthalpies of formation (kJ/mol)
# CO2 (g) and H2O (l) values from standard tables
delta_Hf_CO2 = -393.5  # kJ/mol
delta_Hf_H2O = -285.8  # kJ/mol

# Molecular formula of D-ribose: C5H10O5
M_ribose = 150.13  # g/mol

# Step 1: Calculate calorimeter heat capacity (C_cal)
moles_benzoic = mass_benzoic_acid / M_benzoic_acid
Q_benzoic = moles_benzoic * delta_U_benzoic  # J
C_cal = Q_benzoic / delta_T_benzoic  # J/K

# Step 2: Calculate energy released by ribose combustion
Q_ribose = C_cal * delta_T_ribose  # J

# Step 3: Calculate internal energy change per mole for ribose
moles_ribose = mass_ribose / M_ribose
delta_U_ribose = Q_ribose / moles_ribose  # J/mol

# Step 4: Calculate enthalpy change of combustion for ribose
# Combustion reaction: C5H10O5 + 6 O2 -> 5 CO2 + 5 H2O (liquid)
# Change in moles of gas: (5 CO2 + 0 H2O gas) - (1 ribose + 6 O2) = 5 - 7 = -2
# Assuming water is liquid, delta_ng = -2

delta_ng = -2

delta_H_ribose = delta_U_ribose + delta_ng * R * T  # J/mol

# Step 5: Calculate enthalpy of formation of ribose
# Using Hess's law:
# delta_H_combustion = [5 * delta_Hf_CO2 + 5 * delta_Hf_H2O] - delta_Hf_ribose
# Rearranged:
# delta_Hf_ribose = [5 * delta_Hf_CO2 + 5 * delta_Hf_H2O] - delta_H_combustion

# Convert delta_H_ribose to kJ/mol
delta_H_ribose_kJ = delta_H_ribose / 1000

delta_Hf_ribose = (5 * delta_Hf_CO2 + 5 * delta_Hf_H2O) - delta_H_ribose_kJ

print(f'Enthalpy of formation of D-ribose: {delta_Hf_ribose:.2f} kJ/mol')
