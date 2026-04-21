# filename: calculate_calorimeter_constant.py

# Given data
mass_benzoic_acid_g = 1.35
molar_mass_benzoic_acid_g_per_mol = 122.12
enthalpy_combustion_kJ_per_mol = -3227  # standard enthalpy of combustion of benzoic acid
mass_water_g = 1.55e3  # 1.55 x 10^3 g, part of the calorimeter system
temperature_rise_K = 2.76

# Calculate number of moles of benzoic acid
moles_benzoic_acid = mass_benzoic_acid_g / molar_mass_benzoic_acid_g_per_mol

# Calculate heat released by the reaction in joules (convert kJ to J)
heat_released_J = abs(moles_benzoic_acid * enthalpy_combustion_kJ_per_mol * 1000)

# Calculate calorimeter constant (J/K)
# Assumes all heat released is absorbed by the calorimeter and water
calorimeter_constant_J_per_K = heat_released_J / temperature_rise_K

# Output the result
print(f"Calorimeter constant: {calorimeter_constant_J_per_K:.2f} J/K")
