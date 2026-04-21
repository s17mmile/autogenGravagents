# filename: calculate_argon_mole_fraction.py

# Given data
mass_chemical_nitrogen = 2.2990  # grams
mass_atmospheric_nitrogen = 2.3102  # grams

# Molar masses in g/mol
molar_mass_nitrogen = 28.0134
molar_mass_argon = 39.948

# Calculate total moles from chemical nitrogen sample (pure N2)
total_moles = mass_chemical_nitrogen / molar_mass_nitrogen

# Calculate mole fraction of argon in atmospheric nitrogen
x_argon = (mass_atmospheric_nitrogen / total_moles - molar_mass_nitrogen) / (molar_mass_argon - molar_mass_nitrogen)

print(f'Mole fraction of argon in atmospheric nitrogen: {x_argon:.6f}')
