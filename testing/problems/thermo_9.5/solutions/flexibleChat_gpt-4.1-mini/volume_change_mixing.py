# filename: volume_change_mixing.py

# Given data
x_H2O = 0.45  # mole fraction of water
n_H2O = 3.75  # moles of water
Vbar_H2O = 17.0  # partial molar volume of water in cm^3/mol
Vbar_ethanol = 57.5  # partial molar volume of ethanol in cm^3/mol

density_H2O = 0.997  # g/cm^3
density_ethanol = 0.7893  # g/cm^3

molar_mass_H2O = 18.015  # g/mol
molar_mass_ethanol = 46.07  # g/mol

# Calculate moles of ethanol
n_ethanol = n_H2O * (1 - x_H2O) / x_H2O

# Calculate initial volumes of pure water and ethanol
V_H2O_initial = (n_H2O * molar_mass_H2O) / density_H2O  # cm^3
V_ethanol_initial = (n_ethanol * molar_mass_ethanol) / density_ethanol  # cm^3

# Calculate total volume of mixture using partial molar volumes
V_mixture = n_H2O * Vbar_H2O + n_ethanol * Vbar_ethanol  # cm^3

# Calculate volume change upon mixing
delta_V = V_mixture - (V_H2O_initial + V_ethanol_initial)  # cm^3

print(f"Volume change upon mixing: {delta_V:.2f} cm^3")
