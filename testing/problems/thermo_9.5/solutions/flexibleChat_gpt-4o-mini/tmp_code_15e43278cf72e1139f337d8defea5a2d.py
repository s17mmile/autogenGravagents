def calculate_volume_change(n_H2O, x_H2O, partial_molar_volume_H2O, partial_molar_volume_ethanol, molar_mass_H2O, density_H2O, molar_mass_ethanol, density_ethanol):
    # Calculate mole fraction of ethanol
    x_ethanol = 1 - x_H2O

    # Calculate total moles in the solution
    total_moles = n_H2O / x_H2O
    n_ethanol = total_moles * x_ethanol

    # Calculate total volume of the solution
    V_H2O = n_H2O * partial_molar_volume_H2O
    V_ethanol = n_ethanol * partial_molar_volume_ethanol
    V_total = V_H2O + V_ethanol

    # Calculate initial volumes of water and ethanol
    V_initial_H2O = n_H2O * (molar_mass_H2O / density_H2O)
    V_initial_ethanol = n_ethanol * (molar_mass_ethanol / density_ethanol)

    # Calculate volume change upon mixing
    Delta_V = V_total - (V_initial_H2O + V_initial_ethanol)

    return Delta_V

# Constants
n_H2O = 3.75  # moles of water
x_H2O = 0.45  # mole fraction of water
partial_molar_volume_H2O = 17.0  # cm^3 mol^-1
partial_molar_volume_ethanol = 57.5  # cm^3 mol^-1
molar_mass_H2O = 18.015  # g mol^-1
density_H2O = 0.997  # g cm^-3
molar_mass_ethanol = 46.068  # g mol^-1
density_ethanol = 0.7893  # g cm^-3

# Calculate volume change
volume_change = calculate_volume_change(n_H2O, x_H2O, partial_molar_volume_H2O, partial_molar_volume_ethanol, molar_mass_H2O, density_H2O, molar_mass_ethanol, density_ethanol)

# Output the volume change
print('Volume change upon mixing:', volume_change, 'cm^3')