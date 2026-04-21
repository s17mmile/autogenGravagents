# filename: entropy_change_nitrogen.py
import math

def calculate_entropy_change(mass_g, molar_mass_g_per_mol, temperature_K, volume_ratio):
    """
    Calculate the entropy changes for an isothermal reversible expansion of an ideal gas.

    Parameters:
    mass_g (float): Mass of the gas in grams.
    molar_mass_g_per_mol (float): Molar mass of the gas in grams per mole.
    temperature_K (float): Temperature in Kelvin (included for completeness, not used in calculation).
    volume_ratio (float): Ratio of final volume to initial volume (V2/V1), must be > 0.

    Returns:
    tuple: (delta_S_gas, delta_S_surroundings, delta_S_total) in J/K
    """
    if volume_ratio <= 0:
        raise ValueError("Volume ratio must be greater than zero.")

    R = 8.314  # J/(mol*K), universal gas constant
    n = mass_g / molar_mass_g_per_mol  # moles of gas
    delta_S_gas = n * R * math.log(volume_ratio)  # entropy change of gas
    delta_S_surroundings = -delta_S_gas  # entropy change of surroundings (reversible process)
    delta_S_total = delta_S_gas + delta_S_surroundings  # total entropy change
    return delta_S_gas, delta_S_surroundings, delta_S_total

# Given data
mass_nitrogen = 14.0  # grams
molar_mass_nitrogen = 28.02  # g/mol
temperature = 298.15  # K (not used in calculation but included for completeness)
volume_ratio = 2.0  # final volume / initial volume

# Calculate entropy changes
delta_S_gas, delta_S_surroundings, delta_S_total = calculate_entropy_change(
    mass_nitrogen, molar_mass_nitrogen, temperature, volume_ratio
)

# Print results
print(f"Entropy change of nitrogen gas: {delta_S_gas:.2f} J/K")
print(f"Entropy change of surroundings: {delta_S_surroundings:.2f} J/K")
print(f"Total entropy change: {delta_S_total:.2f} J/K")
