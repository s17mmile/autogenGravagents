# filename: methane_combustion_deltaG.py

def calculate_delta_G(delta_H, delta_S, temperature):
    """
    Calculate the Gibbs free energy change (Delta G) for the combustion of methane.

    Parameters:
    delta_H (float): Enthalpy change (in kJ/mol)
    delta_S (float): Entropy change (in J/K\u00b7mol)
    temperature (float): Temperature in Kelvin

    Returns:
    float: Gibbs free energy change (Delta G) in kJ/mol

    Raises:
    ValueError: If temperature is not positive or inputs are not numeric.
    """
    # Input validation
    if not (isinstance(delta_H, (int, float)) and isinstance(delta_S, (int, float)) and isinstance(temperature, (int, float))):
        raise ValueError("All inputs must be numeric.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero Kelvin.")

    # Convert delta_S from J/K\u00b7mol to kJ/K\u00b7mol for consistency
    delta_S_kJ = delta_S / 1000
    
    # Calculate Delta G
    delta_G = delta_H - (temperature * delta_S_kJ)
    
    return delta_G

# Standard values for methane combustion at 298.15 K
# Enthalpy change (Delta H) in kJ/mol
# Entropy change (Delta S) in J/K\u00b7mol
# These values are typical literature values for the reaction:
# CH4 + 2 O2 -> CO2 + 2 H2O (liquid)
delta_H = -890.3  # kJ/mol
delta_S = 213.7   # J/K\u00b7mol
temperature = 298.15  # K

# Calculate Delta G
delta_G = calculate_delta_G(delta_H, delta_S, temperature)

# Maximum non-expansion work per mole is -Delta G
max_non_expansion_work = -delta_G

print(f"The Gibbs free energy change (Delta G) for the combustion of methane is: {delta_G:.2f} kJ/mol")
print(f"The maximum non-expansion work per mole obtainable from the fuel cell is: {max_non_expansion_work:.2f} kJ/mol")
