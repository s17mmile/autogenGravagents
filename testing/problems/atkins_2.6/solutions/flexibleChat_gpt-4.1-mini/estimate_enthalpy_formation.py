# filename: estimate_enthalpy_formation.py

def estimate_enthalpy_formation(T1, T2, delta_Hf_T1, Cp_H2O, Cp_H2, Cp_O2):
    """Estimate the standard enthalpy of formation of H2O(g) at temperature T2.

    Parameters:
    T1 (float): Initial temperature in Kelvin
    T2 (float): Target temperature in Kelvin
    delta_Hf_T1 (float): Standard enthalpy of formation at T1 in kJ/mol
    Cp_H2O (float): Molar heat capacity of H2O(g) in J/(K mol)
    Cp_H2 (float): Molar heat capacity of H2(g) in J/(K mol)
    Cp_O2 (float): Molar heat capacity of O2(g) in J/(K mol)

    Returns:
    float: Estimated standard enthalpy of formation at T2 in kJ/mol
    """
    # Calculate change in heat capacity (J/K/mol)
    delta_Cp = Cp_H2O - (Cp_H2 + 0.5 * Cp_O2)
    # Convert delta_Cp to kJ/K/mol
    delta_Cp_kJ = delta_Cp / 1000.0
    # Calculate temperature difference
    delta_T = T2 - T1
    # Calculate enthalpy of formation at T2
    delta_Hf_T2 = delta_Hf_T1 + delta_Cp_kJ * delta_T
    return delta_Hf_T2

# Given values
T1 = 298.0  # K
T2 = 373.0  # K (100 degrees Celsius)
delta_Hf_T1 = -241.82  # kJ/mol
Cp_H2O = 33.58  # J/K/mol
Cp_H2 = 28.82   # J/K/mol
Cp_O2 = 29.36   # J/K/mol

# Calculate
estimated_enthalpy = estimate_enthalpy_formation(T1, T2, delta_Hf_T1, Cp_H2O, Cp_H2, Cp_O2)

# Output the result
print(f"Estimated standard enthalpy of formation of H2O(g) at {T2} K: {estimated_enthalpy:.2f} kJ/mol")
