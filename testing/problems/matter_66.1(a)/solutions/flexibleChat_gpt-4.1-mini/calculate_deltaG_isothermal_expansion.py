# filename: calculate_deltaG_isothermal_expansion.py
import math

def calculate_deltaG(n_mol, V1_cm3, V2_cm3, T_K):
    """Calculate the change in Gibbs free energy for isothermal expansion of an ideal gas.

    Parameters:
    n_mol (float): Amount of substance in moles
    V1_cm3 (float): Initial volume in cubic centimeters
    V2_cm3 (float): Final volume in cubic centimeters
    T_K (float): Temperature in Kelvin

    Returns:
    float: Change in Gibbs free energy in Joules
    """
    if V1_cm3 <= 0 or V2_cm3 <= 0 or T_K <= 0:
        raise ValueError("Volumes and temperature must be positive values.")
    R = 8.314  # J/(mol K)
    ratio = V1_cm3 / V2_cm3
    deltaG = n_mol * R * T_K * math.log(ratio)
    return deltaG

# Given values
n_mol = 2.5e-3  # 2.5 mmol in mol
V1_cm3 = 42
V2_cm3 = 600
T_K = 300

deltaG = calculate_deltaG(n_mol, V1_cm3, V2_cm3, T_K)
print(f"Delta G for the isothermal expansion: {deltaG:.4f} J")
