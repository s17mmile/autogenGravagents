# filename: calculate_vapor_pressure_formic_acid.py
import math

def calculate_vapor_pressure(deltaG_gas_kJ_per_mol, deltaG_liquid_kJ_per_mol, temperature_K):
    """
    Calculate the vapor pressure of formic acid at a given temperature.

    Parameters:
    deltaG_gas_kJ_per_mol (float): Standard Gibbs free energy of formation of gaseous formic acid (kJ/mol).
    deltaG_liquid_kJ_per_mol (float): Standard Gibbs free energy of formation of liquid formic acid (kJ/mol).
    temperature_K (float): Temperature in Kelvin.

    Returns:
    float: Vapor pressure in bar (assuming standard pressure of 1 bar).
    """
    R_kJ_per_mol_K = 0.008314  # Gas constant in kJ/(mol*K)
    # Calculate Gibbs free energy change for phase transition (liquid to gas)
    deltaG_phase_transition = deltaG_gas_kJ_per_mol - deltaG_liquid_kJ_per_mol
    # Calculate vapor pressure using the relation P = exp(-DeltaG / RT)
    vapor_pressure = math.exp(-deltaG_phase_transition / (R_kJ_per_mol_K * temperature_K))
    return vapor_pressure

# Given data
T = 298.15  # Temperature in K
DeltaG_f_gas = -351.0  # Gibbs free energy of formation for gaseous formic acid (kJ/mol)
DeltaG_f_liquid = -361.4  # Gibbs free energy of formation for liquid formic acid (kJ/mol)

vapor_pressure = calculate_vapor_pressure(DeltaG_f_gas, DeltaG_f_liquid, T)

print(f"Vapor pressure of formic acid at {T} K is {vapor_pressure:.5f} bar")
