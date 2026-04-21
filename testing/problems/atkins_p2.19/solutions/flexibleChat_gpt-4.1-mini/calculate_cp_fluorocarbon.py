# filename: calculate_cp_fluorocarbon.py
import math

def calculate_cp(T1, T2, P1, P2, R=8.314):
    """
    Calculate the constant-pressure heat capacity (C_p) of a gas undergoing
    reversible adiabatic expansion based on initial and final temperatures and pressures.

    Parameters:
    T1 (float): Initial temperature in Kelvin
    T2 (float): Final temperature in Kelvin
    P1 (float): Initial pressure (units consistent, e.g., kPa)
    P2 (float): Final pressure (same units as P1)
    R (float): Ideal gas constant, default 8.314 J/mol·K

    Returns:
    float: Calculated constant-pressure heat capacity C_p in J/mol·K
    """
    # Basic input validation
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be positive and in Kelvin.")
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be positive.")
    if P2 >= P1:
        raise ValueError("Final pressure must be less than initial pressure for expansion.")
    if T2 >= T1:
        raise ValueError("Final temperature must be less than initial temperature for cooling.")

    # Calculate gamma from pressure data using P2/P1 = (V1/V2)^gamma and V2 = 2*V1
    gamma_p = -math.log(P2 / P1) / math.log(2)

    # Calculate gamma from temperature data using T2/T1 = (V1/V2)^(gamma - 1)
    gamma_t = 1 - (math.log(T2 / T1) / math.log(2))

    # Average gamma values to mitigate discrepancies
    gamma_avg = (gamma_p + gamma_t) / 2

    # Calculate C_p using ideal gas relations: C_p = gamma * R / (gamma - 1)
    Cp = (gamma_avg * R) / (gamma_avg - 1)

    return Cp

# Given data
T1 = 298.15  # Initial temperature in K
T2 = 248.44  # Final temperature in K
P1 = 202.94  # Initial pressure in kPa
P2 = 81.840  # Final pressure in kPa

Cp = calculate_cp(T1, T2, P1, P2)

# Print and save result
result_str = f'Calculated constant-pressure heat capacity C_p: {Cp:.2f} J/mol·K'
print(result_str)
with open('cp_result.txt', 'w') as f:
    f.write(result_str + '\n')
