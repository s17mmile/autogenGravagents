# filename: calculate_entropy_change_function.py
import math

def calculate_entropy_change(n, Cp_m, T1, T2, P1, P2):
    """
    Calculate the entropy change (Delta S) for an ideal gas undergoing changes in temperature and pressure.

    Parameters:
    n (float): amount of substance in moles
    Cp_m (float): molar heat capacity at constant pressure (J/(mol K))
    T1 (float): initial temperature in Kelvin
    T2 (float): final temperature in Kelvin
    P1 (float): initial pressure (units must be consistent, ratio used so units cancel)
    P2 (float): final pressure (units must be consistent)

    Returns:
    float: entropy change in J/K
    """
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be greater than zero Kelvin.")
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be greater than zero.")

    delta_S = n * Cp_m * math.log(T2 / T1) - n * 8.314 * math.log(P2 / P1)
    return delta_S

# Given data
R = 8.314  # J/(mol K), ideal gas constant
n = 3.00  # mol
Cp_m = (5/2) * R  # molar heat capacity at constant pressure
T1 = 298.15  # K (25 C)
T2 = 398.15  # K (125 C)
P1 = 1.00  # atm (units cancel in ratio)
P2 = 5.00  # atm

# Calculate entropy change
delta_S = calculate_entropy_change(n, Cp_m, T1, T2, P1, P2)

print(f"Delta S (entropy change) = {delta_S:.2f} J/K")
