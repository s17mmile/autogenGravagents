# filename: calculate_delta_g.py
import math

def calculate_delta_g(n, T, P_initial, P_final, R=8.314):
    """Calculate Gibbs free energy change for isothermal expansion of an ideal gas.

    Parameters:
    n (float): number of moles
    T (float): temperature in Kelvin
    P_initial (float): initial pressure (units must be consistent with P_final)
    P_final (float): final pressure (units must be consistent with P_initial)
    R (float): ideal gas constant in J/(mol*K), default 8.314

    Returns:
    float: Gibbs free energy change in Joules
    """
    if P_initial <= 0 or P_final <= 0:
        raise ValueError("Pressures must be positive values.")
    return n * R * T * math.log(P_final / P_initial)

# Given values
n = 2.25  # moles
T = 325  # Kelvin
P_initial = 12.0  # bar
P_final = 2.5  # bar

# Calculate Delta G
Delta_G = calculate_delta_g(n, T, P_initial, P_final)

# Output the result
print(f"Delta G for the isothermal expansion is {Delta_G:.2f} J")
