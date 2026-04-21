# filename: calculate_entropy_change_mercury.py
import math

def calculate_entropy_change(n, T_i, T_f, P_i_bar, P_f_bar, beta, rho, C_P_m, M):
    """
    Calculate the total entropy change (Delta S) for heating and pressurizing liquid mercury.

    Parameters:
    n (float): Number of moles (mol)
    T_i (float): Initial temperature (K)
    T_f (float): Final temperature (K)
    P_i_bar (float): Initial pressure (bar)
    P_f_bar (float): Final pressure (bar)
    beta (float): Thermal expansion coefficient (K^-1)
    rho (float): Density (g/cm^3)
    C_P_m (float): Molar heat capacity at constant pressure (J/(mol K))
    M (float): Molar mass (g/mol)

    Returns:
    float: Total entropy change in J/K
    """
    # Input validation
    if n <= 0:
        raise ValueError("Number of moles must be positive.")
    if T_i <= 0 or T_f <= 0:
        raise ValueError("Temperatures must be positive.")
    if P_i_bar < 0 or P_f_bar < 0:
        raise ValueError("Pressures must be non-negative.")
    if rho <= 0 or M <= 0:
        raise ValueError("Density and molar mass must be positive.")
    if C_P_m <= 0:
        raise ValueError("Heat capacity must be positive.")

    # Convert molar volume from g/cm^3 and g/mol to m^3/mol
    V_m = M / rho  # cm^3/mol
    V_m = V_m * 1e-6  # convert cm^3 to m^3

    # Convert pressure from bar to Pa
    P_i = P_i_bar * 1e5  # Pa
    P_f = P_f_bar * 1e5  # Pa

    # Entropy change due to temperature change at constant pressure
    delta_S_T = n * C_P_m * math.log(T_f / T_i)

    # Entropy change due to pressure change at constant temperature
    delta_S_P = -n * V_m * beta * (P_f - P_i)

    # Total entropy change
    delta_S = delta_S_T + delta_S_P

    return delta_S

# Given data
n = 3.00  # mol
T_i = 300.0  # K
T_f = 600.0  # K
P_i_bar = 1.00  # bar
P_f_bar = 3.00  # bar
beta = 1.81e-4  # K^-1
rho = 13.54  # g/cm^3
C_P_m = 27.98  # J/(mol K)
M = 200.59  # g/mol (molar mass of mercury)

# Calculate entropy change
delta_S = calculate_entropy_change(n, T_i, T_f, P_i_bar, P_f_bar, beta, rho, C_P_m, M)

# Output the result
print(f"Total entropy change (Delta S) = {delta_S:.4f} J/K")
