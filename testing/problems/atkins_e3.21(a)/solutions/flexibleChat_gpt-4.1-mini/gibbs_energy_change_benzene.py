# filename: gibbs_energy_change_benzene.py

import math

# Constants
R = 8.314  # J/(mol*K), ideal gas constant
T = 298.15  # K, temperature


def calculate_delta_g(V_dm3, P1_atm, P2_atm, T_kelvin=T):
    """Calculate the change in Gibbs energy for benzene gas when pressure changes from P1 to P2 at constant temperature.

    Parameters:
    V_dm3 (float): Volume in dm^3
    P1_atm (float): Initial pressure in atm
    P2_atm (float): Final pressure in atm
    T_kelvin (float): Temperature in Kelvin (default 298.15 K)

    Returns:
    float: Change in Gibbs energy in Joules
    """
    if V_dm3 <= 0 or P1_atm <= 0 or P2_atm <= 0:
        raise ValueError("Volume and pressures must be positive values.")

    # Convert volume from dm^3 to m^3
    V_m3 = V_dm3 * 1e-3  # m^3

    # Convert pressure from atm to Pa (1 atm = 101325 Pa)
    P1_Pa = P1_atm * 101325  # Pa
    P2_Pa = P2_atm * 101325  # Pa

    # Calculate number of moles using ideal gas law: n = P1 * V / (R * T)
    n = (P1_Pa * V_m3) / (R * T_kelvin)

    # Calculate change in Gibbs energy: delta_G = n * R * T * ln(P2/P1)
    delta_G = n * R * T_kelvin * math.log(P2_Pa / P1_Pa)  # J

    return delta_G


# Example usage
V = 1.0  # dm^3
P_initial = 1.0  # atm
P_final = 100.0  # atm

delta_G_value = calculate_delta_g(V, P_initial, P_final)
print(f"Change in Gibbs energy (delta G) when pressure increases from {P_initial} atm to {P_final} atm:")
print(f"{delta_G_value:.2f} J")
