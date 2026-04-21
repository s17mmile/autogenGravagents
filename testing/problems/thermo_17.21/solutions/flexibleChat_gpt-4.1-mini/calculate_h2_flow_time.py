# filename: calculate_h2_flow_time.py
import math

def calculate_h2_flow_time(V_mL=200, T_K=273, L_m=0.10, r_m=0.00025, P1_atm=1.05, P2_atm=1.00, eta=8.76e-6):
    """
    Calculate the time required to pass a given volume of H2 gas through a capillary tube.

    Parameters:
    V_mL : float - volume in milliliters
    T_K : float - temperature in Kelvin (not used directly but noted)
    L_m : float - length of the capillary tube in meters
    r_m : float - radius of the capillary tube in meters
    P1_atm : float - input pressure in atm
    P2_atm : float - output pressure in atm
    eta : float - dynamic viscosity of hydrogen in Pa.s

    Returns:
    time_seconds : float - time in seconds
    Q : float - volumetric flow rate in m^3/s
    """
    atm_to_Pa = 101325  # conversion factor from atm to Pascal

    # Convert volume from mL to cubic meters
    V_m3 = V_mL * 1e-6  # 1 mL = 1e-6 m^3

    # Calculate pressure difference in Pascals
    delta_P = (P1_atm - P2_atm) * atm_to_Pa

    # Calculate volumetric flow rate Q using Hagen-Poiseuille equation
    Q = (math.pi * r_m**4 * delta_P) / (8 * eta * L_m)  # in m^3/s

    # Calculate time to pass volume V
    time_seconds = V_m3 / Q

    return time_seconds, Q


if __name__ == "__main__":
    time_sec, flow_rate = calculate_h2_flow_time()
    print(f"Volumetric flow rate Q = {flow_rate:.6e} m^3/s")
    print(f"Time required to pass 200 mL of H2 gas = {time_sec:.2f} seconds")
