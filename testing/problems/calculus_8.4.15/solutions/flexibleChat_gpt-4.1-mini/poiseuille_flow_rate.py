# filename: poiseuille_flow_rate.py
import math

def calculate_flow_rate(eta, R, L, delta_P):
    """Calculate volumetric flow rate Q using Poiseuille's Law.

    Parameters:
    eta (float): Dynamic viscosity (dynes*s/cm^2), must be positive.
    R (float): Radius of the vessel (cm), must be positive.
    L (float): Length of the vessel (cm), must be positive.
    delta_P (float): Pressure difference (dynes/cm^2), must be positive.

    Returns:
    float: Volumetric flow rate Q in cm^3/s.
    """
    if eta <= 0 or R <= 0 or L <= 0 or delta_P <= 0:
        raise ValueError("All input parameters must be positive numbers.")
    Q = (math.pi * delta_P * R**4) / (8 * eta * L)
    return Q

if __name__ == '__main__':
    # Given values
    eta = 0.027  # dynamic viscosity in dynes*s/cm^2
    R = 0.008    # radius in cm
    L = 2        # length in cm
    delta_P = 4000  # pressure difference in dynes/cm^2

    flow_rate = calculate_flow_rate(eta, R, L, delta_P)
    print(f"Volumetric flow rate Q = {flow_rate:.6f} cm^3/s")
