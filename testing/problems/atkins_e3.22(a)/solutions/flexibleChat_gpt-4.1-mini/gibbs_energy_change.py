# filename: gibbs_energy_change.py
import math

def calculate_delta_gibbs_energy(T=298.15, P1=1.0, P2=100.0):
    """
    Calculate the change in molar Gibbs energy for an ideal gas during an isothermal pressure change.

    Parameters:
    T (float): Temperature in Kelvin (must be > 0)
    P1 (float): Initial pressure in atm (must be > 0)
    P2 (float): Final pressure in atm (must be > 0)

    Returns:
    float: Change in molar Gibbs energy in joules per mole
    """
    if T <= 0 or P1 <= 0 or P2 <= 0:
        raise ValueError("Temperature and pressures must be positive values.")
    R = 8.314  # J/(mol*K), universal gas constant
    delta_G = R * T * math.log(P2 / P1)
    return delta_G

if __name__ == "__main__":
    delta_G = calculate_delta_gibbs_energy()
    print(f"Change in molar Gibbs energy (J/mol): {delta_G:.2f}")
