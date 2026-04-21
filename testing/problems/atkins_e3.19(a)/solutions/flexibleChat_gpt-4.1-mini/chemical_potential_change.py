# filename: chemical_potential_change.py
import math

def calculate_chemical_potential_change(P1, P2, T):
    """
    Calculate the change in chemical potential (J/mol) for an ideal gas
    undergoing an isothermal pressure change from P1 to P2 at temperature T.

    Parameters:
    P1 (float): Initial pressure in atm (must be > 0)
    P2 (float): Final pressure in atm (must be > 0)
    T (float): Temperature in Kelvin (must be > 0)

    Returns:
    float: Change in chemical potential in Joules per mole
    """
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be positive values.")
    if T <= 0:
        raise ValueError("Temperature must be a positive value in Kelvin.")

    R = 8.3145  # J/(mol·K)
    delta_mu = R * T * math.log(P2 / P1)
    return delta_mu


if __name__ == '__main__':
    # Given data
    P1 = 1.8  # atm
    P2 = 29.5  # atm
    T = 40 + 273.15  # Convert Celsius to Kelvin

    # Calculate change in chemical potential
    delta_mu = calculate_chemical_potential_change(P1, P2, T)

    # Save result to a file
    with open('chemical_potential_change_result.txt', 'w') as f:
        f.write(f'Change in chemical potential (J/mol): {delta_mu:.4f}\n')
