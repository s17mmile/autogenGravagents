# filename: calculate_entropy_change_copper_blocks.py

import math

def calculate_entropy_change(mass_g=10000, c=0.385, T1=373.15, T2=273.15):
    """
    Calculate total entropy change when two copper blocks are placed in contact.

    Parameters:
    mass_g (float): mass of each block in grams
    c (float): specific heat capacity in J/g/K
    T1 (float): initial temperature of hot block in Kelvin
    T2 (float): initial temperature of cold block in Kelvin

    Returns:
    float: total entropy change in J/K
    """
    # Input validation
    if mass_g <= 0:
        raise ValueError("Mass must be positive.")
    if c <= 0:
        raise ValueError("Specific heat capacity must be positive.")
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be greater than zero Kelvin.")

    # Final equilibrium temperature (average since masses and c are equal)
    Tf = (T1 + T2) / 2

    # Calculate entropy changes
    delta_S1 = mass_g * c * math.log(Tf / T1)  # entropy change for hot block
    delta_S2 = mass_g * c * math.log(Tf / T2)  # entropy change for cold block

    delta_S_total = delta_S1 + delta_S2

    return delta_S_total

if __name__ == '__main__':
    delta_S_tot = calculate_entropy_change()
    print(f'Total entropy change (Delta S_tot) = {delta_S_tot:.2f} J/K')
