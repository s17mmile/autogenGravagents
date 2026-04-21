# filename: calculate_gibbs_energy_ag2o.py
import math

def calculate_gibbs_energy(P_O2_Pa=11.85, T=298):
    """
    Calculate the standard Gibbs energy of formation of Ag2O at given temperature.

    Parameters:
    P_O2_Pa (float): Equilibrium oxygen partial pressure in pascals (Pa), must be > 0.
    T (float): Temperature in kelvin (K), must be > 0.

    Returns:
    float: Standard Gibbs energy of formation in joules per mole (J/mol).
    """
    if P_O2_Pa <= 0:
        raise ValueError("Oxygen partial pressure must be positive.")
    if T <= 0:
        raise ValueError("Temperature must be positive.")

    R = 8.314  # J/(mol K)
    P_O2_bar = P_O2_Pa / 1e5  # Convert Pa to bar
    delta_G = -0.5 * R * T * math.log(P_O2_bar)
    return delta_G

gibbs_energy = calculate_gibbs_energy()
print(f"Standard Gibbs energy of formation of Ag2O at 298 K: {gibbs_energy:.2f} J/mol")
