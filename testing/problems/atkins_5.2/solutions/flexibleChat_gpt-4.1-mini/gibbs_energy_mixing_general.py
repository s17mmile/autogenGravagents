# filename: gibbs_energy_mixing_general.py
import math

def gibbs_energy_of_mixing(moles_dict: dict, T_C: float) -> float:
    """
    Calculate the Gibbs energy of mixing for an ideal gas mixture.

    Parameters:
    moles_dict (dict): Dictionary with gas species as keys and mole numbers as values.
    T_C (float): Temperature in degrees Celsius.

    Returns:
    float: Gibbs energy of mixing in Joules.
    """
    R = 8.314  # J/(mol K)
    T = T_C + 273.15  # Convert Celsius to Kelvin

    if T <= 0:
        raise ValueError("Temperature must be greater than 0 K.")

    total_moles = sum(moles_dict.values())
    if total_moles <= 0:
        raise ValueError("Total moles must be positive.")

    delta_G_mix = 0.0
    for species, n_i in moles_dict.items():
        if n_i < 0:
            raise ValueError(f"Moles of {species} must be non-negative.")
        if n_i == 0:
            continue  # zero mole fraction contributes nothing
        x_i = n_i / total_moles
        delta_G_mix += x_i * math.log(x_i)

    delta_G_mix *= total_moles * R * T
    return delta_G_mix

# Given values
moles = {'H2': 3.0, 'N2': 1.0}
T_C = 25

delta_G = gibbs_energy_of_mixing(moles, T_C)

# Print and save result
result_str = f'Gibbs energy of mixing: {delta_G:.2f} J'
print(result_str)
with open('gibbs_energy_result.txt', 'w') as f:
    f.write(result_str + '\n')
