# filename: calculate_reaction_gibbs_energy.py

def calculate_reaction_gibbs_energy(T=500):
    """
    Calculate the standard reaction Gibbs energy (Delta G degree) for the reaction:
    N2(g) + 3 H2(g) -> 2 NH3(g) at temperature T (in Kelvin).

    Assumes ideal gases and uses standard enthalpy of formation and entropy values at 298 K.
    Delta G degree = Delta H degree - T * Delta S degree, with Delta H degree and Delta S degree assumed constant over temperature.

    Parameters:
    T (float): Temperature in Kelvin (must be > 0).

    Returns:
    float: Standard reaction Gibbs energy Delta G degree in kJ/mol.
    """
    if T <= 0:
        raise ValueError("Temperature must be greater than zero Kelvin.")

    # Standard enthalpy of formation at 298 K in kJ/mol
    delta_Hf = {
        'N2': 0.0,
        'H2': 0.0,
        'NH3': -46.11
    }

    # Standard molar entropy at 298 K in J/(mol*K)
    S = {
        'N2': 191.5,
        'H2': 130.7,
        'NH3': 192.5
    }

    # Calculate reaction enthalpy change (Delta H degree) in kJ/mol
    delta_H_reaction = 2 * delta_Hf['NH3'] - (1 * delta_Hf['N2'] + 3 * delta_Hf['H2'])

    # Calculate reaction entropy change (Delta S degree) in J/(mol*K)
    delta_S_reaction = 2 * S['NH3'] - (1 * S['N2'] + 3 * S['H2'])

    # Convert Delta S degree to kJ/(mol*K) for consistency
    delta_S_reaction_kJ = delta_S_reaction / 1000.0

    # Calculate Delta G degree at temperature T (K)
    delta_G_reaction = delta_H_reaction - T * delta_S_reaction_kJ

    return delta_G_reaction

if __name__ == '__main__':
    try:
        T = 500  # Temperature in K
        delta_G = calculate_reaction_gibbs_energy(T)
        print(f'Standard reaction Gibbs energy Delta G degree at {T} K: {delta_G:.2f} kJ/mol')
    except Exception as e:
        print(f'Error during calculation: {e}')
