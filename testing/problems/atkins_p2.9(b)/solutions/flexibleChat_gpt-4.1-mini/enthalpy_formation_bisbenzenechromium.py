# filename: enthalpy_formation_bisbenzenechromium.py

def calculate_enthalpy_formation():
    """Calculate the standard enthalpy of formation of bis(benzene)chromium at 583 K."""
    # Given data
    delta_U_reaction_kJ = 8.0  # Internal energy change at 583 K in kJ/mol
    T = 583  # Temperature in K
    R = 8.314 / 1000  # Gas constant in kJ/mol/K
    delta_n_gas = 2  # Change in moles of gas (0 -> 2 moles of benzene gas)

    # Heat capacities (assumed constant over temperature range)
    Cp_benzene_gas = 81.67 / 1000  # kJ/mol/K

    # Standard enthalpy of formation of benzene gas at 298.15 K (literature value)
    delta_fH_benzene_298 = 82.9  # kJ/mol

    # Calculate reaction enthalpy at 583 K
    delta_H_reaction = delta_U_reaction_kJ + delta_n_gas * R * T

    # Adjust benzene enthalpy of formation from 298.15 K to 583 K
    delta_T = T - 298.15
    delta_H_Cp_benzene = Cp_benzene_gas * delta_T
    delta_fH_benzene_583 = delta_fH_benzene_298 + delta_H_Cp_benzene

    # Enthalpy of formation of chromium solid is zero
    delta_fH_chromium = 0.0

    # Calculate enthalpy of formation of bis(benzene)chromium at 583 K
    delta_fH_complex_583 = (delta_fH_chromium + 2 * delta_fH_benzene_583) - delta_H_reaction

    return {
        'delta_H_reaction_kJ_per_mol': delta_H_reaction,
        'delta_fH_benzene_583_kJ_per_mol': delta_fH_benzene_583,
        'delta_fH_complex_583_kJ_per_mol': delta_fH_complex_583
    }


if __name__ == '__main__':
    results = calculate_enthalpy_formation()
    for key, value in results.items():
        print(f'{key}: {value:.2f} kJ/mol')
