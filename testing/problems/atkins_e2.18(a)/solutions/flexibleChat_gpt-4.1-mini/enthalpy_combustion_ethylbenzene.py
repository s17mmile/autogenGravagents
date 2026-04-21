# filename: enthalpy_combustion_ethylbenzene.py

def calculate_enthalpy_combustion(deltaHf_ethylbenzene=-12.5, deltaHf_CO2=-393.5, deltaHf_H2O=-285.8, deltaHf_O2=0.0):
    """Calculate the standard enthalpy of combustion of ethylbenzene in kJ/mol.

    Parameters:
    deltaHf_ethylbenzene (float): Standard enthalpy of formation of ethylbenzene (kJ/mol).
    deltaHf_CO2 (float): Standard enthalpy of formation of CO2 (kJ/mol).
    deltaHf_H2O (float): Standard enthalpy of formation of H2O (kJ/mol).
    deltaHf_O2 (float): Standard enthalpy of formation of O2 (kJ/mol), usually 0.

    Returns:
    float: Standard enthalpy of combustion of ethylbenzene (kJ/mol).
    """
    # Balanced combustion reaction coefficients
    n_CO2 = 8
    n_H2O = 5
    n_O2 = 10.5

    # Basic validation
    for val in [deltaHf_ethylbenzene, deltaHf_CO2, deltaHf_H2O, deltaHf_O2]:
        if not isinstance(val, (int, float)):
            raise ValueError("Enthalpy values must be numeric.")

    # Calculate enthalpy of combustion using Hess's law
    deltaH_combustion = (n_CO2 * deltaHf_CO2 + n_H2O * deltaHf_H2O) - (deltaHf_ethylbenzene + n_O2 * deltaHf_O2)

    return deltaH_combustion

if __name__ == "__main__":
    deltaH_comb = calculate_enthalpy_combustion()
    print(f"Standard enthalpy of combustion of ethylbenzene: {deltaH_comb:.1f} kJ/mol")
