# filename: gibbs_energy_formation_phenol.py

def calculate_gibbs_energy_formation_phenol():
    # Given data
    delta_H_comb = -3054.0  # kJ/mol, enthalpy of combustion of phenol
    S_phenol = 144.0  # J/(K mol), entropy of phenol
    T = 298.15  # K, temperature

    # Standard enthalpies of formation (kJ/mol)
    delta_Hf_CO2 = -393.5
    delta_Hf_H2O = -285.8

    # Standard molar entropies (J/(K mol))
    S_C_graphite = 5.7
    S_H2 = 130.7
    S_O2 = 205.0

    # Calculate standard enthalpy of formation of phenol
    delta_Hf_phenol = (6 * delta_Hf_CO2 + 2.5 * delta_Hf_H2O) - delta_H_comb

    # Calculate standard entropy of formation of phenol
    delta_Sf_phenol = S_phenol - (6 * S_C_graphite + 3 * S_H2 + 0.5 * S_O2)

    # Convert entropy from J/(K mol) to kJ/(K mol) for consistency
    delta_Sf_phenol_kJ = delta_Sf_phenol / 1000.0

    # Calculate standard Gibbs energy of formation
    delta_Gf_phenol = delta_Hf_phenol - T * delta_Sf_phenol_kJ

    return delta_Gf_phenol

# Calculate and print the result
gibbs_energy_formation = calculate_gibbs_energy_formation_phenol()
print(f"Standard Gibbs energy of formation of phenol at 298 K: {gibbs_energy_formation:.2f} kJ/mol")
