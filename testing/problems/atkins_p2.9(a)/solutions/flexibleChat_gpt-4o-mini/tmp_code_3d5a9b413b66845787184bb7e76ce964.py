def calculate_enthalpy(Delta_r_U_kJ, R, T, Delta_n, Delta_f_H_Cr, Delta_f_H_benzene_kJ):
    try:
        # Convert Delta_r_U from kJ to J
        Delta_r_U = Delta_r_U_kJ * 1000  # in J/mol
        # Step 1: Calculate reaction enthalpy Delta_r_H
        Delta_r_H = Delta_r_U + Delta_n * R * T
        # Step 2: Estimate standard enthalpy of formation of bis(benzene)chromium
        Delta_f_H_bis_benzene_chromium = Delta_r_H - Delta_f_H_Cr - 2 * Delta_f_H_benzene_kJ * 1000
        # Convert results back to kJ/mol
        Delta_r_H_kJ = Delta_r_H / 1000  # Convert J to kJ
        Delta_f_H_bis_benzene_chromium_kJ = Delta_f_H_bis_benzene_chromium / 1000  # Convert J to kJ
        return Delta_r_H_kJ, Delta_f_H_bis_benzene_chromium_kJ
    except Exception as e:
        return str(e)

# Constants
Delta_r_U_kJ = 8.0  # in kJ/mol
R = 8.314  # J/(K*mol)
T = 583  # K
Delta_n = 2  # Change in moles of gas
Delta_f_H_Cr = 0  # Standard enthalpy of formation of solid chromium in kJ/mol
Delta_f_H_benzene_kJ = 82.9  # Standard enthalpy of formation of benzene gas in kJ/mol

# Calculate enthalpy
reaction_enthalpy, formation_enthalpy = calculate_enthalpy(Delta_r_U_kJ, R, T, Delta_n, Delta_f_H_Cr, Delta_f_H_benzene_kJ)

# Output results
print(f'Reaction Enthalpy at 583 K: {reaction_enthalpy:.2f} kJ/mol')
print(f'Standard Enthalpy of Formation of bis(benzene)chromium: {formation_enthalpy:.2f} kJ/mol')