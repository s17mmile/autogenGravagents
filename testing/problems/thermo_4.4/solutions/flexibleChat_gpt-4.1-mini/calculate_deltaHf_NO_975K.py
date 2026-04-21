# filename: calculate_deltaHf_NO_975K.py

# Constants
delta_Hf_298 = 90250  # Standard enthalpy of formation of NO at 298.15 K in J/mol
Cp_NO = 29.0          # Heat capacity of NO at 298.15 K in J/mol.K
Cp_N2 = 29.1          # Heat capacity of N2 at 298.15 K in J/mol.K
Cp_O2 = 29.4          # Heat capacity of O2 at 298.15 K in J/mol.K

T_initial = 298.15    # Initial temperature in K
T_final = 975         # Final temperature in K

# Calculate delta Cp
delta_Cp = Cp_NO - 0.5 * Cp_N2 - 0.5 * Cp_O2

# Calculate delta Hf at T_final
delta_Hf_T = delta_Hf_298 + delta_Cp * (T_final - T_initial)

# Output result in kJ/mol
delta_Hf_T_kJ = delta_Hf_T / 1000

# Print the result
print(f'Standard enthalpy of formation of NO at {T_final} K: {delta_Hf_T_kJ:.2f} kJ/mol')

# Save result to a file with error handling
try:
    with open('deltaHf_NO_975K.txt', 'w') as f:
        f.write(f'Standard enthalpy of formation of NO at {T_final} K: {delta_Hf_T_kJ:.2f} kJ/mol\n')
except IOError as e:
    print(f'Error writing to file: {e}')
