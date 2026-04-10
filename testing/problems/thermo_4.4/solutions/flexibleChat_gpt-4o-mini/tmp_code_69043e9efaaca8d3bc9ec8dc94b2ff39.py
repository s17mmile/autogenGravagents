def calculate_delta_H_f(T_initial, T_final, Delta_H_f_298, C_p):
    # Calculate the integral of C_p from T_initial to T_final
    integral_Cp = C_p * (T_final - T_initial)  # J/mol
    integral_Cp_kJ = integral_Cp / 1000  # Convert to kJ/mol
    
    # Calculate Delta H_f at T_final
    Delta_H_f_final = Delta_H_f_298 + integral_Cp_kJ
    return Delta_H_f_final

# Constants
Delta_H_f_298 = 90.29  # kJ/mol, standard enthalpy of formation at 298.15 K
C_p = 29.1  # J/mol K, heat capacity of NO(g)
T_initial = 298.15  # K
T_final = 975.0  # K

# Calculate Delta H_f
Delta_H_f_result = calculate_delta_H_f(T_initial, T_final, Delta_H_f_298, C_p)

# Print the result
print(f'Delta H_f at {T_final} K: {Delta_H_f_result:.4f} kJ/mol')