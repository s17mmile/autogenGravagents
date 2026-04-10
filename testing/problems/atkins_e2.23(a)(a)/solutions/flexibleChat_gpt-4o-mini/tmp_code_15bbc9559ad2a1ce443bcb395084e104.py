def calculate_enthalpy_change(Delta_H1, Delta_H2):
    # Reverse reaction (1) and multiply by 2
    Delta_H1_reversed_4HCl = +Delta_H1 * 2  # kJ/mol for 4 HCl
    
    # Calculate Delta_r H^ominus for reaction (3) correctly
    Delta_r_H3 = Delta_H1_reversed_4HCl + Delta_H2  # kJ/mol
    return Delta_r_H3

# Given enthalpy changes for the reactions
Delta_H1 = -184.62  # kJ/mol for reaction (1)
Delta_H2 = -483.64  # kJ/mol for reaction (2)

# Calculate and output the result
result = calculate_enthalpy_change(Delta_H1, Delta_H2)
print(f'Delta_r H^ominus for reaction (3): {result:.2f} kJ/mol')