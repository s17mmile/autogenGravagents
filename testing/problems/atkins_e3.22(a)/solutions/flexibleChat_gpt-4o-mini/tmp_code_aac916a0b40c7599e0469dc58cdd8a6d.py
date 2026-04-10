import math

# Constants
R = 8.314  # J/(mol·K)
T = 298.15  # K
P_i = 1.0  # Initial pressure in atm
P_f = 100.0  # Final pressure in atm
n = 1  # Number of moles (for molar Gibbs energy)

# Calculate the change in molar Gibbs energy
try:
    delta_G = n * R * T * math.log(P_f / P_i)
    # Print the result in Joules and kilojoules
    print(f'Change in molar Gibbs energy (Delta G): {delta_G:.2f} J/mol')
    print(f'Change in molar Gibbs energy (Delta G): {delta_G / 1000:.2f} kJ/mol')
except ZeroDivisionError:
    print('Error: Initial pressure P_i cannot be zero.')