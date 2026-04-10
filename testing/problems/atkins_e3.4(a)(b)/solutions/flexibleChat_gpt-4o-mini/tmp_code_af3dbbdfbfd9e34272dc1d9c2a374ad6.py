import math

def calculate_entropy_change(n, C_V_m, T_i, T_f):
    # Calculate the change in entropy for an adiabatic process
    delta_S = n * C_V_m * math.log(T_f / T_i)
    return delta_S

# Constants for the problem
n = 3.00  # moles
C_V_m = 27.5  # J K^-1 mol^-1
T_i = 200.0  # K
T_f = 250.0  # K

# Calculate the change in entropy
entropy_change = calculate_entropy_change(n, C_V_m, T_i, T_f)
print(f'The change in entropy (ΔS) is approximately: {entropy_change:.2f} J K^-1')