import math

def calculate_entropy_change(n, C_v, R, P_i, P_f, T_i, T_f):
    # Step 1: Calculate the entropy change due to temperature change
    Delta_S_T = n * C_v * math.log(T_f / T_i)
    
    # Step 2: Calculate the entropy change due to pressure change
    if P_f <= 0 or P_i <= 0:
        raise ValueError('Pressure must be positive.')
    Delta_S_P = n * R * math.log(P_i / P_f)
    
    # Step 3: Total Entropy Change
    Delta_S = Delta_S_T + Delta_S_P
    return Delta_S

# Given data
n = 1  # moles of water
C_v = 75.3  # J/(mol·K)
R = 8.314  # J/(mol·K)
P_i = 1.00e5  # Initial pressure in Pa
P_f = 590e5  # Final pressure in Pa
T_i = 350  # Initial temperature in K
T_f = 750  # Final temperature in K

# Calculate the change in entropy
try:
    delta_entropy = calculate_entropy_change(n, C_v, R, P_i, P_f, T_i, T_f)
    # Print the result
    print(f'The change in entropy ΔS is approximately: {delta_entropy:.2f} J/K')
except ValueError as e:
    print(e)