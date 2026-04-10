import math

# Constants
n = 2.0  # moles of CO2
T1 = 298.15  # initial temperature in K
P1 = 10.0  # initial pressure in atm
P_ext = 1.0  # external pressure in atm
A = 10.0e-4  # cross-sectional area in m^2
d = 0.2  # displacement of piston in m
C_V_m = 28.8  # molar heat capacity at constant volume in J/(mol K)
R = 8.314  # ideal gas constant in J/(mol K)

# Step 1: Calculate change in volume (ΔV)
Delta_V = A * d  # in m^3

# Step 2: Calculate work done (W) using external pressure
W = -P_ext * Delta_V * 101.325  # converting atm to J/m^3

# Step 3: Calculate change in internal energy (ΔU)
Delta_U = W  # since Q = 0 for adiabatic process

# Step 4: Calculate change in temperature (ΔT)
Delta_T = Delta_U / (n * C_V_m)

# Step 5: Calculate final temperature (T2)
T2 = T1 + Delta_T

# Step 6: Calculate initial and final volumes
V_i = (n * R * T1) / P1  # initial volume using ideal gas law
V_f = V_i + Delta_V  # final volume after expansion

# Note: For an adiabatic process, the change in entropy (ΔS) is zero
Delta_S = 0  # Change in entropy for an adiabatic process is zero

# Output results
print(f'Change in Internal Energy (Delta_U): {Delta_U:.4f} J')
print(f'Final Temperature (T2): {T2:.4f} K')
print(f'Initial Volume (V_i): {V_i:.4f} m^3')
print(f'Final Volume (V_f): {V_f:.4f} m^3')
print(f'Change in Entropy (Delta_S): {Delta_S:.4f} J/K (for adiabatic process, Delta_S = 0)')