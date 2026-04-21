# filename: calculate_delta_U_CO2.py

# Constants
R = 8.314  # J/(mol K), universal gas constant
C_V_m = 28.8  # J/(mol K), molar heat capacity at constant volume for CO2

# Given data
n = 2.0  # mol
T1 = 298.15  # K, initial temperature
P1_atm = 10.0  # atm, initial pressure
P_ext_atm = 1.0  # atm, external pressure
A_cm2 = 10.0  # cm^2, cross-sectional area
delta_x_cm = 20.0  # cm, piston displacement

# Unit conversions
P1 = P1_atm * 1.01325e5  # Pa
P_ext = P_ext_atm * 1.01325e5  # Pa
A = A_cm2 * 1e-4  # m^2
delta_x = delta_x_cm * 1e-2  # m

delta_V = A * delta_x  # m^3, volume change

# Calculate initial volume using ideal gas law: V = nRT/P
V1 = n * R * T1 / P1  # m^3

# Calculate final volume
V2 = V1 + delta_V  # m^3

# Assume final pressure equals external pressure
P2 = P_ext  # Pa

# Calculate final temperature using ideal gas law: T = PV/(nR)
T2 = P2 * V2 / (n * R)  # K

delta_T = T2 - T1  # K

# Calculate change in internal energy: Delta U = n * C_V_m * Delta T
delta_U = n * C_V_m * delta_T  # J

# Output results
print(f"Initial volume V1: {V1:.6f} m^3")
print(f"Final volume V2: {V2:.6f} m^3")
print(f"Final temperature T2: {T2:.2f} K")
print(f"Change in temperature Delta T: {delta_T:.2f} K")
print(f"Change in internal energy Delta U: {delta_U:.2f} J")
