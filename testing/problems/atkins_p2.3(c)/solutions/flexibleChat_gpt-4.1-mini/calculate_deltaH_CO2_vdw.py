# filename: calculate_deltaH_CO2_vdw.py

# Constants for CO2
n = 2.0  # moles
V_dm3 = 15.0  # volume in dm^3
V_L = V_dm3  # volume in liters (1 dm^3 = 1 L)
T1 = 300.0  # initial temperature in K
T2 = 341.0  # final temperature in K
q_kJ = 2.35  # heat supplied in kJ
q_J = q_kJ * 1000  # heat in J

# van der Waals constants for CO2
# a in L^2*bar/mol^2, b in L/mol
a = 3.59  # L^2*bar/mol^2
b = 0.0427  # L/mol

# Gas constant R in L*bar/(mol*K)
R = 0.08314

# Calculate molar volume at initial and final states
Vm = V_L / n  # molar volume in L/mol (constant since volume fixed)

# Check for valid molar volume
if Vm <= b:
    raise ValueError("Molar volume must be greater than b to avoid division by zero.")

# Calculate pressure using van der Waals equation: (P + a(n/V)^2)(Vm - b) = RT
# Rearranged: P = (RT/(Vm - b)) - a/(Vm^2)
P1 = (R * T1) / (Vm - b) - a / (Vm ** 2)  # pressure in bar at T1
P2 = (R * T2) / (Vm - b) - a / (Vm ** 2)  # pressure in bar at T2

# Calculate change in internal energy (Delta U) at constant volume
# q_v = Delta U (since volume constant)
Delta_U_J = q_J  # in J

# Convert P from bar to Pa (1 bar = 1e5 Pa), V from L to m^3 (1 L = 1e-3 m^3)
P1_Pa = P1 * 1e5
P2_Pa = P2 * 1e5
V_m3 = V_L * 1e-3

# Calculate change in PV term (in J)
PV1 = P1_Pa * V_m3  # in J
PV2 = P2_Pa * V_m3  # in J
Delta_PV_J = PV2 - PV1

# Calculate Delta H
Delta_H_J = Delta_U_J + Delta_PV_J
Delta_H_kJ = Delta_H_J / 1000  # convert to kJ

# Output results
print(f"Initial pressure P1: {P1:.3f} bar")
print(f"Final pressure P2: {P2:.3f} bar")
print(f"Change in internal energy (Delta U): {Delta_U_J:.2f} J")
print(f"Change in PV term: {Delta_PV_J:.2f} J")
print(f"Change in enthalpy (Delta H): {Delta_H_kJ:.3f} kJ")

# Return Delta H for further use if needed
Delta_H_kJ
