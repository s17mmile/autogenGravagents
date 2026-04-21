# filename: calculate_work_vdw_co2.py

# Constants for CO2 (van der Waals constants)
a = 3.59  # L^2 atm / mol^2
b = 0.0427  # L / mol
R = 0.08206  # L atm / mol K

# Given data
n = 2.0  # mol
V = 15.0  # dm^3 = L
T1 = 300.0  # K
T2 = 341.0  # K

# Calculate molar volume
V_m = V / n  # L/mol

# Calculate initial and final pressures using van der Waals equation
P1 = (R * T1) / (V_m - b) - a * (n / V)**2  # atm
P2 = (R * T2) / (V_m - b) - a * (n / V)**2  # atm

# Work done w = -P * deltaV, but deltaV = 0 (fixed volume)
w = 0.0  # J

# Output results
print(f"Initial pressure P1 = {P1:.3f} atm")
print(f"Final pressure P2 = {P2:.3f} atm")
print(f"Work done w = {w} J (zero due to fixed volume)")
