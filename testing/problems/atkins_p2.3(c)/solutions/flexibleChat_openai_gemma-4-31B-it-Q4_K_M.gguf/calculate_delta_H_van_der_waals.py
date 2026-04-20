# filename: calculate_delta_H_van_der_waals.py

# Constants
R = 0.08314  # L bar mol^-1 K^-1
a = 3.59     # L^2 bar mol^-2
b = 0.0427   # L mol^-1
n = 2.0      # mol
V = 15.0     # L (volume in dm^3 = L)
T1 = 300     # K (initial temperature)
T2 = 341     # K (final temperature)
q_V = 2.35   # kJ (heat input at constant volume → dU = q_V)

# Conversion factor: 1 L bar = 0.1 kJ
L_bar_to_kJ = 0.1

# Check for valid volume (V > nb)
if V <= n * b:
    raise ValueError(f"Volume {V} L is less than or equal to nb = {n*b} L. Invalid for van der Waals equation.")

# Calculate pressures using van der Waals equation in explicit form
# P = (nRT / (V - nb)) - (a*n^2 / V^2)
P1 = (n * R * T1) / (V - n * b) - (a * n**2) / (V**2)
P2 = (n * R * T2) / (V - n * b) - (a * n**2) / (V**2)

# Calculate d(PV) in L*bar
Delta_PV_Lbar = V * (P2 - P1)

# Convert d(PV) to kJ
Delta_PV_kJ = Delta_PV_Lbar * L_bar_to_kJ

# dU is equal to heat input at constant volume
Delta_U = q_V  # kJ

# Calculate dH = dU + d(PV)
Delta_H = Delta_U + Delta_PV_kJ

# Output results with appropriate precision
print(f"Initial pressure P1: {P1:.4f} bar")
print(f"Final pressure P2: {P2:.4f} bar")
print(f"d(PV) in L*bar: {Delta_PV_Lbar:.4f} L*bar")
print(f"d(PV) in kJ: {Delta_PV_kJ:.4f} kJ")
print(f"dH: {Delta_H:.3f} kJ")  # Rounded to 3 significant figures