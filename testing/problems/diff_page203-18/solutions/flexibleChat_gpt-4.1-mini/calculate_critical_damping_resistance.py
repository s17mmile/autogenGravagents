# filename: calculate_critical_damping_resistance.py
import math

# Given values
C = 0.8e-6  # Capacitance in farads
L = 0.2     # Inductance in henrys

# Calculate resistance for critical damping
R = 2 * math.sqrt(L / C)

print(f"The resistance R for critical damping is {R:.2f} ohms.")
