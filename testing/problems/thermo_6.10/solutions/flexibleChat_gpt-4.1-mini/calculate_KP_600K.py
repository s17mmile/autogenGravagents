# filename: calculate_KP_600K.py
import math

def calculate_KP_T2(KP_T1, delta_H, T1, T2, R=8.314):
    """Calculate equilibrium constant K_P at temperature T2 using van't Hoff equation."""
    ln_KP_ratio = -delta_H / R * (1/T2 - 1/T1)
    KP_T2 = KP_T1 * math.exp(ln_KP_ratio)
    return KP_T2

# Given values
KP_298 = 0.11  # equilibrium constant at 298 K
Delta_H = 58000  # standard reaction enthalpy in J/mol
T1 = 298  # reference temperature in K
T2 = 600  # target temperature in K

KP_600 = calculate_KP_T2(KP_298, Delta_H, T1, T2)
print(f"Equilibrium constant K_P at {T2} K: {KP_600:.4f}")
