# filename: autoclave_pressure_calculation_improved.py
import math

def antoine_pressure(T_celsius):
    """
    Calculate vapor pressure of water at temperature T_celsius using Antoine equation.
    Returns pressure in kPa.
    Antoine constants valid approximately from 99 to 374 C (valid for 120 C):
    log10(P) = A - B / (C + T)
    P in mmHg, T in Celsius
    Constants from NIST for water (99-374 C):
    A = 8.14019
    B = 1810.94
    C = 244.485
    Note: Accuracy decreases outside this range.
    """
    A = 8.14019
    B = 1810.94
    C = 244.485
    log10_P_mmHg = A - (B / (C + T_celsius))
    P_mmHg = 10 ** log10_P_mmHg
    # Convert mmHg to kPa (1 mmHg = 0.133322 kPa)
    P_kPa = P_mmHg * 0.133322
    return P_kPa

# Temperature for autoclave sterilization
T_autoclave = 120.0
pressure_kPa = antoine_pressure(T_autoclave)
pressure_atm = pressure_kPa / 101.325

# Print the pressure required in kPa and atm
print(f"Pressure required to boil water at {T_autoclave} degrees Celsius:")
print(f"{pressure_kPa:.2f} kPa")
print(f"{pressure_atm:.2f} atm")
