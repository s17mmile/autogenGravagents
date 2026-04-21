# filename: calculate_enthalpy_increase_so2.py
from sympy import symbols, integrate

def calculate_enthalpy_increase(a, b, c, d, T1=298.15, T2=1500):
    T = symbols('T')
    Cp = a + b*T + c*T**2 + d*T**3
    delta_H_expr = integrate(Cp, (T, T1, T2))
    delta_H = delta_H_expr.evalf()
    return delta_H

# Example usage:
# Replace these coefficients with actual values from a reliable source
coefficients = {
    'a': 30.0,  # example coefficient in J/(mol*K)
    'b': 0.01,  # example coefficient in J/(mol*K^2)
    'c': -1e-6, # example coefficient in J/(mol*K^3)
    'd': 2e-9   # example coefficient in J/(mol*K^4)
}

enthalpy_increase = calculate_enthalpy_increase(
    coefficients['a'],
    coefficients['b'],
    coefficients['c'],
    coefficients['d']
)

print(f"Increase in standard molar enthalpy of SO2 from 298.15 K to 1500 K: {enthalpy_increase:.2f} J/mol")
