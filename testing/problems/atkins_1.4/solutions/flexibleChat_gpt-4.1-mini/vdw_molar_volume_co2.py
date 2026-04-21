# filename: vdw_molar_volume_co2.py
import numpy as np
from numpy import roots

# Given constants with units
P = 100  # pressure in atm
T = 500  # temperature in K
R = 0.08206  # gas constant in L atm / (mol K)
a = 3.59  # van der Waals constant a for CO2 in L^2 atm / mol^2
b = 0.0427  # van der Waals constant b for CO2 in L / mol

# Coefficients of the cubic equation: P*V_m^3 - (P*b + R*T)*V_m^2 + a*V_m - a*b = 0
coefficients = [P, -(P*b + R*T), a, -a*b]

# Find roots of the cubic equation
roots_all = roots(coefficients)

# Filter real roots (imaginary part close to zero)
real_roots = [r.real for r in roots_all if abs(r.imag) < 1e-6]

# Filter positive roots (physically meaningful molar volumes)
positive_roots = [r for r in real_roots if r > 0]

if positive_roots:
    # Choose the smallest positive root as molar volume (stable phase)
    molar_volume = min(positive_roots)
    print(f"Estimated molar volume of CO2 at {T} K and {P} atm: {molar_volume:.4f} L/mol")
else:
    print("No physically meaningful positive molar volume root found.")
