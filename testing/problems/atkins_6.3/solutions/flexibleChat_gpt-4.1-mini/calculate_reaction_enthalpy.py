# filename: calculate_reaction_enthalpy.py
import numpy as np
from scipy.stats import linregress

# Given data
T = np.array([350, 400, 450, 500], dtype=float)  # Temperature in K
K = np.array([3.98e-4, 1.41e-2, 1.86e-1, 1.48], dtype=float)  # Equilibrium constants

# Input validation
if T.size == 0 or K.size == 0:
    raise ValueError("Temperature and equilibrium constant arrays must not be empty.")
if T.size != K.size:
    raise ValueError("Temperature and equilibrium constant arrays must be of the same length.")
if np.any(K <= 0):
    raise ValueError("Equilibrium constants must be positive to compute their natural logarithm.")

try:
    # Calculate ln(K) and 1/T
    lnK = np.log(K)
    invT = 1 / T

    # Perform linear regression of ln(K) vs 1/T
    slope, intercept, r_value, p_value, std_err = linregress(invT, lnK)

    # Calculate standard reaction enthalpy Delta_H using van't Hoff equation
    # slope = -Delta_H / R, where R = 8.314 J/(mol K)
    R = 8.314  # Gas constant in J/(mol K)
    Delta_H = -slope * R  # in J/mol

    # Convert to kJ/mol
    Delta_H_kJ = Delta_H / 1000

    # Output the result with units
    print(f"Standard reaction enthalpy (Delta_H) = {Delta_H_kJ:.2f} kJ/mol")
    print("Calculated from the slope of ln(K) vs 1/T using the van't Hoff equation.")
except Exception as e:
    print(f"An error occurred during calculation: {e}")
