# filename: calculate_total_charge_sphere.py
import math

# Given values
R_cm = 5.60  # radius in centimeters
R = R_cm / 100  # convert radius to meters

# Charge density coefficient (14.1 pC/m^3) converted to C/m^3
# The volume charge density varies as rho(r) = rho_coefficient * (r / R)
rho_coefficient = 14.1e-12  # in Coulombs per cubic meter

# Total charge Q derived from integration:
# Q = \u03C0 * rho_coefficient * R^3
Q = math.pi * rho_coefficient * R**3  # total charge in Coulombs
Q_pC = Q * 1e12  # convert charge to picoCoulombs

# Output the result
print(f"Total charge Q = {Q:.4e} C")
print(f"Total charge Q = {Q_pC:.4f} pC")
