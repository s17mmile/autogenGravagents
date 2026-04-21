# filename: vapor_pressure_solution_function.py

"""
Calculate the vapor pressure of an ideal binary solution using Raoult's Law.
Raoult's Law states that the total vapor pressure is the sum of the partial pressures,
which are the product of the mole fraction and the pure component vapor pressure.
"""

def calculate_vapor_pressure(x_benzene, P_benzene_pure, P_hexane_pure):
    # Validate mole fraction input
    if not (0 <= x_benzene <= 1):
        raise ValueError("Mole fraction of benzene must be between 0 and 1.")
    x_hexane = 1.0 - x_benzene
    if not (0 <= x_hexane <= 1):
        raise ValueError("Calculated mole fraction of hexane is out of bounds.")

    # Calculate total vapor pressure using Raoult's Law
    P_solution = x_benzene * P_benzene_pure + x_hexane * P_hexane_pure
    return P_solution


# Given data
P_benzene_pure = 120.0  # vapor pressure of pure benzene in Torr
P_hexane_pure = 189.0   # vapor pressure of pure hexane in Torr
x_benzene = 0.28        # mole fraction of benzene

# Calculate vapor pressure of the solution
P_solution = calculate_vapor_pressure(x_benzene, P_benzene_pure, P_hexane_pure)

# Output the result
print(f"Vapor pressure of the solution at 303 K: {P_solution:.2f} Torr")
