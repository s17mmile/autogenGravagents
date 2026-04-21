# filename: surface_charge_density.py
import math

# Given values
diameter = 1.3  # meters
charge_microC = 2.4  # microCoulombs

# Convert charge to Coulombs
charge_C = charge_microC * 1e-6

# Calculate radius
radius = diameter / 2

# Calculate surface area of the sphere
surface_area = 4 * math.pi * radius ** 2

# Calculate surface charge density
surface_charge_density = charge_C / surface_area

# Print the result with units
print(f"Surface charge density: {surface_charge_density:.6e} C/m^2")
