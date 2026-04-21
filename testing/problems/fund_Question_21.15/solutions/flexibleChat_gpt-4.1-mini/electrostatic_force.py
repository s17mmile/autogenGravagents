# filename: electrostatic_force.py
import math

# Constants
k = 8.988e9  # Coulomb's constant in N m^2/C^2

# Given charges in microcoulombs
q1_microC = 3.0
q2_microC = -4.0

# Convert charges to coulombs
q1 = q1_microC * 1e-6
q2 = q2_microC * 1e-6

# Given coordinates in centimeters
x1_cm, y1_cm = 3.5, 0.50
x2_cm, y2_cm = -2.0, 1.5

# Convert coordinates to meters
x1 = x1_cm * 1e-2
y1 = y1_cm * 1e-2
x2 = x2_cm * 1e-2
y2 = y2_cm * 1e-2

# Calculate distance between particles
r = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

if r == 0:
    raise ValueError("Particles occupy the same position; distance cannot be zero.")

# Calculate magnitude of electrostatic force
F = k * abs(q1 * q2) / r**2

# Output the result with units
print(f"Distance between particles: {r:.5f} meters")
print(f"Magnitude of electrostatic force on particle 2 due to particle 1: {F:.5e} newtons")

# Return force magnitude for potential further use
F
