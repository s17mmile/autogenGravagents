# filename: calculate_rotational_temperature.py

# Constants
h = 6.626e-34  # Planck's constant in J*s
c = 2.998e10   # Speed of light in cm/s
k = 1.381e-23  # Boltzmann constant in J/K

# Given values
B = 8.46       # Rotational constant in cm^-1
J_max = 4      # Maximum intensity transition from J=4 to 5

# Calculate temperature T using the formula:
# T = (2 * h * c * B / k) * (J + 1/2)^2
J_plus_half = J_max + 0.5
T = (2 * h * c * B / k) * (J_plus_half ** 2)

print(f"Temperature at which the spectrum was obtained: {T:.2f} K")
