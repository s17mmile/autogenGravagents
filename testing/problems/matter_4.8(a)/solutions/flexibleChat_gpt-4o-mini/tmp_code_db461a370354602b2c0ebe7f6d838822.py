import numpy as np
from scipy.optimize import fsolve

# Constants
h = 6.626e-34  # Planck's constant in Js
m = 9.109e-31  # Mass of electron in kg
c = 3.00e8     # Speed of light in m/s
wavelength = 100e-12  # Wavelength in meters (100 pm)

# Function to calculate speed using the relativistic de Broglie wavelength

def calculate_speed(h, m, wavelength, c):
    # Function to find the root
    def equation(v):
        gamma = 1 / np.sqrt(1 - (v**2 / c**2))
        return h / (m * wavelength * gamma) - v

    # Initial guess for speed
    initial_guess = 0.9 * c
    speed, = fsolve(equation, initial_guess)
    return speed

# Calculate the speed
speed = calculate_speed(h, m, wavelength, c)

# Print the result
print(f'The speed required for an electron to have a wavelength of 100 pm is approximately {speed:.2e} m/s.')