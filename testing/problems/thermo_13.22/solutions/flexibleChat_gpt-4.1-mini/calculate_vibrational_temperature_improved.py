# filename: calculate_vibrational_temperature_improved.py

import math
from scipy.constants import h, c, k

# Function to calculate temperature at which the population of the first excited vibrational state
# is half that of the ground state, given vibrational frequency in cm^-1

def calculate_temperature(vibrational_frequency_cm):
    """
    Calculate temperature T (in Kelvin) where N1/N0 = 1/2 for a vibrational frequency.
    Parameters:
        vibrational_frequency_cm (float): Vibrational frequency in inverse centimeters (cm^-1)
    Returns:
        float: Temperature in Kelvin
    """
    # Convert frequency from cm^-1 to m^-1
    vibrational_frequency_m = vibrational_frequency_cm * 100

    # Calculate energy difference E = h * c * frequency
    E = h * c * vibrational_frequency_m

    # Calculate temperature T = E / (k_B * ln(2))
    T = E / (k * math.log(2))
    return T

# Given vibrational frequency for I2
vibrational_frequency_I2 = 208

# Calculate and print the temperature
temperature = calculate_temperature(vibrational_frequency_I2)
print(f"Temperature at which population of first excited state is half that of ground state: {temperature:.2f} K")
