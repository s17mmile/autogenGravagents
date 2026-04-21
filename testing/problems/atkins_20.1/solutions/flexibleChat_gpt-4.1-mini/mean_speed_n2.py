# filename: mean_speed_n2.py

import math

def mean_speed_n2(T_celsius=25):
    """
    Calculate the mean speed of N2 molecules at a given temperature in Celsius.

    Parameters:
    T_celsius (float): Temperature in degrees Celsius (default is 25).

    Returns:
    float: Mean speed in meters per second.
    """
    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    N_A = 6.02214076e23  # Avogadro's number in mol^-1
    M = 0.028  # Molar mass of N2 in kg/mol

    # Convert temperature to Kelvin
    T = T_celsius + 273.15  # Temperature in Kelvin

    # Mass of one N2 molecule
    m = M / N_A  # Mass in kg

    # Mean speed formula
    mean_speed = math.sqrt((8 * k_B * T) / (math.pi * m))

    return mean_speed

if __name__ == '__main__':
    speed = mean_speed_n2()
    print(f'Mean speed of N2 molecules at 25 C: {speed:.2f} m/s')
