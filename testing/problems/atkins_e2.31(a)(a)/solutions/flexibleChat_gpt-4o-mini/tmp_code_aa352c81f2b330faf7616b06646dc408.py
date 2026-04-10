import numpy as np

# Constants
R = 8.314  # J/(mol K)
T = 298.15  # K
V_i = 1.00e-3  # Initial volume in m^3
V_f = 24.8e-3  # Final volume in m^3

# Van der Waals constants for nitrogen
# a in m^6 Pa / mol^2, b in m^3 / mol
a = 0.139  # m^6 Pa / mol^2
b = 3.91e-5  # m^3 / mol

def calculate_number_of_moles(P, V, R, T):
    return (P * V) / (R * T)  # Ideal gas law

def calculate_work(V_i, V_f, a, b, R, T):
    if V_i <= b or V_f <= b:
        raise ValueError('Volumes must be greater than b to avoid logarithm of zero or negative.')
    W = (R * T * (np.log(V_f - b) - np.log(V_i - b))) - (a * (-1/V_f + 1/V_i))
    return W

def calculate_average_pressure(V_i, V_f, a, b, R, T):
    P_i = (R * T) / (V_i - b) - (a / V_i**2)
    P_f = (R * T) / (V_f - b) - (a / V_f**2)
    return (P_i + P_f) / 2

try:
    # Calculate number of moles
    P_i = 101325  # Initial pressure in Pa (1 atm)
    n = calculate_number_of_moles(P_i, V_i, R, T)

    # Calculate work done during isothermal expansion
    W = calculate_work(V_i, V_f, a, b, R, T)

    # Calculate average pressure
    P_avg = calculate_average_pressure(V_i, V_f, a, b, R, T)

    # Output results
    print(f'Work done during expansion (W): {W:.2f} J')
    print(f'Average pressure during expansion (P_avg): {P_avg:.2f} Pa')

except ValueError as e:
    print(f'Error: {e}')
except Exception as e:
    print(f'An unexpected error occurred: {e}')