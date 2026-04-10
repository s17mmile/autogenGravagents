import math

def calculate_change_in_chemical_potential(R, T, P1, P2):
    if P1 <= 0:
        raise ValueError('Initial pressure P1 must be greater than zero.')
    # Calculate the change in chemical potential
    change_in_chemical_potential = R * T * math.log(P2 / P1)
    return change_in_chemical_potential

# Constants
R = 8.314  # J/(mol·K)
T = 40 + 273.15  # Convert Celsius to Kelvin
P1 = 1.8  # Initial pressure in atm
P2 = 29.5  # Final pressure in atm

# Calculate the change in chemical potential
Delta_mu = calculate_change_in_chemical_potential(R, T, P1, P2)

# Output the result
print(f'The change in chemical potential (Delta mu) is approximately {Delta_mu:.2f} J/mol.')