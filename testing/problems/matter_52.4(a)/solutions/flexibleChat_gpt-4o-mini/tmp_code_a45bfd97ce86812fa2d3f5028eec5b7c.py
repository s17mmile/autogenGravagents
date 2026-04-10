import numpy as np

# Constants
h = 6.626e-34  # Planck's constant in J*s
c = 2.998e8    # Speed of light in m/s
k = 1.381e-23  # Boltzmann's constant in J/K
T = 298        # Temperature in K (25 degrees Celsius)

# Function to calculate the rotational partition function

def calculate_rotational_partition_function(A_cm, B_cm, C_cm, T):
    # Convert rotational constants from cm^-1 to J
    A_J = A_cm * h * c * 100  # Conversion factor: 1 cm^-1 = h * c * 100
    B_J = B_cm * h * c * 100
    C_J = C_cm * h * c * 100

    # Calculate characteristic rotational temperatures
    Theta_A = h * c / (k * A_J)
    Theta_B = h * c / (k * B_J)
    Theta_C = h * c / (k * C_J)

    # Calculate the rotational partition function
    Z_rot = (T / Theta_A)**(1/2) * (T / Theta_B)**(1/2) * (T / Theta_C)**(1/2)
    return Z_rot

# Rotational constants in cm^-1
A_cm = 3.1752
B_cm = 0.3951
C_cm = 0.3505

# Calculate and output the rotational partition function
Z_rot = calculate_rotational_partition_function(A_cm, B_cm, C_cm, T)
print(f'The rotational partition function Z_rot is approximately: {Z_rot:.2e}')