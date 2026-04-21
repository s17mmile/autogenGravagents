# filename: rotational_partition_function_SO2.py
from math import sqrt, pi
from scipy.constants import k, h, c

# Function to calculate rotational partition function for an asymmetric top molecule
# Inputs:
#   T: Temperature in Kelvin
#   B_A, B_B, B_C: Rotational constants in cm^-1 (must be positive)
# Returns:
#   q_rot: Rotational partition function (dimensionless)
def rotational_partition_function(T, B_A, B_B, B_C):
    # Validate inputs
    if B_A <= 0 or B_B <= 0 or B_C <= 0:
        raise ValueError("Rotational constants must be positive.")
    
    # Speed of light in cm/s (converted from m/s)
    c_cm_s = c * 100
    
    # Calculate q_rot using the formula for asymmetric top molecules
    q_rot = sqrt(pi) * ((k * T) / (h * c_cm_s))**(3/2) / sqrt(B_A * B_B * B_C)
    return q_rot

# Given rotational constants in cm^-1
B_A = 2.03
B_B = 0.344
B_C = 0.293

# Temperature in Kelvin
T = 298

# Calculate and print the rotational partition function
q_rot = rotational_partition_function(T, B_A, B_B, B_C)
print(f"Rotational partition function q_rot for SO2 at {T} K: {q_rot:.4e}")
