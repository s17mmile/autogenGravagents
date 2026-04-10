import numpy as np

# Constants
L = 1.0  # Length of the string (in meters)
max_velocity = 1.0  # Maximum velocity at x = L/4 (in m/s)

# Amplitude ratio for second harmonic
amplitude_ratio = 0.5  # Assuming second harmonic has half the amplitude of the fundamental

# Initial conditions
# Velocity profile function
def velocity_profile(x):
    if 0 <= x <= L/4:
        return (4 * max_velocity / L) * x
    else:
        return 0

# Calculate amplitudes based on the initial velocity profile
A1 = velocity_profile(L/4)  # Amplitude of the fundamental
A2 = A1 * amplitude_ratio  # Amplitude of the second harmonic

# Calculate decibel difference
def decibel_difference(A1, A2):
    if A1 == 0:
        return float('inf')  # Return infinity if A1 is zero to avoid division by zero
    return 20 * np.log10(A2 / A1)

# Calculate the decibel difference between second harmonic and fundamental
decibel_diff = decibel_difference(A1, A2)

# Output results
print(f"Amplitude of Fundamental (A1): {A1} m/s")
print(f"Amplitude of Second Harmonic (A2): {A2} m/s")
print(f"Decibel difference (A2 from A1): {decibel_diff} dB")