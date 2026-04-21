# filename: temperature_center_silver_rod.py
import numpy as np

def temperature_center(t, L=0.2, T_i=100, alpha=1.74e-5, terms=10):
    """
    Calculate temperature at the center of the rod at time t.
    Parameters:
    - t: time in seconds (must be positive)
    - L: length of the rod in meters (must be positive)
    - T_i: initial uniform temperature in degrees Celsius
    - alpha: thermal diffusivity in m^2/s (must be positive)
    - terms: number of terms in Fourier series (positive integer)
    Returns:
    - Temperature at center (x=L/2) at time t
    """
    if t < 0 or L <= 0 or alpha <= 0 or terms <= 0:
        raise ValueError("Time, length, thermal diffusivity, and terms must be positive.")
    x = L / 2
    n_values = np.arange(1, 2*terms, 2)  # odd terms
    B_n = 4 / (n_values * np.pi)
    sin_terms = np.sin(n_values * np.pi * x / L)
    exp_terms = np.exp(-alpha * (n_values * np.pi / L)**2 * t)
    temp_sum = np.sum(B_n * sin_terms * exp_terms)
    return T_i * temp_sum

# Parameters
L = 0.2  # meters
T_i = 100  # degrees Celsius
alpha = 1.74e-5  # m^2/s for silver

t = 30  # seconds

# Calculate temperature at center
T_center = temperature_center(t, L, T_i, alpha, terms=10)

# Save result to a file
with open('temperature_center_result.txt', 'w') as f:
    f.write(f'Temperature at center after {t} seconds: {T_center:.2f} degrees Celsius\n')
