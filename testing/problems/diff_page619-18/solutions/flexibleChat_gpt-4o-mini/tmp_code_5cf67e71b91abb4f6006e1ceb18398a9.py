import numpy as np

# Constants
L = 0.2  # Length of the rod in meters
T0 = 0  # Boundary temperature in degrees Celsius
Ti = 100  # Initial temperature in degrees Celsius
alpha = 406 / (10490 * 235)  # Thermal diffusivity in m^2/s

t = 30  # Time in seconds

# Function to calculate temperature at the center of the rod

def temperature_at_center(t, L, T0, Ti, alpha, n_terms=10):
    """
    Calculate the temperature at the center of a metallic rod over time.
    
    Parameters:
    t (float): Time in seconds.
    L (float): Length of the rod in meters.
    T0 (float): Boundary temperature in degrees Celsius.
    Ti (float): Initial temperature in degrees Celsius.
    alpha (float): Thermal diffusivity in m^2/s.
    n_terms (int): Number of terms in the series to consider.
    
    Returns:
    float: Temperature at the center of the rod.
    """
    sum_terms = 0
    for n in range(1, n_terms + 1):
        lambda_n = (n**2 * np.pi**2 / L**2) * alpha
        term = np.exp(-lambda_n * t) * np.sin(n * np.pi / 2)
        sum_terms += term
    T_center = T0 + (Ti - T0) * sum_terms
    return T_center

# Calculate temperature at the center
T_center = temperature_at_center(t, L, T0, Ti, alpha)

# Output the result
print(f'Temperature at the center of the rod after {t} seconds: {T_center:.2f} °C')