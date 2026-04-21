# filename: calculate_diffusion_time.py

def calculate_diffusion_time(D, x):
    """Calculate diffusion time for a molecule to travel distance x with diffusion coefficient D.
    Assumes simple diffusion without barriers or active transport.
    Args:
        D (float): Diffusion coefficient in m^2/s (must be positive).
        x (float): Distance in meters (must be positive).
    Returns:
        float: Diffusion time in seconds.
    Raises:
        ValueError: If D or x is not positive.
    """
    if D <= 0 or x <= 0:
        raise ValueError("Diffusion coefficient and distance must be positive numbers.")
    t = (x ** 2) / (2 * D)
    return t

# Given values
D = 1.0e-11  # diffusion coefficient in m^2/s
x = 1.0e-6   # distance in meters (1 micrometer)

# Calculate diffusion time
time_seconds = calculate_diffusion_time(D, x)

# Print the result
print(f"Diffusion time: {time_seconds:.3e} seconds")
print(f"Diffusion time: {time_seconds * 1000:.3f} milliseconds")
