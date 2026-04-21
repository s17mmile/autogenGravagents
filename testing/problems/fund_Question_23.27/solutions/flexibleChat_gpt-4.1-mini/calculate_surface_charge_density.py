# filename: calculate_surface_charge_density.py
import math

def calculate_sigma(linear_charge_density, radius):
    """Calculate surface charge density sigma on a cylindrical shell.

    Args:
        linear_charge_density (float): Linear charge density lambda in C/m.
        radius (float): Radius of the cylindrical shell in meters.

    Returns:
        float: Surface charge density sigma in C/m^2.

    Raises:
        ValueError: If radius is not positive.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive and non-zero.")
    sigma = -linear_charge_density / (2 * math.pi * radius)
    return sigma

if __name__ == '__main__':
    # Given values
    lambda_coulombs_per_meter = -3.6e-9  # -3.6 nC/m
    radius_meters = 0.015  # 1.5 cm in meters

    sigma = calculate_sigma(lambda_coulombs_per_meter, radius_meters)
    print(f"Surface charge density sigma = {sigma:.3e} C/m^2")
