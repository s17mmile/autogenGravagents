# filename: calculate_cosmic_ray_current.py
import math

# Constants
EARTH_RADIUS_M = 6.371e6  # meters
PROTON_FLUX_PER_M2_PER_S = 1500  # protons per second per square meter
CHARGE_PROTON_COULOMBS = 1.602e-19  # coulombs


def calculate_cosmic_ray_current():
    """Calculate the total electric current intercepted by Earth's surface due to cosmic ray protons."""
    earth_surface_area_m2 = 4 * math.pi * EARTH_RADIUS_M ** 2
    total_protons_per_s = PROTON_FLUX_PER_M2_PER_S * earth_surface_area_m2
    total_current_amperes = total_protons_per_s * CHARGE_PROTON_COULOMBS
    return total_current_amperes


if __name__ == '__main__':
    current = calculate_cosmic_ray_current()
    print(f"Total electric current intercepted by Earth's surface: {current:.6e} A")
