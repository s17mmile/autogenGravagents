# filename: calculate_pressure_mean_free_path.py
import math

# Constants
k_B = 1.38e-23  # Boltzmann constant in J/K


def calculate_pressure_mean_free_path(volume_cm3, temperature_C, sigma_nm2):
    """
    Calculate the pressure at which the mean free path of argon gas equals the diameter of a spherical vessel.

    Parameters:
    volume_cm3 (float): Volume of the vessel in cubic centimeters.
    temperature_C (float): Temperature in degrees Celsius.
    sigma_nm2 (float): Collision cross-section in nm^2.

    Returns:
    tuple: (diameter_m, pressure_Pa, pressure_atm)
    """
    # Validate inputs
    if volume_cm3 <= 0:
        raise ValueError("Volume must be positive.")
    if sigma_nm2 <= 0:
        raise ValueError("Collision cross-section must be positive.")

    # Convert units
    V_m3 = volume_cm3 * 1e-6  # cm^3 to m^3
    T = temperature_C + 273.15  # Celsius to Kelvin
    sigma_m2 = sigma_nm2 * 1e-18  # nm^2 to m^2

    # Calculate diameter of spherical vessel: V = (pi/6)*d^3 => d = (6V/pi)^(1/3)
    d = (6 * V_m3 / math.pi) ** (1/3)

    # Calculate pressure using P = (k_B * T) / (sqrt(2) * sigma * d)
    P = (k_B * T) / (math.sqrt(2) * sigma_m2 * d)  # pressure in Pascals

    # Convert pressure to atmospheres (1 atm = 101325 Pa)
    P_atm = P / 101325

    return d, P, P_atm


# Example usage
volume_cm3 = 100
temperature_C = 20
sigma_nm2 = 0.36

diameter_m, pressure_Pa, pressure_atm = calculate_pressure_mean_free_path(volume_cm3, temperature_C, sigma_nm2)

print(f"Vessel diameter (m): {diameter_m:.6e}")
print(f"Pressure (Pa) at which mean free path equals vessel diameter: {pressure_Pa:.6e}")
print(f"Pressure (atm) at which mean free path equals vessel diameter: {pressure_atm:.6e}")
