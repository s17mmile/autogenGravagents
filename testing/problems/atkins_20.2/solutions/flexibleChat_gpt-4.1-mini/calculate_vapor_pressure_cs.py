# filename: calculate_vapor_pressure_cs.py
import math

def calculate_vapor_pressure(mass_loss_g, hole_diameter_m, time_s, T_K, M_g_per_mol=132.91, R=8.314):
    """
    Calculate vapor pressure of liquid caesium using Hertz-Knudsen equation.
    Parameters:
        mass_loss_g: mass loss in grams
        hole_diameter_m: diameter of hole in meters
        time_s: time hole was open in seconds
        T_K: temperature in Kelvin
        M_g_per_mol: molar mass in g/mol (default for Cs)
        R: gas constant in J/mol/K
    Returns:
        Vapor pressure in Pascals
    """
    hole_radius_m = hole_diameter_m / 2
    area_m2 = math.pi * hole_radius_m**2
    mass_flux_g_per_m2_s = mass_loss_g / (area_m2 * time_s)
    M_kg_per_mol = M_g_per_mol / 1000
    mass_flux_kg_per_m2_s = mass_flux_g_per_m2_s / 1000
    Pv_Pa = mass_flux_kg_per_m2_s * math.sqrt(2 * math.pi * R * T_K / M_kg_per_mol)
    return Pv_Pa

# Note: The problem states heating to 500 degrees Celsius but uses 500 K for calculation.
# Here we use 500 K as given.

# Given data
hole_diameter_m = 0.50e-3  # meters
mass_loss_g = 0.385  # grams
time_s = 100  # seconds
T_K = 500  # temperature in Kelvin

vapor_pressure = calculate_vapor_pressure(mass_loss_g, hole_diameter_m, time_s, T_K)
print(f"Vapor pressure of liquid caesium at {T_K} K is {vapor_pressure:.2f} Pa")
