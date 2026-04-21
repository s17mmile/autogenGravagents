# filename: calculate_effusion_mass_loss.py
import math
from scipy.constants import R

def calculate_mass_loss(diameter_mm, molar_mass_g_per_mol, pressure_Pa, temperature_K, time_s):
    """
    Calculate mass loss due to effusion through a circular hole.

    Parameters:
    diameter_mm (float): Diameter of the hole in millimeters.
    molar_mass_g_per_mol (float): Molar mass of the solid in grams per mole.
    pressure_Pa (float): Vapor pressure in Pascals.
    temperature_K (float): Temperature in Kelvin.
    time_s (float): Time period in seconds.

    Returns:
    float: Mass loss in grams.
    """
    if diameter_mm <= 0 or molar_mass_g_per_mol <= 0 or pressure_Pa <= 0 or temperature_K <= 0 or time_s <= 0:
        raise ValueError("All input parameters must be positive numbers.")

    diameter_m = diameter_mm * 1e-3
    area = math.pi * (diameter_m / 2) ** 2
    molar_mass_kg_per_mol = molar_mass_g_per_mol / 1000
    rate_mol_per_s = (pressure_Pa * area) / math.sqrt(2 * math.pi * molar_mass_kg_per_mol * R * temperature_K)
    total_moles = rate_mol_per_s * time_s
    mass_loss_kg = total_moles * molar_mass_kg_per_mol
    mass_loss_g = mass_loss_kg * 1000
    return mass_loss_g

# Given values
diameter_mm = 1.50
molar_mass_g_per_mol = 300
pressure_Pa = 0.735
temperature_K = 500
time_s = 3600  # 1 hour

mass_loss = calculate_mass_loss(diameter_mm, molar_mass_g_per_mol, pressure_Pa, temperature_K, time_s)

print(f"Mass loss due to effusion over 1 hour: {mass_loss:.6f} grams")

with open('mass_loss_result.txt', 'w') as f:
    f.write(f'Mass loss due to effusion over 1 hour: {mass_loss:.6f} grams\n')
