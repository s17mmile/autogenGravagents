# filename: calculate_moment_of_inertia_h2o.py
import math

def moment_of_inertia_h2o(bond_angle_deg=104.5, bond_length_pm=95.7):
    """
    Calculate the moment of inertia of H2O molecule around the bisector axis of the HOH angle.

    Parameters:
    bond_angle_deg (float): HOH bond angle in degrees
    bond_length_pm (float): O-H bond length in picometers

    Returns:
    float: Moment of inertia in kg*m^2
    """
    # Atomic masses in atomic mass units (u)
    mass_O_u = 15.999  # oxygen
    mass_H_u = 1.008   # hydrogen

    # Conversion factors
    pm_to_m = 1e-12  # picometers to meters
    u_to_kg = 1.66053906660e-27  # atomic mass unit to kg

    # Convert bond length to meters
    bond_length_m = bond_length_pm * pm_to_m

    # Calculate half angle from bisector to each hydrogen
    half_angle_rad = math.radians(bond_angle_deg / 2)

    # Perpendicular distance of each hydrogen from bisector axis
    r_h = bond_length_m * math.sin(half_angle_rad)

    # Masses in kg
    mass_H_kg = mass_H_u * u_to_kg

    # Oxygen lies on the axis, so its contribution to moment of inertia is zero

    # Moment of inertia calculation: sum over hydrogens
    I = 2 * mass_H_kg * r_h**2

    return I

# Calculate and print the moment of inertia
I = moment_of_inertia_h2o()
print(f"Moment of inertia of H2O around bisector axis: {I:.3e} kg m^2")

I