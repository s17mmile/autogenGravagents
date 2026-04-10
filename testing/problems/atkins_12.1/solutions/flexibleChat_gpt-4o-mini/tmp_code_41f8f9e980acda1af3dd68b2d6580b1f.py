import math

# Constants
mass_H_u = 1.00784  # mass of hydrogen in atomic mass units
mass_O_u = 15.999  # mass of oxygen in atomic mass units
u_to_kg = 1.66053906660e-27  # conversion factor from atomic mass units to kg

def calculate_moment_of_inertia(bond_angle_deg, bond_length_pm):
    # Error handling for input values
    if bond_length_pm <= 0:
        raise ValueError('Bond length must be positive.')
    if not (0 < bond_angle_deg < 180):
        raise ValueError('Bond angle must be between 0 and 180 degrees.')

    # Convert bond length to meters
    bond_length_m = bond_length_pm * 1e-12

    # Convert masses to kg
    mass_H_kg = mass_H_u * u_to_kg
    mass_O_kg = mass_O_u * u_to_kg

    # Calculate the angle in radians
    bond_angle_rad = math.radians(bond_angle_deg)

    # Calculate the coordinates of the hydrogen atoms
    angle_half_rad = bond_angle_rad / 2
    r1 = bond_length_m * math.sin(angle_half_rad)  # x-coordinate for H1
    r2 = -bond_length_m * math.sin(angle_half_rad)  # x-coordinate for H2

    # Calculate the moment of inertia
    I = 2 * mass_H_kg * (r1 ** 2)  # since r1 = -r2
    return I

# Example usage
bond_angle = 104.5  # HOH bond angle in degrees
bond_length = 95.7  # bond length in picometers
moment_of_inertia = calculate_moment_of_inertia(bond_angle, bond_length)

# Output the result
print(f'Moment of Inertia of H2O around the bisector axis: {moment_of_inertia:.3e} kg m²')