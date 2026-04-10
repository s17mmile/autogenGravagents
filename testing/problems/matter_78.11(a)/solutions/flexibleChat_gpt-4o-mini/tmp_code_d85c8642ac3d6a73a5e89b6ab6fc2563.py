import math

def calculate_mass_loss(diameter_mm, molar_mass_g_mol, vapor_pressure_Pa, R, T):
    # Step 1: Calculate the area of the hole
    radius_m = (diameter_mm / 2) * 1e-3  # convert mm to m
    area_m2 = math.pi * (radius_m ** 2)  # area in m^2

    # Step 2: Calculate the effusion rate (number of moles per second)
    effusion_rate_mol_s = (vapor_pressure_Pa * area_m2) / (R * T)

    # Step 3: Calculate the total moles effused in 1 hour (3600 seconds)
    total_moles = effusion_rate_mol_s * 3600  # total moles in 1 hour

    # Step 4: Calculate the mass loss
    mass_loss_g = total_moles * molar_mass_g_mol  # mass loss in grams
    return mass_loss_g

# Constants
R = 8.314  # ideal gas constant in J/(mol*K)
T = 500  # temperature in K

# Example parameters
mass_loss = calculate_mass_loss(1.50, 300, 0.735, R, T)

# Output the result
print(f'Mass loss over 1 hour: {mass_loss:.6f} g')