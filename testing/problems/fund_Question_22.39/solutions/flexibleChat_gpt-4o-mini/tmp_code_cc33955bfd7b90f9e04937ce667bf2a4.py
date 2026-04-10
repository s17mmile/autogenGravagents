import math

def calculate_charge_on_oil_drop():
    """
    Calculate the charge on an oil drop in Millikan's experiment.
    Returns the charge in Coulombs and in terms of elementary charge (e).
    """
    # Given parameters
    radius_m = 1.64e-6  # radius in meters
    density_kg_m3 = 0.851 * 1000  # density in kg/m^3
    electric_field_N_C = 1.92e5  # electric field in N/C
    g = 9.81  # acceleration due to gravity in m/s^2
    e = 1.602e-19  # elementary charge in C

    # Step 1: Calculate the volume of the oil drop
    volume_m3 = (4/3) * math.pi * (radius_m ** 3)

    # Step 2: Calculate the mass of the oil drop
    mass_kg = density_kg_m3 * volume_m3

    # Step 3: Calculate the gravitational force
    F_gravity = mass_kg * g

    # Step 4: Calculate the charge on the drop
    charge_C = F_gravity / electric_field_N_C

    # Step 5: Convert charge to elementary charge (e)
    charge_in_e = charge_C / e

    return charge_C, charge_in_e

# Calculate and print the charge in both representations
charge_C, charge_in_e = calculate_charge_on_oil_drop()
print(f'Charge in Coulombs: {charge_C:.2e} C')
print(f'Charge in terms of e: {charge_in_e:.2f} e')