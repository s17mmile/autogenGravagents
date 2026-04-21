# filename: millikan_charge_calculation.py
import math

def calculate_charge_on_oil_drop(radius_micrometers=1.64, density_g_per_cm3=0.851, electric_field=1.92e5):
    """Calculate the charge on an oil drop suspended in an electric field in terms of elementary charge e."""
    # Constants
    g = 9.8  # m/s^2, acceleration due to gravity
    e = 1.6e-19  # C, elementary charge

    # Convert units
    radius_meters = radius_micrometers * 1e-6  # micrometers to meters
    density = density_g_per_cm3 * 1000  # g/cm^3 to kg/m^3

    # Calculate volume of the oil drop
    volume = (4/3) * math.pi * radius_meters**3

    # Calculate mass of the oil drop
    mass = density * volume

    # Calculate weight (gravitational force)
    weight = mass * g

    # Calculate charge q on the drop
    q = weight / electric_field

    # Express charge in terms of elementary charge e
    charge_in_terms_of_e = q / e

    return charge_in_terms_of_e

charge_e_units = calculate_charge_on_oil_drop()
print(f"Charge on the oil drop in terms of e: {charge_e_units:.2f} e")
