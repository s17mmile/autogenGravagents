# filename: calculate_temperature_rise.py

def calculate_temperature_rise(heat_energy_joules, mass_kg, specific_heat_capacity):
    """
    Calculate the temperature rise of a body treated as an isolated system.

    Parameters:
    heat_energy_joules (float): Heat energy produced in joules.
    mass_kg (float): Mass of the body in kilograms.
    specific_heat_capacity (float): Specific heat capacity in J/kg.K.

    Returns:
    float: Temperature rise in degrees Celsius.
    """
    return heat_energy_joules / (mass_kg * specific_heat_capacity)


# Given data
heat_energy_joules = 10e6  # 10 MJ in joules
mass_kg = 65  # mass of the human body in kg
specific_heat_capacity = 4186  # specific heat capacity of water in J/kg.K

# Calculate temperature rise
temperature_rise = calculate_temperature_rise(heat_energy_joules, mass_kg, specific_heat_capacity)

# Output the result
print(f"Temperature rise of the body: {temperature_rise:.4f} degrees Celsius")
