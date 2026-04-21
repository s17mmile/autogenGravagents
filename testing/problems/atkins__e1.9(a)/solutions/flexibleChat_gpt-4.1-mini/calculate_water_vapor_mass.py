# filename: calculate_water_vapor_mass.py

def calculate_water_vapor_mass(volume_m3, temperature_celsius, relative_humidity):
    """Calculate the mass of water vapor in a room given volume, temperature, and relative humidity."""
    # Saturation vapor pressure at 27 degrees Celsius in kPa (can be adjusted for other temperatures)
    P_sat_kPa = 3.565

    # Calculate actual vapor pressure in kPa
    P_actual_kPa = relative_humidity * P_sat_kPa

    # Convert vapor pressure from kPa to Pa
    P_actual_Pa = P_actual_kPa * 1000

    # Molar mass of water vapor in kg/mol
    M = 0.01801528

    # Universal gas constant in J/(mol*K)
    R = 8.314

    # Convert temperature to Kelvin
    T = temperature_celsius + 273.15

    # Calculate density of water vapor using ideal gas law: rho = (P * M) / (R * T)
    rho = (P_actual_Pa * M) / (R * T)  # kg/m^3

    # Calculate mass of water vapor
    mass = rho * volume_m3  # kg

    return mass


# Given parameters
volume = 400  # m^3
temperature = 27  # degrees Celsius
relative_humidity = 0.60  # 60%

# Calculate mass
mass_water_vapor = calculate_water_vapor_mass(volume, temperature, relative_humidity)

# Print the result
print(f"Mass of water vapor in the room: {mass_water_vapor:.3f} kg")
