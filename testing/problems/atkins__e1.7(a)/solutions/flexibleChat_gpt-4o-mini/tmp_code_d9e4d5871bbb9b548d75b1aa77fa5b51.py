def calculate_gas_constant(mass_helium_g, volume_dm3, pressure_cm_water, temperature_C):
    # Constants
    molar_mass_helium_g_per_mol = 4.0026  # Molar mass of helium in g/mol
    water_density_g_per_cm3 = 0.99707  # Density of water in g/cm^3
    g = 9.81  # Acceleration due to gravity in m/s^2

    # Step 1: Validate inputs
    if mass_helium_g <= 0 or volume_dm3 <= 0 or pressure_cm_water < 0 or temperature_C < -273.15:
        raise ValueError('Invalid input values. Mass, volume must be positive, pressure cannot be negative, and temperature must be above absolute zero.')

    # Step 2: Convert volume to liters
    volume_L = volume_dm3  # 1 dm^3 = 1 L

    # Step 3: Calculate number of moles of helium
    n_moles = mass_helium_g / molar_mass_helium_g_per_mol

    # Step 4: Convert pressure from cm of water to atm
    # Pressure in kPa = height in cm * density in g/cm^3 * g (acceleration due to gravity) / 1000
    pressure_kPa = pressure_cm_water * water_density_g_per_cm3 * g / 1000
    # Convert kPa to atm
    pressure_atm = pressure_kPa / 101.325

    # Step 5: Convert temperature to Kelvin
    temperature_K = temperature_C + 273.15

    # Step 6: Calculate R using the ideal gas law
    R = (pressure_atm * volume_L) / (n_moles * temperature_K)
    return R

# Example usage
mass_helium = 0.25132  # in grams
volume = 20.000  # in dm^3
pressure = 206.402  # in cm of water
temperature = 500  # in Celsius

# Calculate R
R_value = calculate_gas_constant(mass_helium, volume, pressure, temperature)
print(f'The calculated value of the gas constant R is: {R_value:.5f} L·atm/(mol·K)')