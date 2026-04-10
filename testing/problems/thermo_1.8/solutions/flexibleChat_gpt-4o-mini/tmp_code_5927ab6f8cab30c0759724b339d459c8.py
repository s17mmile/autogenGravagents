def calculate_volume_increase(mass_water_g, vapor_pressure_torr, temperature_c):
    """
    Calculate the volume increase necessary for all the water to evaporate.

    Parameters:
    mass_water_g (float): Mass of water in grams.
    vapor_pressure_torr (float): Vapor pressure of water in torr.
    temperature_c (float): Temperature in degrees Celsius.

    Returns:
    tuple: Volume increase (L), total vapor volume (L), initial liquid volume (L).
    """
    # Input validation
    if mass_water_g < 0:
        raise ValueError('Mass of water must be non-negative.')
    if vapor_pressure_torr < 0:
        raise ValueError('Vapor pressure must be non-negative.')

    # Constants
    molar_mass_water = 18.015  # g/mol
    R = 0.0821  # L·atm/(K·mol)
    torr_to_atm = 760  # torr/atm

    # Calculate number of moles of water
    n = mass_water_g / molar_mass_water

    # Convert vapor pressure from torr to atm
    P = vapor_pressure_torr / torr_to_atm

    # Convert temperature from Celsius to Kelvin
    T = temperature_c + 273.15

    # Calculate volume using Ideal Gas Law
    V_vapor = (n * R * T) / P

    # Calculate initial volume of liquid water (1 g/mL density)
    V_liquid = mass_water_g / 1000  # in liters

    # Calculate volume increase
    volume_increase = V_vapor - V_liquid
    return volume_increase, V_vapor, V_liquid

# Example usage
mass_water = 1.15  # grams
vapor_pressure = 31.82  # torr
temperature = 30  # degrees Celsius
volume_increase, total_vapor_volume, initial_liquid_volume = calculate_volume_increase(mass_water, vapor_pressure, temperature)
print(f'Volume increase necessary for all water to evaporate: {volume_increase:.3f} L')
print(f'Total vapor volume: {total_vapor_volume:.3f} L')
print(f'Initial liquid volume: {initial_liquid_volume:.3f} L')