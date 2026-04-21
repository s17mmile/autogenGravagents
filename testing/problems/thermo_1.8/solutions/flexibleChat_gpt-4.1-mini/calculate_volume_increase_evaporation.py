# filename: calculate_volume_increase_evaporation.py

def calculate_volume_increase(mass_g=1.15, temp_c=30, vapor_pressure_torr=31.82):
    """
    Calculate the volume increase necessary for all liquid water to evaporate at given temperature and vapor pressure.
    Assumes ideal gas behavior.

    Parameters:
    mass_g (float): mass of liquid water in grams
    temp_c (float): temperature in degrees Celsius
    vapor_pressure_torr (float): vapor pressure of water in torr

    Returns:
    float: volume increase in liters
    """
    # Constants
    molar_mass_h2o = 18.015  # g/mol
    R = 0.08206  # L atm / (mol K)
    atm_per_torr = 1 / 760
    density_h2o = 1.0  # g/mL (density of liquid water)

    # Input validation
    if mass_g <= 0 or temp_c < -273.15 or vapor_pressure_torr <= 0:
        raise ValueError("Mass must be positive, temperature above absolute zero, and vapor pressure positive.")

    # Convert mass to moles
    n = mass_g / molar_mass_h2o

    # Convert temperature to Kelvin
    T = temp_c + 273.15

    # Convert vapor pressure to atm
    P = vapor_pressure_torr * atm_per_torr

    # Calculate volume of vapor using ideal gas law (in liters)
    V_vapor = (n * R * T) / P

    # Calculate initial volume of liquid water in liters
    # density is in g/mL, so divide by 1000 to convert mL to L
    V_liquid = mass_g / (density_h2o * 1000)

    # Calculate volume increase
    delta_V = V_vapor - V_liquid

    return delta_V

# Calculate and print the volume increase
volume_increase = calculate_volume_increase()
print(f"Volume increase necessary for complete evaporation: {volume_increase:.4f} L")
