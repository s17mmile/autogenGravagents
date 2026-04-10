def calculate_melting_point(P_bar):
    # Constants
    L = 334000  # Latent heat of fusion in J/kg
    T_initial = 273.15  # Initial melting point of ice in K
    rho_ice = 920  # Density of ice in kg/m^3
    rho_water = 1000  # Density of water in kg/m^3

    # Error handling for densities
    if rho_ice <= 0 or rho_water <= 0:
        raise ValueError('Densities must be positive values.')

    P_change = P_bar * 100000  # Pressure change in Pa

    # Calculate volumes
    V_ice = 1 / rho_ice  # Volume of 1 kg of ice in m^3
    V_water = 1 / rho_water  # Volume of 1 kg of water in m^3

    # Change in volume
    change_in_volume = V_water - V_ice  # Change in volume during melting

    # Calculate dT using the Clausius-Clapeyron equation
    change_in_temperature = (L / (T_initial * change_in_volume)) * P_change

    # New melting point
    T_new = T_initial - change_in_temperature

    return T_new

# Example usage
melting_point = calculate_melting_point(50)
print(f'New melting point of ice under 50 bar: {melting_point:.2f} K')
print(f'New melting point of ice under 50 bar: {melting_point - 273.15:.2f} °C')