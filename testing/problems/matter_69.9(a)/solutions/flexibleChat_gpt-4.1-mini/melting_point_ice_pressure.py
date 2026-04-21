# filename: melting_point_ice_pressure.py

def melting_point_ice_at_pressure(P_bar=50):
    """Calculate the melting point of ice at a given pressure using the Clausius-Clapeyron relation.

    Args:
        P_bar (float): Pressure in bar at which to calculate the melting point.

    Returns:
        float: Melting temperature in degrees Celsius.
    """
    # Constants
    T0 = 273.15  # Standard melting temperature in K
    delta_H = 6.01e3  # Enthalpy of fusion in J/mol
    M = 18.015  # Molar mass of water in g/mol
    rho_ice = 0.92  # Density of ice in g/cm^3
    rho_water = 1.00  # Density of water in g/cm^3

    # Convert densities to molar volumes in m^3/mol
    # 1 cm^3 = 1e-6 m^3
    V_ice = M / rho_ice * 1e-6  # m^3/mol
    V_water = M / rho_water * 1e-6  # m^3/mol

    delta_V = V_water - V_ice  # m^3/mol

    # Pressure difference in Pa (1 bar = 1e5 Pa)
    P0 = 1.01325  # Standard atmospheric pressure in bar
    delta_P = (P_bar - P0) * 1e5  # Pa

    # Clausius-Clapeyron equation (linear approx)
    delta_T = (T0 * delta_V / delta_H) * delta_P  # K

    T_new = T0 + delta_T  # K

    # Convert to Celsius
    T_new_C = T_new - 273.15

    return T_new_C

if __name__ == '__main__':
    melting_temp = melting_point_ice_at_pressure(50)
    print(f"Melting point of ice at 50 bar: {melting_temp:.4f} degrees Celsius")
