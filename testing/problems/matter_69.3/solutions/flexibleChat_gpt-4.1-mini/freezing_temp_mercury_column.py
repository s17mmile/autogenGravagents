# filename: freezing_temp_mercury_column.py

def freezing_temperature_mercury_column():
    # Given constants
    delta_H_f = 2292  # J/mol (enthalpy of fusion)
    T_0 = 234.3       # K (normal freezing point)
    delta_V = 0.517e-6  # m^3/mol (change in molar volume on melting)
    density = 13600    # kg/m^3 (density of mercury)
    height = 10.0      # m (height of mercury column)
    g = 9.81           # m/s^2 (acceleration due to gravity)

    # Calculate pressure at bottom of column
    P = density * g * height  # Pa (N/m^2)

    # Calculate dT/dP using Clapeyron equation
    dT_dP = (T_0 * delta_V) / delta_H_f  # K/Pa

    # Calculate change in freezing temperature
    delta_T = dT_dP * P

    # Calculate new freezing temperature at bottom
    T = T_0 + delta_T

    return T

# Calculate and print the freezing temperature at the bottom of the mercury column
freezing_temp = freezing_temperature_mercury_column()
print(f"Freezing temperature at bottom of 10.0 m mercury column: {freezing_temp:.4f} K")
