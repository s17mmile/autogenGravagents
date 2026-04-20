# filename: calculate_ethane_pressure.py

def calculate_pressure(n_mol=10.0, V_dm3=4.860, T_celsius=27.0):
    """
    Calculate the pressure exerted by an ideal gas using the ideal gas law.

    Parameters:
    n_mol (float): Number of moles of gas (must be positive).
    V_dm3 (float): Volume in dm^3 (liters) (must be positive).
    T_celsius (float): Temperature in degrees Celsius.

    Returns:
    float: Pressure in atm.
    """
    if n_mol <= 0:
        raise ValueError("Number of moles must be positive.")
    if V_dm3 <= 0:
        raise ValueError("Volume must be positive.")

    R = 0.0821  # L atm / mol K
    T_kelvin = T_celsius + 273.15
    if T_kelvin <= 0:
        raise ValueError("Temperature in Kelvin must be positive.")

    P_atm = (n_mol * R * T_kelvin) / V_dm3
    return P_atm

pressure = calculate_pressure()
print(f"Pressure exerted by ethane gas: {pressure:.2f} atm")
