# filename: calculate_ethane_pressure_function.py

def calculate_pressure(n_moles, volume_liters, temperature_celsius):
    """
    Calculate the pressure exerted by an ideal gas.
    Parameters:
        n_moles (float): Number of moles of gas (must be positive).
        volume_liters (float): Volume in liters (must be positive).
        temperature_celsius (float): Temperature in degrees Celsius.
    Returns:
        float: Pressure in atmospheres.
    """
    if n_moles <= 0:
        raise ValueError("Number of moles must be positive.")
    if volume_liters <= 0:
        raise ValueError("Volume must be positive.")

    # Ideal gas constant in L·atm/(mol·K)
    R = 0.08206

    # Convert temperature to Kelvin
    temperature_kelvin = temperature_celsius + 273.15

    # Calculate pressure using ideal gas law: P = nRT / V
    pressure_atm = (n_moles * R * temperature_kelvin) / volume_liters

    return pressure_atm


# Given parameters
n = 10.0  # moles
V = 4.860  # volume in liters
T = 27.0  # temperature in Celsius

# Calculate pressure
pressure = calculate_pressure(n, V, T)

# Print the result
print(f"Pressure exerted by ethane gas: {pressure:.2f} atm")

# Return pressure for further use if needed
pressure
