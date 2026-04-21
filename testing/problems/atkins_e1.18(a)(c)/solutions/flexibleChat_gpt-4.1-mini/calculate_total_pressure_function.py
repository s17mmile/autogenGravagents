# filename: calculate_total_pressure_function.py

def calculate_total_pressure(n_H2, n_N2, V, T):
    """
    Calculate the total pressure of a gas mixture using the ideal gas law.

    Parameters:
    n_H2 (float): moles of hydrogen gas
    n_N2 (float): moles of nitrogen gas
    V (float): volume of the vessel in liters (dm^3)
    T (float): temperature in Kelvin

    Returns:
    float: total pressure in atmospheres
    """
    R = 0.0821  # Ideal gas constant in L*atm/(mol*K)
    n_total = n_H2 + n_N2
    P_total = (n_total * R * T) / V
    return P_total

# Given data
V = 22.4  # volume in dm^3 (liters)
n_H2 = 2.0  # moles of H2
n_N2 = 1.0  # moles of N2
T = 273.15  # temperature in Kelvin

# Calculate total pressure
pressure = calculate_total_pressure(n_H2, n_N2, V, T)

print(f"Total pressure of the gas mixture: {pressure:.2f} atm")

pressure  # Return the value for further use if needed
