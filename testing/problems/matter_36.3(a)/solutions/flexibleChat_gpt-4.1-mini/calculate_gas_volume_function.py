# filename: calculate_gas_volume_function.py

def calculate_gas_volume(P, T, Z, n_mmol):
    """Calculate the volume of a gas using the real gas equation with compression factor Z.

    Parameters:
    P (float): Pressure in atm (must be positive)
    T (float): Temperature in K (must be positive)
    Z (float): Compression factor (must be positive)
    n_mmol (float): Amount of gas in mmol (must be positive)

    Returns:
    float: Volume in liters
    """
    # Input validation
    if P <= 0 or T <= 0 or Z <= 0 or n_mmol <= 0:
        raise ValueError("All input parameters must be positive numbers.")

    R = 0.0821  # ideal gas constant in L*atm/(mol*K)
    n = n_mmol * 1e-3  # convert mmol to mol

    V = (Z * n * R * T) / P
    return V


# Test the function with given values
pressure = 20  # atm
temperature = 300  # K
compression_factor = 0.86
amount_mmol = 8.2

volume = calculate_gas_volume(pressure, temperature, compression_factor, amount_mmol)
print(f"Volume occupied by {amount_mmol} mmol of gas at {temperature} K and {pressure} atm with Z={compression_factor}: {volume:.6f} L")
