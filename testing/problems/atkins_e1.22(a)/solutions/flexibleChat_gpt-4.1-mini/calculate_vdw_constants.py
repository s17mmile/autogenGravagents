# filename: calculate_vdw_constants.py

def calculate_vdw_constants(P, V, T, a, R=8.314):
    """
    Calculate the van der Waals constant b and compression factor Z for a gas.
    Parameters:
        P (float): Pressure in Pascals (Pa), must be positive.
        V (float): Molar volume in cubic meters per mole (m^3/mol), must be positive.
        T (float): Temperature in Kelvin (K), must be positive.
        a (float): van der Waals constant a in m^6 Pa mol^-2.
        R (float): Universal gas constant, default 8.314 J mol^-1 K^-1.
    Returns:
        b (float): van der Waals constant b in m^3/mol.
        Z (float): Compression factor (dimensionless).
    """
    # Input validation
    if P <= 0 or V <= 0 or T <= 0:
        raise ValueError("Pressure, volume, and temperature must be positive values.")

    try:
        # Calculate b using the rearranged van der Waals equation
        b = V - (R * T) / (P + a / V**2)

        # Calculate compression factor Z
        Z = (P * V) / (R * T)

    except ZeroDivisionError as e:
        raise ZeroDivisionError("Division by zero encountered in calculation.") from e

    return b, Z


# Given constants
P = 3.0e6  # Pressure in Pa
V = 5.00e-4  # Molar volume in m^3/mol
T = 273  # Temperature in K

a = 0.50  # van der Waals constant a in m^6 Pa mol^-2

# Perform calculation
b, Z = calculate_vdw_constants(P, V, T, a)

# Print results
print(f"van der Waals constant b = {b:.6e} m^3/mol")
print(f"Compression factor Z = {Z:.6f}")
