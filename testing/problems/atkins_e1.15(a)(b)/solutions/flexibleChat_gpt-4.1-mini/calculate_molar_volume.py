# filename: calculate_molar_volume.py

def calculate_actual_molar_volume(T=250, P=15, reduction_percent=12):
    """Calculate the actual molar volume of a gas given temperature, pressure, and percentage reduction from ideal."""
    R = 0.0821  # L·atm/(mol·K)
    V_ideal = (R * T) / P
    V_actual = V_ideal * (1 - reduction_percent / 100)
    return V_actual

# Calculate and print the molar volume
actual_volume = calculate_actual_molar_volume()
print(f"Actual molar volume of the gas: {actual_volume:.4f} L/mol")
