# filename: constant_volume_gas_thermometer.py

def calculate_pressure_at_temperature(P1_kPa, T1_K, T2_C):
    """
    Calculate the pressure indicated by a constant-volume perfect gas thermometer at temperature T2_C.
    Parameters:
    - P1_kPa: Pressure at reference temperature T1_K (in kPa)
    - T1_K: Reference temperature in Kelvin
    - T2_C: Temperature at which to find pressure (in Celsius)

    Returns:
    - Pressure at T2_C in kPa
    """
    # Convert Celsius to Kelvin (standard conversion)
    T2_K = T2_C + 273.15

    # Calculate pressure at T2 using proportionality P1/T1 = P2/T2
    P2_kPa = P1_kPa * (T2_K / T1_K)
    return P2_kPa


# Given data
P1_kPa = 6.69  # pressure at triple point in kPa
T1_K = 273.16  # triple point temperature in Kelvin
T2_C = 100.00  # temperature in Celsius

# Calculate pressure at 100 degrees Celsius
pressure_at_100C = calculate_pressure_at_temperature(P1_kPa, T1_K, T2_C)

# Output the result
print(f"Pressure at {T2_C} degrees Celsius: {pressure_at_100C:.3f} kPa")
