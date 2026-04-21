# filename: pressure_change_gas_thermometer.py

def pressure_change_constant_volume_gas_thermometer(P0_kPa, T0_K, delta_T_K):
    """
    Calculate the pressure change for a given temperature change in a constant-volume gas thermometer.

    Parameters:
    P0_kPa (float): Initial pressure at temperature T0 in kilopascals. Must be non-negative.
    T0_K (float): Initial temperature in kelvin. Must be positive.
    delta_T_K (float): Temperature change in kelvin. Can be positive or negative.

    Returns:
    float: Pressure change in kilopascals.

    Raises:
    ValueError: If T0_K is not positive or P0_kPa is negative.
    """
    if T0_K <= 0:
        raise ValueError('Initial temperature T0_K must be positive and non-zero.')
    if P0_kPa < 0:
        raise ValueError('Initial pressure P0_kPa must be non-negative.')

    delta_P_kPa = P0_kPa * (delta_T_K / T0_K)
    return delta_P_kPa

# Given values
P0 = 6.69  # kPa at triple point
T0 = 273.16  # K triple point temperature
delta_T = 1.00  # K temperature change

try:
    # Calculate pressure change
    pressure_change = pressure_change_constant_volume_gas_thermometer(P0, T0, delta_T)

    # Save result to a file
    with open('pressure_change_result.txt', 'w') as f:
        f.write(f'Pressure change for 1 K temperature change: {pressure_change:.5f} kPa\n')
except (ValueError, IOError) as e:
    # Handle errors gracefully
    with open('pressure_change_result.txt', 'w') as f:
        f.write(f'Error occurred: {str(e)}\n')
