def calculate_pressure_change(P0, T0, Delta_T):
    """
    Calculate the change in pressure for a constant-volume perfect gas thermometer
    given the initial pressure, initial temperature, and change in temperature.
    """
    # Calculate the constant of proportionality k
    k = P0 / T0
    # Calculate the change in pressure for the given change in temperature
    Delta_P = k * Delta_T
    return Delta_P

# Constants
P0 = 6.69  # Initial pressure in kPa at the triple point of water
T0 = 273.16  # Temperature at the triple point of water in K
Delta_T = 1.00  # Change in temperature in K

# Validate inputs
if P0 <= 0 or T0 <= 0:
    raise ValueError('Pressure and temperature must be positive values.')

# Calculate the change in pressure
Delta_P = calculate_pressure_change(P0, T0, Delta_T)

# Output the result
print(f'The change in pressure for a change of 1.00 K is approximately {Delta_P:.4f} kPa.')