import math

# Constants
pressure_input_atm = 1.05  # atm
pressure_output_atm = 1.00  # atm
V = 200e-6    # Volume in m^3 (200 mL)
L = 0.1      # Length of the tube in meters (10 cm)
D = 0.25e-3  # Diameter of the tube in meters (0.25 mm)

# Dynamic viscosity of hydrogen at 273 K in Pa.s
eta = 8.76e-6

# 1. Calculate the pressure difference in Pascals
Delta_P = (pressure_input_atm - pressure_output_atm) * 101325  # Convert atm to Pa

# 2. Calculate the radius of the tube
r = D / 2

# 3. Use the Hagen-Poiseuille equation to calculate the flow rate (Q)
Q = (math.pi * r**4 * Delta_P) / (8 * eta * L)

# 4. Calculate the time required to pass the volume of gas
if Q > 0:
    time_required = V / Q
    # Output the time required in seconds
    print(f'Time required to pass 200 mL of H2 through the capillary tube: {time_required:.6e} seconds')
else:
    print('Error: Flow rate is zero, cannot calculate time required.')