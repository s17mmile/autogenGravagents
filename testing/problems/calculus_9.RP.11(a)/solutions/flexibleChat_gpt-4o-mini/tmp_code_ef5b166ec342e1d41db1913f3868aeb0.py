import sympy as sp

# Given parameters
initial_height = 60  # ft
final_height = 100    # ft
rate_of_conveyor = 60000 * sp.pi  # ft^3/h

# Check for valid heights
if initial_height < 0 or final_height <= initial_height:
    raise ValueError('Invalid height values: initial_height must be non-negative and final_height must be greater than initial_height.')

# Define the variable for height
h = sp.symbols('h')

# Volume of the cone as a function of height
volume = 0.75 * sp.pi * h**3

# Rate of change of volume with respect to height
dV_dh = sp.diff(volume, h)

# Rate of change of height with respect to time
dh_dt = rate_of_conveyor / dV_dh

# Set up the integral to find time
# Integrate dh from initial_height to final_height
time_expr = sp.integrate(1/dh_dt, (h, initial_height, final_height))

# Calculate the time
try:
    time_to_fill = time_expr.evalf()
    # Print the result
    print(f'Time to fill the pile to the top of the silo: {time_to_fill:.2f} hours')
except ZeroDivisionError:
    print('Error: Division by zero encountered in calculations.')