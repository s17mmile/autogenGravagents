import numpy as np
from scipy.optimize import fsolve

# Constants
k = 8.99e9  # Coulomb's constant in N m^2/C^2
q = 0.01  # Charge value in C (smaller value for testing)

# Electric field values
E1 = np.array([7.2 * 4.0, 7.2 * 3.0])  # Electric field at point (3.0, 3.0) in N/C
E2 = np.array([100.0, 0.0])  # Electric field at point (2.0, 0.0) in N/C

# Points in meters
point1 = np.array([0.03, 0.03])  # (3.0 cm, 3.0 cm)
point2 = np.array([0.02, 0.0])  # (2.0 cm, 0.0 cm)

# Function to calculate the electric field components

def equations(x):
    # Position of the charge
    charge_position = np.array([x[0], 0])  # Use x[0] to get the scalar value
    r1 = np.linalg.norm(point1 - charge_position)
    r2 = np.linalg.norm(point2 - charge_position)
    
    # Electric field components from the charge
    E1_calc = k * q * (point1 - charge_position) / r1**3
    E2_calc = k * q * (point2 - charge_position) / r2**3
    
    # Debugging outputs
    print(f'Charge Position: {charge_position}')
    print(f'Distance r1: {r1}, Distance r2: {r2}')
    print(f'E1_calc: {E1_calc}, E2_calc: {E2_calc}')
    
    # Set up equations based on the electric field components
    eq1 = E1_calc[0] - E1[0]  # x-component
    return [eq1]  # Only return the x-component equation

# Initial guess for the x-coordinate of the charge (midpoint)
initial_guess = [0.025]  # 2.5 cm as a list

# Solve for the x-coordinate
x_solution = fsolve(equations, initial_guess)

# Check for convergence
if not np.isclose(equations(x_solution), [0]).all():
    print('Warning: The solution may not have converged.')

# Output the result
print(f'The x-coordinate of the charge is: {x_solution[0]} m')