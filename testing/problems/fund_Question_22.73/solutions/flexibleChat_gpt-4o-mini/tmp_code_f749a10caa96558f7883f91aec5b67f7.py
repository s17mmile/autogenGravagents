import numpy as np
from scipy.optimize import fsolve

# Constants
k = 8.99e9  # Coulomb's constant in N m^2/C^2
q = 1.0  # Charge value in C (can be parameterized)

# Electric field values
E1 = np.array([7.2 * 4.0, 7.2 * 3.0])  # Electric field at point (3.0, 3.0) in N/C
E2 = np.array([100.0, 0.0])  # Electric field at point (2.0, 0.0) in N/C

# Points in meters
point1 = np.array([0.03, 0.03])  # (3.0 cm, 3.0 cm)
point2 = np.array([0.02, 0.0])  # (2.0 cm, 0.0 cm)

# Function to calculate the electric field components

def equations(x):
    # Position of the charge
    charge_position = np.array([x, 0])
    r1 = np.linalg.norm(point1 - charge_position)
    r2 = np.linalg.norm(point2 - charge_position)
    
    # Electric field components from the charge
    E1_calc = k * q * (point1 - charge_position) / r1**3
    E2_calc = k * q * (point2 - charge_position) / r2**3
    
    # Set up equations based on the electric field components
    eq1 = E1_calc[0] - E1[0]  # x-component
    eq2 = E1_calc[1] - E1[1]  # y-component
    
    return [eq1, eq2]  # Only return the equations needed for x

# Initial guess for the x-coordinate of the charge
initial_guess = 0.01  # 1 cm

# Solve for the x-coordinate
x_solution = fsolve(equations, initial_guess)

# Output the result
print(f'The x-coordinate of the charge is: {x_solution[0]} m')