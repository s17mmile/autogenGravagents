import numpy as np
from scipy.optimize import fsolve

# This function computes the time T at which the solution of the initial value problem first reaches 3.99.

def find_time_to_reach_value(target_value=3.99, initial_guess=0.1):
    # Define the function based on the rearranged equation
    def equation(T):
        return target_value * ((1 + T)**4 + 4 * np.exp(4 * T)) - 4 * np.exp(4 * T)

    # Solve for T
    T_solution = fsolve(equation, initial_guess)

    # Check if the solution converged
    if equation(T_solution[0]) == 0:
        return T_solution[0]
    else:
        raise ValueError('The solver did not converge to a solution.')

# Call the function and print the result
try:
    T = find_time_to_reach_value()
    print(f'The time T at which the solution first reaches 3.99 is approximately: {T:.4f}')
except ValueError as e:
    print(e)