# filename: calculate_delta_U_with_units.py
import numpy as np
from scipy.optimize import fsolve

# Constants
BAR_TO_PASCAL = 1e5  # Conversion factor from bar to Pascal

# Function to calculate final temperature and change in internal energy
def calculate_delta_U(n=2.50, C_V_m=12.47, T_i=325, P_i_bar=2.50, P_f_bar=1.25, P_ext_bar=1.00, R=8.314):
    """
    Calculate final temperature and change in internal energy (Delta U) for adiabatic expansion
    of an ideal gas against constant external pressure.

    Parameters:
    n : float - number of moles (mol)
    C_V_m : float - molar heat capacity at constant volume (J/mol/K)
    T_i : float - initial temperature (K)
    P_i_bar : float - initial pressure (bar)
    P_f_bar : float - final pressure (bar)
    P_ext_bar : float - external pressure (bar)
    R : float - ideal gas constant (J/mol/K)

    Returns:
    T_f : float - final temperature (K)
    delta_U : float - change in internal energy (J)
    """
    # Convert pressures from bar to Pascal for unit consistency
    P_i = P_i_bar * BAR_TO_PASCAL
    P_f = P_f_bar * BAR_TO_PASCAL
    P_ext = P_ext_bar * BAR_TO_PASCAL

    # Define the nonlinear equation derived from the first law and ideal gas law
    def equation(T_f):
        # Left side: change in internal energy per mole
        left = C_V_m * (T_f - T_i)
        # Right side: work done per mole against external pressure
        right = -P_ext * R * (T_f / P_f - T_i / P_i)
        return left - right

    # Initial guess for final temperature
    T_f_guess = T_i

    # Solve for final temperature
    T_f_solution, info, ier, mesg = fsolve(equation, T_f_guess, full_output=True)

    if ier != 1:
        print('Warning: Solver did not converge. Message:', mesg)

    T_f = T_f_solution[0]

    # Calculate change in internal energy
    delta_U = n * C_V_m * (T_f - T_i)

    return T_f, delta_U

# Run calculation with given values
final_temperature, delta_U = calculate_delta_U()

# Print results for immediate feedback
print(f'Final Temperature (K): {final_temperature:.2f}')
print(f'Change in Internal Energy Delta U (J): {delta_U:.2f}')

# Save results to a text file
with open('delta_U_results.txt', 'w') as f:
    f.write(f'Final Temperature (K): {final_temperature:.2f}\n')
    f.write(f'Change in Internal Energy Delta U (J): {delta_U:.2f}\n')
