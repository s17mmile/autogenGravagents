import numpy as np
from scipy.optimize import fsolve

# Constants
N = 6.022e23  # number of particles (1 mole)
Delta_E_cm = 1000  # energy difference in cm^-1
Delta_E = Delta_E_cm * 1.986e-23  # convert to Joules
U_target = 3000  # target internal energy in Joules
k_B = 1.38e-23  # Boltzmann constant in J/K

# Function to calculate the difference between target internal energy and calculated internal energy

def internal_energy_equation(T):
    """
    Calculate the difference between the target internal energy and the calculated internal energy.
    
    Parameters:
    T : float
        Temperature in Kelvin.
    
    Returns:
    float
        Difference between calculated internal energy and target internal energy.
    """
    beta = 1 / (k_B * T)
    P1 = np.exp(-beta * Delta_E) / (1 + np.exp(-beta * Delta_E))
    U = N * Delta_E * P1  # internal energy
    return U - U_target  # we want this to be zero

# Initial guess for temperature
T_initial_guess = 300  # in Kelvin

# Solve for temperature
try:
    T_solution = fsolve(internal_energy_equation, T_initial_guess)
    print(f'Temperature at which internal energy equals {U_target} J: {T_solution[0]:.2f} K')
except Exception as e:
    print(f'Error in solving for temperature: {e}')