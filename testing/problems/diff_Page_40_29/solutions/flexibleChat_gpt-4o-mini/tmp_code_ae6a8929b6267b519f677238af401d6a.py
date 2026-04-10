import numpy as np
from scipy.optimize import fsolve

def equation(t):
    return 32 * (1/4 * np.cos(2 * t) + 2 * np.sin(2 * t)) - 788 * np.exp(-1/4 * t)

# Range of initial guesses for t
initial_guesses = np.linspace(0, 10, 100)  # 100 guesses from 0 to 10

# Store results
results = []

for guess in initial_guesses:
    solution, info, ier, msg = fsolve(equation, guess, full_output=True)
    if ier == 1:
        results.append(solution[0])
        print(f'Converged at initial guess {guess:.2f}: t = {solution[0]:.4f}')  

if not results:
    print('No convergence found for any initial guess in the range.')