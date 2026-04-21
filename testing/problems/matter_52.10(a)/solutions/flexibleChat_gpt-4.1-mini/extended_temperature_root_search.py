# filename: extended_temperature_root_search.py
import numpy as np
from scipy.constants import h, c, k
from scipy.optimize import root_scalar

wavenumber_cm = 323.2
frequency = wavenumber_cm * c * 100

def q_vib_exact(T):
    beta = h * frequency / (k * T)
    return 1 / (1 - np.exp(-beta))

def q_vib_approx(T):
    return k * T / (h * frequency)

def relative_difference(T):
    q_exact = q_vib_exact(T)
    q_approx = q_vib_approx(T)
    return abs(q_exact - q_approx) / q_exact

def func(T):
    return relative_difference(T) - 0.05

# Check function values at 2000 K and 10000 K
T_low = 2000
T_high = 10000
val_low = func(T_low)
val_high = func(T_high)

print(f"func({T_low}) = {val_low}")
print(f"func({T_high}) = {val_high}")

if val_low * val_high < 0:
    # Root exists in this interval
    solution = root_scalar(func, bracket=[T_low, T_high], method='bisect')
    if solution.converged:
        temperature_5_percent = solution.root
        print(f"Temperature at which the vibrational partition function's exact and approximate values differ by 5%: {temperature_5_percent:.2f} K")
    else:
        print("Root finding did not converge.")
else:
    print("No root found in the extended temperature range.")
