# filename: check_relative_difference_bracket.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k

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

# Evaluate at endpoints
T_low = 1
T_high = 2000
val_low = func(T_low)
val_high = func(T_high)

print(f"func({T_low}) = {val_low}")
print(f"func({T_high}) = {val_high}")

# Evaluate over a range for plotting
T_values = np.linspace(1, 2000, 500)
func_values = [func(T) for T in T_values]

plt.plot(T_values, func_values, label='relative_difference(T) - 0.05')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Temperature (K)')
plt.ylabel('Function value')
plt.title('Check root bracket for relative difference function')
plt.legend()
plt.grid(True)
plt.savefig('relative_difference_bracket_check.png')
