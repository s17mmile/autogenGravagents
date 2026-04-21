# filename: find_decay_time.py
import numpy as np
from scipy.optimize import fsolve

# Function to compute the smallest T such that |u(t)| <= threshold for all t > T
def find_decay_time(threshold=0.1):
    # Given constants from characteristic equation
    alpha = -0.2  # Real part of roots
    beta = np.sqrt(34)/5  # Imaginary part of roots

    # Initial conditions
    u0 = 2
    u_prime0 = 1

    # Compute constants C1 and C2 from initial conditions
    C1 = u0
    C2 = (u_prime0 - alpha * C1) / beta

    # Amplitude of the envelope of the damped oscillation
    amplitude = np.sqrt(C1**2 + C2**2)

    # Envelope function: amplitude * exp(alpha * t)
    # We want to find T such that envelope(T) = threshold
    def envelope_eq(T):
        return amplitude * np.exp(alpha * T) - threshold

    # Initial guess for T
    T_initial_guess = 0

    # Solve for T numerically
    T_solution = fsolve(envelope_eq, T_initial_guess)[0]

    # Ensure T_solution is non-negative
    T_solution = max(T_solution, 0)

    return T_solution

# Compute and print the smallest T
T = find_decay_time(0.1)
print(f"The smallest T such that |u(t)| <= 0.1 for all t > T is approximately {T:.4f} seconds.")

# This code can be executed directly to obtain the result.