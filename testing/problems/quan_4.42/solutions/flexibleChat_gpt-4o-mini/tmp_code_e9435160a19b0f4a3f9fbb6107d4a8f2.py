import numpy as np

# Constants
hbar = 1.0545718e-34  # Planck's constant over 2*pi in J.s
m = 1.0  # mass of the particle in kg
omega = 1.0  # angular frequency in rad/s
E0 = 0.5 * hbar * omega  # total energy for v=0 state

# Define the potential energy function for the harmonic oscillator
def potential(x):
    return 0.5 * m * (omega ** 2) * (x ** 2)

# Manual trapezoidal integration function

def trapezoidal_integration(y, x):
    integral = 0.0
    for i in range(1, len(y)):
        integral += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    return integral

# Numerov method to compute the wave function

def numerov_wavefunction(x):
    dx = x[1] - x[0]
    N = len(x)
    psi = np.zeros(N)
    psi[0] = 1e-10  # small non-zero value to avoid instability
    psi[1] = dx  # small value to start the recursion

    for i in range(1, N-1):
        k1 = (2 * m / hbar) * (E0 - potential(x[i]))
        k2 = (2 * m / hbar) * (E0 - potential(x[i+1]))
        k0 = (2 * m / hbar) * (E0 - potential(x[i-1]))
        psi[i+1] = (2 * (1 - (dx ** 2) * k1 / 12) * psi[i] - (1 + (dx ** 2) * k0 / 12) * psi[i-1]) / (1 + (dx ** 2) * k2 / 12)

    # Normalize the wave function
    norm = trapezoidal_integration(psi**2, x)
    if norm == 0:
        raise ValueError('Normalization factor is zero, cannot normalize wave function.')
    psi /= np.sqrt(norm)
    return psi

# Generate x values and compute the wave function
x = np.arange(-5, 5.1, 0.1)
psi = numerov_wavefunction(x)

# Identify the classically forbidden region
forbidden_mask = potential(x) > E0
forbidden_region = x[forbidden_mask]

# Calculate the probability in the forbidden region
probability = trapezoidal_integration(psi[forbidden_mask]**2, forbidden_region)

# Output the probability with formatting
print(f'Probability of being in the classically forbidden region for v=0 state: {probability:.6f}')