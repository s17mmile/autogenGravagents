import numpy as np
import matplotlib.pyplot as plt

# Constants (parameterized)
V0 = 2.0 * 1.602e-19  # Barrier height in Joules
E = 1.5 * 1.602e-19   # Electron energy in Joules
m = 9.11e-31          # Mass of electron in kg
hbar = 1.055e-34      # Reduced Planck's constant in Js

# Barrier parameters
width = 100e-12       # Barrier width in meters
N = 1000              # Number of points in the grid
L = 1e-9             # Length of the domain in meters
x = np.linspace(0, L, N)  # Spatial domain

# Potential array
V = np.zeros(N)
V[int(N/2)-int(width/(L/N)/2):int(N/2)+int(width/(L/N)/2)] = V0

# Hamiltonian matrix using finite difference method
h = L / N
H = np.zeros((N, N))
for i in range(1, N-1):
    H[i, i-1] = -hbar**2 / (2 * m * h**2)
    H[i, i] = 2 * hbar**2 / (2 * m * h**2) + V[i]

# Boundary conditions
H[0, 0] = H[N-1, N-1] = 1e10  # Large value to simulate infinite potential walls

# Solve the eigenvalue problem
energies, wavefuncs = np.linalg.eigh(H)

# Find the wave function corresponding to the energy level E
idx = np.argmin(np.abs(energies - E))
if idx >= len(energies):
    raise ValueError('Energy level does not correspond to any eigenvalue.')

psi = wavefuncs[:, idx]

# Normalize the wave function
psi /= np.sqrt(np.trapz(psi**2, x))

# Calculate tunneling probability (approximation)
T = np.abs(psi[int(N/2)])**2

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x * 1e9, V / 1.602e-19, label='Potential Barrier (V)', color='red')
plt.plot(x * 1e9, psi * np.max(V) * 0.5, label='Wave Function (ψ)', color='blue')
plt.title('Potential Barrier and Wave Function')
plt.xlabel('Position (nm)')
plt.ylabel('Energy (eV)')
plt.axhline(y=E / 1.602e-19, color='green', linestyle='--', label='Electron Energy (E)')
plt.axhline(y=V0 / 1.602e-19, color='orange', linestyle='--', label='Barrier Height (V0)')
plt.legend()
plt.grid()
plt.savefig('tunneling_probability_plot.png')

# Output the tunneling probability
print(f'Tunneling Probability (T): {T:.6e}')  # Output in scientific notation