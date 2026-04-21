# filename: calculate_cavity_radius.py
import math

# Constants
h = 6.62607015e-34  # Planck constant, J*s
hbar = h / (2 * math.pi)  # Reduced Planck constant, J*s
c = 299792458  # Speed of light, m/s
m_e = 9.10938356e-31  # Electron mass, kg

# Given wavelength of absorption
wavelength_nm = 1500  # nm
wavelength_m = wavelength_nm * 1e-9  # Convert nm to meters

# Zeros of spherical Bessel function for ground and first excited states
k1 = math.pi  # first zero
k2 = 4.493  # second zero

# Energy difference corresponding to photon absorption
delta_E = h * c / wavelength_m  # Energy in Joules

# Calculate radius R
numerator = k2**2 - k1**2
R = hbar * math.sqrt(numerator / (2 * m_e * delta_E))  # radius in meters

# Convert radius to angstroms for convenience
R_angstrom = R * 1e10

print(f"Radius of the cavity: {R:.3e} meters (m)")
print(f"Radius of the cavity: {R_angstrom:.3f} angstroms (A)")

# Return radius in meters and angstroms
R, R_angstrom
