# filename: calculate_F2_minus_entropy_corrected.py
import math

# Constants
k_B = 1.380649e-23  # Boltzmann constant, J/K
h = 6.62607015e-34  # Planck constant, J s
c = 2.99792458e8    # Speed of light, m/s
R = 8.314462618     # Gas constant, J/(mol K)
T = 298.0           # Temperature, K
p = 1.0e5           # Pressure, Pa (1 bar)

# Molecular parameters for F2-
# Atomic mass of F (fluorine) in kg
m_F = 18.998403163 * 1.66053906660e-27  # kg

# Reduced mass (mu) for F2- (two identical atoms)
mu = m_F / 2

# Equilibrium internuclear distance (r_e) in meters
r_e = 190.0e-12  # 190 pm to meters

# Vibrational wavenumber (nu) in cm^-1
nu = 450.0

# Electronic states energies in eV
E_elec = [0.0, 1.609, 1.702]  # ground and two excited states

# Degeneracies of electronic states
# Ground state: doublet (2), excited states assumed doublet as well
g_elec = [2, 2, 2]

# Calculate translational entropy (ideal gas, 1 mole)
# Using Sackur-Tetrode equation for entropy of ideal gas
m_mol = 2 * m_F  # molecular mass in kg
lambda_th = h / math.sqrt(2 * math.pi * m_mol * k_B * T)  # thermal wavelength
V_m = R * T / p  # molar volume at 1 bar and T
S_trans = R * (2.5 + math.log(V_m / (lambda_th**3)))

# Calculate rotational entropy
# Moment of inertia I = mu * r_e^2
I = mu * r_e**2
# Rotational constant B in energy units (J)
B_J = h**2 / (8 * math.pi**2 * I)
# Rotational temperature theta_rot = B_J / k_B
theta_rot = B_J / k_B
# Symmetry number sigma = 2 for homonuclear diatomic
sigma = 2
# Rotational partition function q_rot = T / (sigma * theta_rot)
q_rot = T / (sigma * theta_rot)
S_rot = R * (math.log(q_rot) + 1)

# Calculate vibrational entropy
# Convert vibrational wavenumber to frequency (Hz)
nu_hz = nu * 100 * c
# Vibrational temperature theta_vib = h * nu_hz / k_B
theta_vib = h * nu_hz / k_B
x = theta_vib / T
S_vib = R * (x / (math.exp(x) - 1) - math.log(1 - math.exp(-x)))

# Calculate electronic entropy
# Convert eV to J
eV_to_J = 1.602176634e-19
E_elec_J = [e * eV_to_J for e in E_elec]
# Electronic partition function
q_elec = sum(g * math.exp(-E / (k_B * T)) for g, E in zip(g_elec, E_elec_J))
S_elec = R * (math.log(q_elec) + (sum(g * (E / (k_B * T)) * math.exp(-E / (k_B * T)) for g, E in zip(g_elec, E_elec_J)) / q_elec))

# Total standard molar entropy
S_total = S_trans + S_rot + S_vib + S_elec

print(f"Standard molar entropy of F2- at {T} K: {S_total:.2f} J/(mol K)")
