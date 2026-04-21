# filename: calculate_anharmonicity_constant.py
from scipy.constants import h, c, N_A

# Given constants
omega_e_tilde = 2886  # in cm^-1
D_kJ_per_mol = 440.2  # in kJ/mol

# Convert D from kJ/mol to J per molecule
D_J_per_mol = D_kJ_per_mol * 1e3  # kJ to J
D_J_per_molecule = D_J_per_mol / N_A

# Speed of light in cm/s
c_cm_per_s = c * 100

# Calculate tilde{x}_e
x_e_tilde = (h * c_cm_per_s * omega_e_tilde) / (4 * D_J_per_molecule)

print(f"Anharmonicity constant tilde{{x}}_e = {x_e_tilde:.6f} cm^-1")
