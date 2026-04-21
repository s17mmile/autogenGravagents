# filename: calculate_beta_hcl.py
import math

# Constants
c_cm_s = 2.9979e10  # speed of light in cm/s
N_A = 6.022e23      # Avogadro's number in mol^-1
nu_tilde_obs = 2886  # observed vibrational wavenumber in cm^-1
D_kJ_per_mol = 440.2  # dissociation energy in kJ/mol

# Atomic masses in atomic mass units (u)
m_H = 1.00784
m_Cl35 = 34.96885

# Convert atomic mass units to kg
amu_to_kg = 1.66054e-27


def calculate_beta(nu_tilde_obs, D_kJ_per_mol, m_H, m_Cl35):
    """Calculate beta for H35Cl molecule."""
    # Calculate reduced mass in kg
    mu_u = (m_H * m_Cl35) / (m_H + m_Cl35)
    mu = mu_u * amu_to_kg

    # Convert dissociation energy to Joules per molecule
    D_J_per_mol = D_kJ_per_mol * 1e3
    D_J = D_J_per_mol / N_A

    # Calculate beta
    beta = 2 * math.pi * c_cm_s * nu_tilde_obs * math.sqrt(mu / (2 * D_J))
    return beta


beta_value = calculate_beta(nu_tilde_obs, D_kJ_per_mol, m_H, m_Cl35)
print(f"Calculated beta: {beta_value:.3e} s^-1")

beta_value
