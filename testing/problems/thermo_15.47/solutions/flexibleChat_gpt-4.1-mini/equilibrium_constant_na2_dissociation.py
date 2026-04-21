# filename: equilibrium_constant_na2_dissociation.py
import numpy as np
from scipy.constants import R, h, c, k, N_A, physical_constants

def equilibrium_constant_na2_dissociation(T=298.0, P=101325):
    # Given data
    B_cm = 0.155  # rotational constant in cm^-1
    nu_tilde = 159.0  # vibrational frequency in cm^-1
    D0_kJ_per_mol = 70.4  # dissociation energy in kJ/mol
    g_e_Na = 2  # electronic degeneracy of Na atom

    # Convert dissociation energy from kJ/mol to J per molecule
    D0 = D0_kJ_per_mol * 1000 / N_A  # J per molecule

    # Masses (kg)
    mass_Na = 22.98976928 * physical_constants['atomic mass constant'][0]  # Na atomic mass
    mass_Na2 = 2 * mass_Na

    # Rotational partition function for Na2 (linear rotor)
    # q_rot = kT / (sigma * h * c * B) where sigma=2 for homonuclear diatomic
    sigma = 2
    q_rot = k * T / (sigma * h * c * B_cm)

    # Vibrational partition function
    theta_vib = h * c * nu_tilde / k
    q_vib = 1 / (1 - np.exp(-theta_vib / T))

    # Electronic partition functions
    q_e_Na = g_e_Na
    q_e_Na2 = 1  # Assume ground state degeneracy 1 for Na2 molecule

    # Translational partition functions per molecule (ideal gas)
    q_trans_Na = ((2 * np.pi * mass_Na * k * T) / (h ** 2)) ** (3 / 2) * k * T / P
    q_trans_Na2 = ((2 * np.pi * mass_Na2 * k * T) / (h ** 2)) ** (3 / 2) * k * T / P

    # Total partition functions per molecule
    q_Na = q_trans_Na * q_e_Na
    q_Na2 = q_trans_Na2 * q_rot * q_vib * q_e_Na2

    # Zero-point energy correction (ZPE = 0.5 * h * nu)
    nu = nu_tilde * c * 100  # convert cm^-1 to Hz
    ZPE = 0.5 * h * nu
    D0_corrected = D0 - ZPE  # corrected dissociation energy

    # Calculate equilibrium constant Kp
    # Kp = (q_Na^2 / q_Na2) * exp(-D0_corrected / (k * T))
    Kp = (q_Na ** 2 / q_Na2) * np.exp(-D0_corrected / (k * T))

    return Kp

if __name__ == '__main__':
    Kp_value = equilibrium_constant_na2_dissociation()
    print(f'Equilibrium constant Kp for Na2 dissociation at 298 K: {Kp_value:.3e}')
