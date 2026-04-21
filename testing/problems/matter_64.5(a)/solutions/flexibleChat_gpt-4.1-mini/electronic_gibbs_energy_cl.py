# filename: electronic_gibbs_energy_cl.py
from math import exp, log
from scipy.constants import k, h, c, N_A

def calculate_electronic_gibbs_energy(T=500):
    # Given data
    energy_diff_cm = 881  # energy difference in cm^-1
    
    # Degeneracies of the two levels
    ground_state_deg = 4  # 2P3/2 has 2J+1 = 4
    excited_state_deg = 2  # 2P1/2 has 2J+1 = 2

    # Convert energy difference from cm^-1 to Joules
    # 1 cm^-1 = 100 m^-1, so multiply by 100 to convert to m^-1
    energy_diff_J = h * c * energy_diff_cm * 100

    # Calculate partition function Z
    Z = ground_state_deg * exp(0) + excited_state_deg * exp(-energy_diff_J / (k * T))

    # Calculate electronic Helmholtz free energy per molecule
    A_per_molecule = -k * T * log(Z)

    # Convert to per mole
    A_per_mole = A_per_molecule * N_A

    # For ideal gases, electronic contribution to Gibbs energy approximated by Helmholtz free energy
    return A_per_mole

if __name__ == '__main__':
    result = calculate_electronic_gibbs_energy(500)
    print(f'Electronic contribution to molar Gibbs energy of Cl at 500 K: {result:.2f} J/mol')
    with open('electronic_gibbs_energy_cl_500K.txt', 'w') as f:
        f.write(f'Electronic contribution to molar Gibbs energy of Cl at 500 K: {result:.2f} J/mol\n')
