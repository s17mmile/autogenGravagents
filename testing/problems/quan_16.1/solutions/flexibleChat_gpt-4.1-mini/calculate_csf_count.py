# filename: calculate_csf_count.py
import math

def binomial(n, k):
    if k > n or k < 0:
        return 0
    return math.comb(n, k)

def calculate_csf(n_electrons, n_orbitals):
    """
    Calculate an upper bound on the number of Configuration State Functions (CSFs) for a full CI calculation.
    Assumes singlet state (S=0) and uses the number of Slater determinants as an upper bound.

    Parameters:
    n_electrons (int): Total number of electrons in the molecule.
    n_orbitals (int): Total number of spatial orbitals (approximate number of basis functions).

    Returns:
    int: Estimated upper bound on the number of CSFs.
    """
    n_alpha = n_electrons // 2
    n_beta = n_electrons - n_alpha

    alpha_configs = binomial(n_orbitals, n_alpha)
    beta_configs = binomial(n_orbitals, n_beta)

    n_determinants = alpha_configs * beta_configs

    return n_determinants

# Electron count for CH2SiHF:
# C: 6 electrons
# H: 3 atoms * 1 electron each = 3 electrons
# Si: 14 electrons
# F: 9 electrons
# Total electrons = 6 + 3 + 14 + 9 = 32
n_electrons = 32

# Approximate number of basis functions for 6-31G** basis set:
# C: 15 basis functions
# H: 5 basis functions each
# Si: 28 basis functions
# F: 15 basis functions
# Total basis functions = 15 (C) + 3*5 (H) + 28 (Si) + 15 (F) = 73
n_basis_functions = 73

n_orbitals = n_basis_functions  # Assuming one spatial orbital per basis function

csf_count = calculate_csf(n_electrons, n_orbitals)

print(f"Estimated upper bound on number of CSFs for full CI: {csf_count}")
