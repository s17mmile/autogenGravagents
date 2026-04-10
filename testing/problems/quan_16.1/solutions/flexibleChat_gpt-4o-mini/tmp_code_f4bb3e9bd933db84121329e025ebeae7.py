import psi4

# Set the molecular geometry for CH2SiHF
# Ensure that the coordinates accurately represent the intended structure
molecule = psi4.geometry("""
  0 1
  C 0.000000 0.000000 0.000000
  H 0.000000 0.000000 1.000000
  H 0.000000 0.000000 -1.000000
  Si 0.000000 1.500000 0.000000
  F 0.000000 2.500000 0.000000
""")

# Set the basis set
basis_set = '6-31G**'  # A double-zeta basis set with polarization functions

# Set the calculation options
psi4.set_options({
    'basis': basis_set,
    'scf_type': 'df',
    'reference': 'rhf',
})

try:
    # Perform a geometry optimization before the CI calculation
    psi4.optimize('scf')

    # Perform a CI calculation
    energy, wfn = psi4.energy('CISD', return_wfn=True)

    # Calculate the number of CSFs
    num_csfs = wfn.n_det

    # Output the results
    print(f'Number of CSFs using {basis_set} and CISD method: {num_csfs}')

except Exception as e:
    print(f'An error occurred during the calculation: {e}')