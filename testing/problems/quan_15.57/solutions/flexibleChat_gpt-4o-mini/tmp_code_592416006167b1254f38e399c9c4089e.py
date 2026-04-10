import psi4

# Set the memory and output file
psi4.set_memory('500 MB')
output_file = 'output.dat'
psi4.set_output_file(output_file, overwrite=True)

# Define the molecular geometry of HCOOH with OCOH dihedral angle of 0 degrees
# Coordinates are set to ensure the dihedral angle is 0
hcooh = """
0 1
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.200000
O 0.000000 1.200000 0.000000
H 0.000000 -0.800000 0.000000
H 0.000000 0.800000 1.200000
"""

# Set the calculation options
psi4.set_options({
    'basis': '6-31G*',
    'scf_type': 'df',
    'reference': 'rhf'
})

try:
    # Perform geometry optimization
    energy, wfn = psi4.optimize('HF', return_wfn=True, molecule=hcooh)
    
    # Calculate dipole moment
    dipole_moment = wfn.dipole_moment()
    
    # Save results to a file
    with open('results.txt', 'w') as f:
        f.write(f'Optimized Energy: {energy} Hartree\n')
        f.write(f'Dipole Moment: {dipole_moment}\n')
        
    print(f'Results saved to results.txt')
except Exception as e:
    print(f'An error occurred: {e}')