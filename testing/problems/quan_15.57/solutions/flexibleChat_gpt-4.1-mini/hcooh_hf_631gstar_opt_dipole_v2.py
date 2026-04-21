# filename: hcooh_hf_631gstar_opt_dipole_v2.py
import psi4
import math


def main():
    # Set memory and output options
    psi4.set_memory('500 MB')
    psi4.core.set_output_file('output.dat', False)  # Optional output file

    # Define HCOOH molecule using Z-matrix to control dihedral angle
    # Atom order: C, O (carbonyl), O (hydroxyl), H (formyl), H (hydroxyl)
    # Dihedral angle O-C-O-H set to 0 degrees explicitly
    molecule = psi4.geometry("""
    0 1
    C
    O 1 1.20
    O 1 1.35 2 120.0
    H 1 1.10 2 120.0 3 0.0
    H 3 0.96 1 124.0 2 0.0
    units angstrom
    """)

    # Verify dihedral angle O-C-O-H (atoms 2-1-3-4 in 1-based indexing)
    dihedral = molecule.measure_dihedral(1, 0, 2, 3) * 180.0 / math.pi  # convert rad to degrees
    print(f'Initial O-C-O-H dihedral angle: {dihedral:.2f} degrees')

    # Set computational options
    psi4.set_options({
        'basis': '6-31G*',
        'scf_type': 'pk',
        'reference': 'rhf'
    })

    try:
        # Perform geometry optimization at HF/6-31G* level
        energy, wfn = psi4.optimize('hf/6-31G*', molecule=molecule, return_wfn=True)
    except Exception as e:
        print(f'Geometry optimization failed: {e}')
        return

    # Calculate dipole moment from optimized geometry
    dipole = wfn.variable('SCF DIPOLE')  # Dipole moment components in atomic units
    dipole_debye = [component * 2.541746 for component in dipole]  # Convert to Debye

    dipole_magnitude = math.sqrt(sum(x**2 for x in dipole_debye))

    # Print results
    print(f'Optimized energy (Hartree): {energy:.6f}')
    print(f'Dipole moment components (Debye): X={dipole_debye[0]:.4f}, Y={dipole_debye[1]:.4f}, Z={dipole_debye[2]:.4f}')
    print(f'Total dipole moment (Debye): {dipole_magnitude:.4f}')


if __name__ == '__main__':
    main()
