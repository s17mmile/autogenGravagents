import psi4
import numpy as np

# Settings (preset, no user input)
psi4.set_memory('4 GB')
psi4.set_num_threads(4)

# Geometry: HCOOH with OCOH dihedral target = 0 degrees, using a modredundant constraint
# Atom ordering in the geometry block (1-based):
# 1: C, 2: O(carbonyl), 3: O(hydroxyl), 4: H(formyl), 5: H(hydroxyl)
# Dihedral to constrain: O(2) - C(1) - O(3) - H(4) = 0 deg
geom_str = '''
0 1
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.200000
O 1.430000 0.000000 0.000000
H 0.000000 -1.090000 0.000000
H 1.430000 0.000000 -0.960000
D 2 1 3 4 0.0
'''

mol = psi4.geometry(geom_str)

# Constrained HF optimization with 6-31G* basis
E_opt, wfn = psi4.optimize('HF/6-31G*', return_wfn=True)
print("Optimized HF energy (Hartree):", float(E_opt))

# Robust dipole moment extraction (try multiple API paths)
def extract_dipole_from_wfn(wfn):
    # Try common Psi4 attribute names
    for attr in ('dip_moment', 'dipole_moment', 'dipole'):
        if hasattr(wfn, attr):
            val = getattr(wfn, attr)()
            arr = np.asarray(val, dtype=float).reshape(-1)
            return arr
    # Fallback: if the above fail, try direct property access
    if hasattr(wfn, 'dipole'):
        val = wfn.dipole()
        return np.asarray(val, dtype=float).reshape(-1)
    raise AttributeError('Dipole moment API not found on this Psi4 wavefunction object.')

try:
    mu_vec_au = extract_dipole_from_wfn(wfn)
    mu_vec_debye = float(np.linalg.norm(mu_vec_au) * 2.541746)
    print("Dipole moment (Debye, magnitude):", mu_vec_debye)
    print("Dipole vector (a.u.) x, y, z:", mu_vec_au.tolist())
except Exception as e:
    print("Warning: could not extract dipole moment robustly:", e)

# Final dihedral verification (best effort across versions)
# Attempt to fetch final coordinates from the optimized geometry and compute the O-C-O-H dihedral
coords_final = None
try:
    # Try to pull coordinates from the molecule object if available
    g = mol.geometry()
    n = g.natoms()
    coords = []
    for i in range(n):
        # Psi4 atom indices are 1-based; helper methods may be named x(i+1)/y(i+1)/z(i+1) depending on version
        try:
            coords.append([float(g.x(i+1)), float(g.y(i+1)), float(g.z(i+1))])
        except Exception:
            coords.append([float(g.xyz(i)[0]), float(g.xyz(i)[1]), float(g.xyz(i)[2])])
    coords_final = np.array(coords, dtype=float)
except Exception:
    coords_final = None

def dihedral_deg(p, i, j, k, l):
    # p: (N,3) array of coordinates
    pi, pj, pk, pl = p[i], p[j], p[k], p[l]
    b0 = -1.0 * (pj - pi)
    b1 = pk - pj
    b2 = pl - pk
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.degrees(np.arctan2(y, x))
    return angle

if coords_final is not None:
    # OCOH dihedral: O(2) - C(1) - O(3) - H(4) in the defined ordering (1-based)
    # Convert to 0-based indices: [1, 0, 2, 3]
    i, j, k, l = 1, 0, 2, 3
    try:
        final_dihedral = dihedral_deg(coords_final, i, j, k, l)
        print("Final O-C-O-H dihedral (degrees):", final_dihedral)
    except Exception:
        print("Warning: could not compute final dihedral from coordinates.")
else:
    print("Final coordinates could not be retrieved to compute the dihedral. If available, please verify with your Psi4 version.")

print("Note: The dihedral constraint targets 0 degrees. If the final dihedral deviates significantly, consider re-running with a tighter convergence or alternative constrained optimizer.")