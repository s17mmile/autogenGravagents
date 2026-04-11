import psi4
import numpy as np

# Configuration (preset, no user input)
DIH_TARGET = 0.0        # target OCOH dihedral in degrees
DIH_TOL = 0.5           # tolerance for final dihedral verification in degrees
HF_BASIS = 'HF/6-31G*'
MEMORY = '4 GB'
NTHREADS = 4

psi4.set_memory(MEMORY)
psi4.set_num_threads(NTHREADS)
print("Psi4 version:", psi4.__version__)

# Helper: compute dihedral angle in degrees from coordinates

def dihedral_deg(p, i, j, k, l):
    pi, pj, pk, pl = p[i], p[j], p[k], p[l]
    b0 = -1.0 * (pj - pi)
    b1 = pk - pj
    b2 = pl - pk
    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0:
        return np.nan
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

# Robust dipole extraction from a psi4 wavefunction (version-guarded)

def extract_dipole_from_wfn(wfn):
    candidates = ["dip_moment", "dipole_moment", "dipole", "dip_mom"]
    for attr in candidates:
        if hasattr(wfn, attr):
            val = getattr(wfn, attr)()
            arr = np.asarray(val, dtype=float).reshape(-1)
            return arr
    raise AttributeError("Dipole moment API not found on this Psi4 wavefunction object.")

# Geometry: HCOOH with OCOH dihedral constrained to 0 degrees
# Atom order (1-based): 1:C, 2:O(carbonyl), 3:O(hydroxyl), 4:H(formyl), 5:H(hydroxyl)
# Dihedral constraint: O(2) - C(1) - O(3) - H(4) = DIH_TARGET
geom_str = (
    "0 1\n"
    "C 0.000000 0.000000 0.000000\n"
    "O 0.000000 0.000000 1.200000\n"
    "O 1.430000 0.000000 0.000000\n"
    "H 0.000000 -1.090000 0.000000\n"
    "H 1.430000 0.000000 -0.960000\n"
    f"D 2 1 3 4 {DIH_TARGET:.6f}\n"
)

mol = psi4.geometry(geom_str)

# Constrained HF optimization with 6-31G* basis (DIH_TARGET enforced via constraint line in geometry)
E_opt, wfn = psi4.optimize(HF_BASIS, return_wfn=True)
print("Optimized HF energy (Hartree):", float(E_opt))

# Dipole moment extraction (robust)
try:
    mu_vec_au = extract_dipole_from_wfn(wfn)
    mu_debye = float(np.linalg.norm(mu_vec_au) * 2.541746)
    print("Dipole moment (Debye, magnitude):", mu_debye)
    print("Dipole vector (a.u., x, y, z):", mu_vec_au.tolist())
except Exception as e:
    print("Warning: could not extract dipole moment:", e)

# Final coordinates and dihedral verification
coords = None
try:
    g = mol.geometry()
    try:
        coords = np.array(g.to_array(), dtype=float)
    except Exception:
        # Fallback to explicit x,y,z getters (1-based indexing)
        nat = g.natoms()
        coords = []
        for idx in range(1, nat + 1):
            coords.append([float(g.x(idx)), float(g.y(idx)), float(g.z(idx))])
        coords = np.array(coords, dtype=float)
except Exception as e:
    coords = None

if coords is not None and coords.shape[0] >= 4:
    # Dihedral order used above: O(2) - C(1) - O(3) - H(4) -> indices [1,0,2,3] in 0-based
    final_dihed = dihedral_deg(coords, 1, 0, 2, 3)
    print("Final O-C-O-H dihedral (deg):", final_dihed)
    if not np.isnan(final_dihed) and abs(final_dihed - DIH_TARGET) > DIH_TOL:
        print("Warning: final dihedral deviates from target by more than tolerance (", DIH_TOL, "deg). Consider re-running with a tighter constraint or verify constraint presence.")
else:
    print("Warning: final coordinates not available for dihedral verification.")

# Optional: write final geometry to XYZ for inspection
if coords is not None:
    labels = ['C','O','O','H','H']
    try:
        with open('hcooh_optimized.xyz', 'w') as f:
            f.write(f"5\nHCOOH optimized with OCOH dihedral {DIH_TARGET:.1f} deg\n")
            for lab, (x, y, z) in zip(labels, coords):
                f.write(f"{lab} {x:.6f} {y:.6f} {z:.6f}\n")
        print("Wrote final XYZ to hcooh_optimized.xyz")
    except Exception as e:
        print("Warning: could not write XYZ file:", e)

print("Done.")
