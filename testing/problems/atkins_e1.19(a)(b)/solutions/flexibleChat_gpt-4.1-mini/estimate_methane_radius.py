# filename: estimate_methane_radius.py
import math

# More precise Avogadro's number
AVOGADRO_NUMBER = 6.02214076e23

# Given critical volume in cm^3/mol
CRITICAL_VOLUME_CM3_PER_MOL = 98.7

# Function to estimate molecular radius from critical volume
def estimate_molecular_radius(V_c_cm3_per_mol):
    # Convert volume from cm^3 to m^3
    V_c_m3_per_mol = V_c_cm3_per_mol * 1e-6
    
    # Volume per molecule in m^3
    V_molecule = V_c_m3_per_mol / AVOGADRO_NUMBER
    
    # Calculate radius assuming spherical molecule: V = (4/3) * pi * r^3
    r = (3 * V_molecule / (4 * math.pi))**(1/3)
    return r

radius = estimate_molecular_radius(CRITICAL_VOLUME_CM3_PER_MOL)
print(f"Estimated radius of methane molecule: {radius:.3e} meters")
