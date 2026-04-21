# filename: calculate_bond_length_hcl.py
import math

def calculate_bond_length(line_spacing_hz):
    if line_spacing_hz <= 0:
        raise ValueError("Line spacing must be positive.")

    # Constants
    h = 6.62607015e-34  # Planck's constant in J*s

    # Atomic masses in atomic mass units (u)
    m_H_u = 1.007825
    m_Cl_u = 34.968853

    # Atomic mass unit in kg
    u_kg = 1.66053906660e-27

    # Convert atomic masses to kg
    m_H = m_H_u * u_kg
    m_Cl = m_Cl_u * u_kg

    # Reduced mass in kg
    mu = (m_H * m_Cl) / (m_H + m_Cl)

    # Rotational constant B in Hz
    B = line_spacing_hz / 2

    # Calculate bond length r in meters
    r_m = math.sqrt(h / (8 * math.pi**2 * mu * B))

    # Convert bond length to angstroms
    r_angstrom = r_m * 1e10

    return r_m, r_angstrom

# Given line spacing
line_spacing = 6.26e11  # Hz

bond_length_m, bond_length_angstrom = calculate_bond_length(line_spacing)

print(f"Bond length of H-35Cl molecule: {bond_length_m:.3e} meters")
print(f"Bond length of H-35Cl molecule: {bond_length_angstrom:.3f} angstroms")
