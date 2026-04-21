# filename: calculate_deltaH0_H2_dissociation.py

def calculate_deltaH0(D0_eV):
    """Calculate Delta H_0^circ in kJ/mol from bond dissociation energy in eV."""
    # Conversion factor from eV to kJ/mol
    eV_to_kJ_per_mol = 96.485
    # Validate input
    if D0_eV <= 0:
        raise ValueError("Bond dissociation energy must be positive.")
    # Calculate Delta H_0^circ in kJ/mol
    deltaH0_kJ_per_mol = D0_eV * eV_to_kJ_per_mol
    return deltaH0_kJ_per_mol

# Given bond dissociation energy for H2 in eV
D0_H2 = 4.4781

# Calculate Delta H_0^circ
deltaH0 = calculate_deltaH0(D0_H2)

print(f"Delta H_0^circ for H2(g) -> 2H(g) = {deltaH0:.4f} kJ/mol")
