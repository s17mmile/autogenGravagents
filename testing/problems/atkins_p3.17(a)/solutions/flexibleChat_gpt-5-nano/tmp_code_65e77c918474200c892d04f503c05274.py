import math

# Temperature and constants
T_TARGET = 500.0  # K
T_REF = 298.0     # K
R = 8.314462618   # J/mol-K
P = 101325.0        # Pa (1 bar)

# Helper: try to use CoolProp if available
def compute_with_coolprop():
    try:
        from CoolProp.CoolProp import PropsSI
        species = ['N2', 'H2', 'NH3']
        H = {}
        S = {}
        for sp in species:
            # HMOLAR and SMOLAR give molar enthalpy/entropy at T and P
            H[sp] = PropsSI('HMOLAR', 'T', T_TARGET, 'P', P, sp)
            S[sp] = PropsSI('SMOLAR', 'T', T_TARGET, 'P', P, sp)
        # Reaction: N2 + 3 H2 -> 2 NH3
        dH = 2 * H['NH3'] - (H['N2'] + 3.0 * H['H2'])  # J/mol
        dS = 2 * S['NH3'] - (S['N2'] + 3.0 * S['H2'])  # J/mol-K
        dG = dH - T_TARGET * dS                       # J/mol
        K = math.exp(-dG / (R * T_TARGET))
        print("Delta H_rxn(500 K) = {0:.6f} kJ/mol".format(dH / 1000.0))
        print("Delta S_rxn(500 K) = {0:.6f} J/mol-K".format(dS))
        print("Delta G_rxn(500 K) = {0:.6f} kJ/mol".format(dG / 1000.0))
        print("K(500 K) = {0:.3e}".format(K))
        return True
    except Exception:
        return False

# Fallback: approximate data (explicit, transparent, and replaceable with JANAF/NASA data)
def compute_with_approx():
    # Data at 298 K (placeholders consistent with common tables)
    # Formation enthalpies in J/mol: N2,H2 are zero; NH3(g) ~ -46.11 kJ/mol
    Hf = {'N2': 0.0, 'H2': 0.0, 'NH3': -46110.0}
    # Standard molar entropies at 298 K in J/mol-K
    S298 = {'N2': 191.5, 'H2': 130.6, 'NH3': 188.0}
    # Approximate heat capacities (constant over 298-500 K) in J/mol-K
    Cp = {'N2': 29.1, 'H2': 28.8, 'NH3': 35.7}

    # Enthalpy at 500 K: H(T) = Hf298 + int Cp dT, with Cp treated as constant
    dT = T_TARGET - T_REF
    H_N2  = Hf['N2']  + Cp['N2'] * dT
    H_H2  = Hf['H2']  + Cp['H2'] * dT
    H_NH3 = Hf['NH3'] + Cp['NH3'] * dT

    # Entropy at 500 K: S(T) = S298 + Cp * ln(T/298)
    import math as _m
    S_N2  = S298['N2']  + Cp['N2'] * _m.log(T_TARGET / T_REF)
    S_H2  = S298['H2']  + Cp['H2'] * _m.log(T_TARGET / T_REF)
    S_NH3 = S298['NH3'] + Cp['NH3'] * _m.log(T_TARGET / T_REF)

    # Reaction thermodynamics at 500 K
    dH = 2.0 * H_NH3 - (H_N2 + 3.0 * H_H2)  # J/mol
    dS = 2.0 * S_NH3 - (S_N2 + 3.0 * S_H2)  # J/mol-K
    dG = dH - T_TARGET * dS               # J/mol
    K = math.exp(-dG / (R * T_TARGET))

    print("Delta H_rxn(500 K) (approx) = {0:.6f} kJ/mol".format(dH / 1000.0))
    print("Delta S_rxn(500 K) (approx) = {0:.6f} J/mol-K".format(dS))
    print("Delta G_rxn(500 K) (approx) = {0:.6f} kJ/mol".format(dG / 1000.0))
    print("K(500 K) (approx) = {0:.3e}".format(K))

    return None

def main():
    # Attempt to compute with CoolProp first; if unavailable, fall back
    if not compute_with_coolprop():
        compute_with_approx()

if __name__ == "__main__":
    main()
