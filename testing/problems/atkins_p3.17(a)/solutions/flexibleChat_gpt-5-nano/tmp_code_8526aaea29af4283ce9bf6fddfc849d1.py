import json
import math
import os

# Temperature and constants
T_TARGET = 500.0  # K
R = 8.314462618  # J/mol-K
T_REF = 298.0    # K

# Attempt to load NASA7 coefficients from a compact data bundle
DATA_PATHS = [
    os.path.join("data", "nasa_coeffs.json"),
    "nasa_coeffs.json",
]


def nasa7_eval(a7: list, T: float) -> tuple:
    """
    Evaluate NASA7 polynomial for Cp, H, S at temperature T (K).
    a7: list of 7 coefficients [a1, a2, a3, a4, a5, a6, a7]
    Returns (H, S, Cp) in J/mol, J/mol-K, J/mol-K respectively.
    Assumes standard NASA7 form:
      Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5/T^2
      H/(RT) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 - a5/(2*T^2) + a6/T
      S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 - a5/(2*T^2) + a7
    Then H = H/(RT) * R * T, S = (S/R) * R, Cp = (Cp/R) * R
    """
    t = T
    a1, a2, a3, a4, a5, a6, a7 = a7
    cp_R = a1 + a2 * t + a3 * t * t + a4 * t * t * t + a5 / (t * t)
    H_RT = (
        a1 + a2 * t / 2.0 + a3 * t * t / 3.0 + a4 * t * t * t / 4.0
        - a5 / (2.0 * t * t) + a6 / t
    )
    S_R = a1 * math.log(t) + a2 * t + a3 * t * t / 2.0 + a4 * t * t * t / 3.0 - a5 / (2.0 * t * t) + a7
    H = H_RT * R * T
    S = S_R * R
    Cp = cp_R * R
    return H, S, Cp


def load_coeffs(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def compute_with_nasa(coeffs: dict):
    # Expecting keys: 'N2', 'H2', 'NH3', each with 'coeffs': [a1..a7], 'Tmin','Tmax'
    species = ['N2', 'H2', 'NH3']
    H = {}
    S = {}
    Cp = {}
    for sp in species:
        a7 = coeffs[sp]['coeffs']
        h, s, cp = nasa7_eval(a7, T_TARGET)
        H[sp] = h
        S[sp] = s
        Cp[sp] = cp
    # Reaction: N2 + 3 H2 -> 2 NH3
    dH = 2.0 * H['NH3'] - (H['N2'] + 3.0 * H['H2'])
    dS = 2.0 * S['NH3'] - (S['N2'] + 3.0 * S['H2'])
    dG = dH - T_TARGET * dS
    K = math.exp(-dG / (R * T_TARGET))

    print("Delta Hrxn(500 K) = {0:.6f} kJ/mol".format(dH / 1000.0))
    print("Delta Srxn(500 K) = {0:.6f} J/mol-K".format(dS))
    print("Delta Grxn(500 K) = {0:.6f} kJ/mol".format(dG / 1000.0))
    print("K(500 K) = {0:.3e}".format(K))


def compute_with_fallback():
    # Placeholder, transparent approximate data (to be replaced by NASA data)
    Hf = {'N2': 0.0, 'H2': 0.0, 'NH3': -46110.0}  # J/mol
    S298 = {'N2': 191.5, 'H2': 130.6, 'NH3': 188.0}  # J/mol-K
    Cp = {'N2': 29.1, 'H2': 28.8, 'NH3': 35.7}        # J/mol-K
    dT = T_TARGET - T_REF
    H_N2  = Hf['N2']  + Cp['N2'] * dT
    H_H2  = Hf['H2']  + Cp['H2'] * dT
    H_NH3 = Hf['NH3'] + Cp['NH3'] * dT

    import math as _m
    S_N2  = S298['N2']  + Cp['N2'] * _m.log(T_TARGET / T_REF)
    S_H2  = S298['H2']  + Cp['H2'] * _m.log(T_TARGET / T_REF)
    S_NH3 = S298['NH3'] + Cp['NH3'] * _m.log(T_TARGET / T_REF)

    dH = 2.0 * H_NH3 - (H_N2 + 3.0 * H_H2)
    dS = 2.0 * S_NH3 - (S_N2 + 3.0 * S_H2)
    dG = dH - T_TARGET * dS
    K = math.exp(-dG / (R * T_TARGET))

    print("Delta Hrxn(500 K) (approx) = {0:.6f} kJ/mol".format(dH / 1000.0))
    print("Delta Srxn(500 K) (approx) = {0:.6f} J/mol-K".format(dS))
    print("Delta Grxn(500 K) (approx) = {0:.6f} kJ/mol".format(dG / 1000.0))
    print("K(500 K) (approx) = {0:.3e}".format(K))


def main():
    # Try NASA7 bundle first
    coeff_path = None
    for p in DATA_PATHS:
        if os.path.exists(p):
            coeff_path = p
            break
    if coeff_path is not None:
        try:
            data = load_coeffs(coeff_path)
            compute_with_nasa(data)
            return
        except Exception as e:
            print("Warning: failed to parse NASA coefficients: {}".format(e))
            print("Falling back to approximate data.")
            compute_with_fallback()
            return
    else:
        print("NASA7 data bundle not found. Falling back to approximate data (placeholders).")
        compute_with_fallback()


if __name__ == "__main__":
    main()
