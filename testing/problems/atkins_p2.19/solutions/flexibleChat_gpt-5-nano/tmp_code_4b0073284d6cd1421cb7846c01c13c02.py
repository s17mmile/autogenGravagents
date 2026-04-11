import math
from typing import Optional, Tuple, Dict

# Preset data (P in Pa, T in K)
P1 = 202.94e3       # Pa
P2 = 81.840e3       # Pa
T1 = 298.15          # K
T2 = 248.44           # K
R = 8.314             # J/mol/K
ln2 = math.log(2.0)

# Optional measurement uncertainties (set to None to disable propagation)
# Units: P in Pa, T in K
sigmaP1: Optional[float] = None  # Pa uncertainty for P1
sigmaP2: Optional[float] = None  # Pa uncertainty for P2
sigmaT1: Optional[float] = None  # K uncertainty for T1
sigmaT2: Optional[float] = None  # K uncertainty for T2


def compute_gamma_from_pressure(P1_val: float, P2_val: float) -> float:
    if P1_val <= 0 or P2_val <= 0:
        raise ValueError("P1 and P2 must be positive")
    return math.log(P1_val / P2_val) / math.log(2.0)


def compute_gamma_from_temperature(T1_val: float, T2_val: float) -> float:
    if T1_val <= 0 or T2_val <= 0:
        raise ValueError("T1 and T2 must be positive")
    return 1.0 - math.log(T2_val / T1_val) / math.log(2.0)


def compute_sigma_gamma_pressure(P1_val: float, P2_val: float,
                                 sigmaP1_val: Optional[float], sigmaP2_val: Optional[float]) -> Optional[float]:
    if (sigmaP1_val is None) or (sigmaP2_val is None):
        return None
    sigma_lnP = math.sqrt((sigmaP1_val / P1_val)**2 + (sigmaP2_val / P2_val)**2)
    return sigma_lnP / math.log(2.0)


def compute_sigma_gamma_temperature(T1_val: float, T2_val: float,
                                  sigmaT1_val: Optional[float], sigmaT2_val: Optional[float]) -> Optional[float]:
    if (sigmaT1_val is None) or (sigmaT2_val is None):
        return None
    sigma_lnT = math.sqrt((sigmaT2_val / T2_val)**2 + (sigmaT1_val / T1_val)**2)
    return sigma_lnT / math.log(2.0)


def combine_gammas(gamma_p: float, gamma_t: float,
                   sigma_gamma_p: Optional[float] = None,
                   sigma_gamma_t: Optional[float] = None) -> Tuple[float, Optional[float]]:
    # If uncertainties are provided, perform inverse-variance weighted combination
    if (sigma_gamma_p is not None) and (sigma_gamma_t is not None):
        inv_p = 1.0 / (sigma_gamma_p**2)
        inv_t = 1.0 / (sigma_gamma_t**2)
        gamma_comb = (gamma_p * inv_p + gamma_t * inv_t) / (inv_p + inv_t)
        sigma_gamma_comb = 1.0 / math.sqrt(inv_p + inv_t)
        return gamma_comb, sigma_gamma_comb
    # Otherwise, fall back to simple average
    gamma_comb = 0.5 * (gamma_p + gamma_t)
    return gamma_comb, None


def cp_from_gamma(gamma_val: float, Rval: float = 8.314) -> Tuple[float, float]:
    if gamma_val <= 1.0:
        raise ValueError("gamma must be > 1 for positive Cv")
    Cv = Rval / (gamma_val - 1.0)
    Cp = gamma_val * Cv
    return Cp, Cv


def main() -> Dict[str, float]:
    # Step 1: compute gammas from data
    gamma_p = compute_gamma_from_pressure(P1, P2)
    gamma_t = compute_gamma_from_temperature(T1, T2)

    # Step 2: compute uncertainties from provided inputs (if any)
    sigma_gamma_p = compute_sigma_gamma_pressure(P1, P2, sigmaP1, sigmaP2)
    sigma_gamma_t = compute_sigma_gamma_temperature(T1, T2, sigmaT1, sigmaT2)

    # Step 3: combine gammas
    gamma_comb, sigma_gamma_comb = combine_gammas(gamma_p, gamma_t, sigma_gamma_p, sigma_gamma_t)

    # Step 4: compute Cp and Cv from gammas
    Cp_p, Cv_p = cp_from_gamma(gamma_p, R)
    Cp_t, Cv_t = cp_from_gamma(gamma_t, R)
    Cp_comb, Cv_comb = cp_from_gamma(gamma_comb, R)

    # End-member Cp for comparison
    Cp_p_end = R * gamma_p / (gamma_p - 1.0)
    Cp_t_end = R * gamma_t / (gamma_t - 1.0)

    # Optional uncertainty propagation to Cp
    sigma_Cp_comb = None
    if sigma_gamma_comb is not None:
        sigma_Cp_comb = abs(-R / (gamma_comb - 1.0)**2) * sigma_gamma_comb

    results = {
        "gamma_p": gamma_p,
        "gamma_t": gamma_t,
        "gamma_combined": gamma_comb,
        "Cp_p": Cp_p,
        "Cv_p": Cv_p,
        "Cp_t": Cp_t,
        "Cv_t": Cv_t,
        "Cp_combined": Cp_comb,
        "Cv_combined": Cv_comb,
        "Cp_p_end": Cp_p_end,
        "Cp_t_end": Cp_t_end,
        "sigma_gamma_combined": sigma_gamma_comb,
        "sigma_Cp_combined": sigma_Cp_comb,
    }

    # Quick human-friendly summary
    print("P1/P2 =", P1 / P2)
    print("A = ln(P1/P2) =", math.log(P1 / P2))
    print("D = ln(T2/T1) =", math.log(T2 / T1))
    print("gamma_p =", gamma_p)
    print("gamma_t =", gamma_t)
    print("gamma_combined =", gamma_comb)
    if sigma_gamma_comb is not None:
        print("sigma_gamma_combined =", sigma_gamma_comb)
    print("")
    print("Cp_p =", Cp_p, "J/mol/K; Cv_p =", Cv_p)
    print("Cp_t =", Cp_t, "J/mol/K; Cv_t =", Cv_t)
    print("Cp_combined =", Cp_comb, "J/mol/K; Cv_combined =", Cv_comb)
    print("Cp_p_end =", Cp_p_end, "; Cp_t_end =", Cp_t_end)

    return results


if __name__ == "__main__":
    main()
