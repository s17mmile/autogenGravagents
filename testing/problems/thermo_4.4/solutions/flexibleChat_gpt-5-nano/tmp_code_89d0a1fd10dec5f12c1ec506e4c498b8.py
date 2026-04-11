from typing import TypedDict

class HfResult(TypedDict):
    Hf_NO_975_kJ_per_mol: float
    DeltaCp_J_per_K: float
    DeltaT_K: float
    DeltaH_kJ_per_mol: float


def compute_hf_no_975(
    Hf_NO_298: float = 90.29,
    Cp_NO: float = 37.11,
    Cp_N2: float = 29.10,
    Cp_O2: float = 29.37,
    T_ref: float = 298.15,
    T: float = 975.0,
) -> HfResult:
    """Compute DeltaCp, DeltaT, DeltaH (298 K to T), and Hf(NO) at T under constant Cp°.

    Assumptions:
    - Cp° values are taken as constants at 298.15 K for all species.
    - Reaction considered: N2(g) + 1/2 O2(g) -> NO(g)

    Parameters:
    - Hf_NO_298: Standard enthalpy of formation of NO(g) at 298 K (kJ/mol)
    - Cp_NO, Cp_N2, Cp_O2: Heat capacities at constant pressure (J/mol/K)
    - T_ref: Reference temperature (K) for Cp° and Hf° data
    - T: Target temperature (K)

    Returns:
    - A dict-like object (TypedDict) with keys:
      Hf_NO_975_kJ_per_mol, DeltaCp_J_per_K, DeltaT_K, DeltaH_kJ_per_mol
    """
    if Cp_NO < 0 or Cp_N2 < 0 or Cp_O2 < 0:
        raise ValueError("Cp values must be non-negative.")
    if T < T_ref:
        raise ValueError("Target temperature T must be >= T_ref.")

    delta_cp = Cp_NO - Cp_N2 - 0.5 * Cp_O2  # J/mol/K
    delta_T = T - T_ref                    # K
    delta_H_kJ = delta_cp * delta_T / 1000.0  # convert to kJ/mol
    Hf_975 = Hf_NO_298 + delta_H_kJ

    return {
        "Hf_NO_975_kJ_per_mol": Hf_975,
        "DeltaCp_J_per_K": delta_cp,
        "DeltaT_K": delta_T,
        "DeltaH_kJ_per_mol": delta_H_kJ,
    }


def main():
    res = compute_hf_no_975()
    print(f"DeltaCp = {res['DeltaCp_J_per_K']:.2f} J/mol/K")
    print(f"DeltaT = {res['DeltaT_K']:.2f} K")
    print(f"DeltaH (298 -> 975) = {res['DeltaH_kJ_per_mol']:.3f} kJ/mol")
    print(f"Hf_NO (975 K) = {res['Hf_NO_975_kJ_per_mol']:.3f} kJ/mol")


if __name__ == "__main__":
    main()
