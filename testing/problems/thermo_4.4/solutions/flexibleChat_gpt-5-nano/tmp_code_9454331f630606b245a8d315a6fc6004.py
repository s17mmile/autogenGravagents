def compute_hf_no_975(Hf_NO_298: float = 90.29,
                      Cp_NO: float = 37.11,
                      Cp_N2: float = 29.10,
                      Cp_O2: float = 29.37,
                      T_ref: float = 298.15,
                      T: float = 975.0) -> dict:
    # Cp values are in J/mol/K
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
