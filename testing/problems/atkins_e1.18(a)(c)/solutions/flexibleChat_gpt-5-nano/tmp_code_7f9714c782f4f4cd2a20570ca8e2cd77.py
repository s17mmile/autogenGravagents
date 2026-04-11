from typing import Dict
from dataclasses import dataclass

@dataclass
class Pressures:
    partials_atm: Dict[str, float]
    total_atm: float
    total_kPa: float


def compute_pressures(species_moles: Dict[str, float], V_dm3: float, T_K: float, R: float = 0.082057) -> Pressures:
    """
    Compute partial and total pressures for a gas mixture using the ideal gas law.

    Returns a Pressures dataclass with:
      - partials_atm: dict of P_i for each species in atm
      - total_atm: total pressure in atm
      - total_kPa: total pressure in kPa
    """
    if not species_moles:
        raise ValueError("species_moles must be a non-empty mapping of species to moles.")
    if V_dm3 <= 0:
        raise ValueError("Volume must be positive.")
    if T_K <= 0:
        raise ValueError("Temperature must be positive.")

    # Non-negative mole check
    for name, n in species_moles.items():
        if n < 0:
            raise ValueError(f"Moles for species '{name}' must be non-negative. Got {n}.")

    n_total = sum(species_moles.values())
    V_L = V_dm3  # 1 dm3 = 1 L
    P_total_atm = n_total * R * T_K / V_L

    # Partial pressures from n_i * R * T / V
    partials_atm = {name: n * R * T_K / V_L for name, n in species_moles.items()}
    P_total_kPa = P_total_atm * 101.325

    return Pressures(partials_atm=partials_atm, total_atm=P_total_atm, total_kPa=P_total_kPa)


def main():
    # Example: H2 and N2 in 22.4 dm3 at 273.15 K
    species = {"H2": 2.0, "N2": 1.0}
    V = 22.4
    T = 273.15
    pr = compute_pressures(species, V, T)

    print("Partial pressures (atm):")
    for gas, p in pr.partials_atm.items():
        print(f"{gas}: {p:.6f} atm")
    print(f"Total pressure: {pr.total_atm:.6f} atm")
    print(f"Total pressure (kPa): {pr.total_kPa:.3f} kPa")
    print(f"Sum of partials: {sum(pr.partials_atm.values()):.6f} atm")


if __name__ == "__main__":
    main()
