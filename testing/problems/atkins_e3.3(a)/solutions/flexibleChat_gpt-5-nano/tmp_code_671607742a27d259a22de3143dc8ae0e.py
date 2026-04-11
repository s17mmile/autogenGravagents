import math
from typing import Tuple, Optional


def calculate_delta_s(n: float, T1: float, P1: float, T2: float, P2: float,
                      *, unit: str = 'atm', Cp_m: Optional[float] = None, R: float = 8.314462618) -> Tuple[float, float]:
    """
    Compute entropy change for an ideal gas during a state change from (T1, P1) to (T2, P2).

    Parameters:
    - n: amount of substance in moles (must be positive)
    - T1, T2: temperatures in Kelvin (must be > 0)
    - P1, P2: pressures in the specified unit (atm by default) or Pa if unit == 'pa'
    - unit: 'atm' or 'pa'. If 'atm', P1 and P2 are interpreted as atm and converted to Pa internally
    - Cp_m: molar heat capacity at constant pressure (J/mol-K). If None, defaults to 2.5*R
    - R: universal gas constant (J/mol-K)

    Returns:
    - (delta_s_per_mol, delta_s_total): per-mole and total entropy changes in J/(mol K) and J/K
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be > 0 (Kelvin).")
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Pressures must be > 0.")
    if unit not in ('atm', 'pa'):
        raise ValueError("unit must be 'atm' or 'pa'.")

    # Normalize to Pa unconditionally based on provided unit
    if unit == 'atm':
        P1_pa = P1 * 101325.0
        P2_pa = P2 * 101325.0
    else:
        P1_pa = P1
        P2_pa = P2

    if Cp_m is None:
        Cp_m = 2.5 * R

    delta_s_per_mol = Cp_m * math.log(T2 / T1) - R * math.log(P2_pa / P1_pa)
    delta_s_total = n * delta_s_per_mol
    return delta_s_per_mol, delta_s_total


def main():
    # Preset problem values (no user input)
    n = 3.0
    T1 = 298.15  # K
    P1 = 1.0     # atm
    T2 = 398.15  # K
    P2 = 5.0     # atm

    ds_per_mol, ds_total = calculate_delta_s(n, T1, P1, T2, P2, unit='atm')

    print(f"Delta S per mole = {ds_per_mol:.6f} J/(mol K)")
    print(f"Delta S total = {ds_total:.3f} J/K")


def test_known_values():
    # Basic consistency test for the provided problem
    ds_per_mol, ds_total = calculate_delta_s(3.0, 298.15, 1.0, 398.15, 5.0, unit='atm')
    # Check negativity and internal consistency
    assert ds_per_mol < 0
    assert ds_total < 0
    assert math.isclose(ds_total, 3.0 * ds_per_mol, rel_tol=1e-9, abs_tol=1e-9)
    print("Test passed.")


if __name__ == "__main__":
    main()
    test_known_values()
