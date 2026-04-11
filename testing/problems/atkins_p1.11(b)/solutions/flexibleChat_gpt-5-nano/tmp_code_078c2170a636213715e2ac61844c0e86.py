from typing import NamedTuple
import math


class ZOutput(NamedTuple):
    Z: float
    v_m_L_per_mol: float


def compute_Z(rho_kg_per_m3: float, M_g_per_mol: float, P_atm: float, T_K: float, R: float = 0.08205736) -> ZOutput:
    """Compute the compressibility factor Z and molar volume v_m (in L/mol) from density, molar mass, pressure and temperature.

    Arguments:
        rho_kg_per_m3: density in kg/m^3
        M_g_per_mol: molar mass in g/mol
        P_atm: pressure in atm
        T_K: temperature in K
        R: gas constant in L*atm/(mol*K) (default 0.08205736)

    Returns:
        ZOutput(Z, v_m_L_per_mol) where Z is dimensionless and v_m_L_per_mol is the molar volume in L/mol
    """
    # Basic input validation
    if not math.isfinite(rho_kg_per_m3) or rho_kg_per_m3 <= 0:
        raise ValueError("rho_kg_per_m3 must be a positive finite number")
    if not math.isfinite(M_g_per_mol) or M_g_per_mol <= 0:
        raise ValueError("M_g_per_mol must be a positive finite number")
    if not math.isfinite(P_atm) or P_atm <= 0:
        raise ValueError("P_atm must be a positive finite number")
    if not math.isfinite(T_K) or T_K <= 0:
        raise ValueError("T_K must be a positive finite number")

    M_kg_per_mol = M_g_per_mol / 1000.0
    v_m3_per_mol = M_kg_per_mol / rho_kg_per_m3
    v_m_L_per_mol = v_m3_per_mol * 1000.0
    Z = (P_atm * v_m_L_per_mol) / (R * T_K)
    return ZOutput(Z, v_m_L_per_mol)


if __name__ == "__main__":
    rho = 133.2
    M = 18.02
    P = 327.6
    T = 776.4

    Zout = compute_Z(rho, M, P, T)
    print("Molar volume v_m = {:.6f} L/mol".format(Zout.v_m_L_per_mol))
    print("Compressibility factor Z = {:.6f}".format(Zout.Z))