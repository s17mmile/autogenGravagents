import math

# Constants based on the provided data (ASCII, no unicode)
M_kg_per_mol = 0.018015       # kg/mol
rho_ice_kgm3 = 920.0          # kg/m^3
rho_liq_kgm3 = 1000.0         # kg/m^3
DeltaHfus_J_per_mol = 6010.0  # J/mol
T0_K = 273.15                   # K, reference coexistence at P0 = 1 bar
P0_bar = 1.0                    # bar
P_target_bar = 50.0             # bar
step_bar = 1.0                  # integration step in bar (1 bar ~ 1e5 Pa)

# Derived molar volumes and DeltaV
Vm_ice = M_kg_per_mol / rho_ice_kgm3   # m^3/mol
Vm_liq = M_kg_per_mol / rho_liq_kgm3   # m^3/mol
DeltaV_base = Vm_liq - Vm_ice           # m^3/mol (negative for ice Ih -> liquid)

# Default, constant DeltaV and DeltaHfus unless user supplies functions (accepts Pa input)
def _DeltaV(P_pa):
    return DeltaV_base

def _DeltaHfus(P_pa):
    return DeltaHfus_J_per_mol

def dT_dP(T_K, P_pa, DeltaV_func=_DeltaV, DeltaHfus_func=_DeltaHfus):
    """Return dT/dP in K/Pa for given T and P (P in Pa)."""
    return T_K * DeltaV_func(P_pa) / DeltaHfus_func(P_pa)

def estimate_Tm_at_P(P_target_bar: float = P_target_bar,
                     P0_bar: float = P0_bar,
                     T0_K: float = T0_K,
                     M_kg_per_mol: float = M_kg_per_mol,
                     rho_ice_kgm3: float = rho_ice_kgm3,
                     rho_liq_kgm3: float = rho_liq_kgm3,
                     DeltaHfus_J_per_mol: float = DeltaHfus_J_per_mol,
                     step_bar: float = step_bar,
                     deltaV_func=None,
                     deltaHfus_func=None,
                     use_rk4: bool = True) -> (float, float):
    """Estimate melting temperature Tm in Kelvin at target pressure (bar).

    Returns a tuple (Tm_K, P_final_bar).

    - Uses the Clapeyron relation dT/dP = T * DeltaV(P) / DeltaHfus(P).
    - DeltaV = V_liq - V_ice in m^3/mol. DeltaHfus in J/mol.
    - Pressure is propagated from P0_bar to P_target_bar using RK4 (default).
    - If deltaV_func/DeltaHfus_func are provided, they must accept P_pa (Pa) and return
      DeltaV (m^3/mol) and DeltaHfus (J/mol) respectively.
    """
    if P_target_bar < P0_bar:
        raise ValueError("P_target_bar must be >= P0_bar (non-decreasing pressure path).")

    # Initial molar volumes and constants
    Vm_ice = M_kg_per_mol / rho_ice_kgm3
    Vm_liq = M_kg_per_mol / rho_liq_kgm3
    DeltaV_base_local = Vm_liq - Vm_ice

    def DeltaV(P_pa: float) -> float:
        return (DeltaV_base_local if deltaV_func is None else deltaV_func(P_pa))

    def DeltaHfus(P_pa: float) -> float:
        return (DeltaHfus_J_per_mol if deltaHfus_func is None else deltaHfus_func(P_pa))

    # dT/dP as a function of T and P (Pa). Returns K/Pa.
    def dT_dP_local(T_K, P_pa):
        return T_K * DeltaV(P_pa) / DeltaHfus(P_pa)

    # Initial state in SI units (Pa)
    P_pa = P0_bar * 1e5
    target_pa = P_target_bar * 1e5
    T_K = T0_K
    h_pa = step_bar * 1e5

    if P_target_bar == P0_bar:
        return T_K, P_target_bar

    if use_rk4:
        while P_pa < target_pa - 1e-9:
            h = min(h_pa, target_pa - P_pa)
            k1 = dT_dP_local(T_K, P_pa)
            k2 = dT_dP_local(T_K + 0.5 * k1 * h, P_pa + 0.5 * h)
            k3 = dT_dP_local(T_K + 0.5 * k2 * h, P_pa + 0.5 * h)
            k4 = dT_dP_local(T_K + k3 * h, P_pa + h)
            T_K += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            P_pa += h
    else:
        while P_pa < target_pa - 1e-9:
            h = min(h_pa, target_pa - P_pa)
            dT = dT_dP_local(T_K, P_pa) * h
            T_K += dT
            P_pa += h

    return T_K, P_target_bar

# Linear (first-order) approximation for comparison: dT ≈ (T0 * DeltaV / DeltaHfus) * DeltaP
def linear_approx_Tm_K(P_target_bar: float = P_target_bar) -> float:
    DeltaV = DeltaV_base
    dT_dP_const = T0_K * DeltaV / DeltaHfus_J_per_mol
    deltaP_pa = (P_target_bar - P0_bar) * 1e5
    return T0_K + dT_dP_const * deltaP_pa

# Bring DeltaV_base into module scope for the linear approximation
DeltaV_base = Vm_liq - Vm_ice if 'Vm_liq' in globals() else (M_kg_per_mol / rho_liq_kgm3 - M_kg_per_mol / rho_ice_kgm3)

if __name__ == "__main__":
    Tm_K, P_final_bar = estimate_Tm_at_P(P_target_bar=P_target_bar)
    Tm_C = Tm_K - 273.15
    T_lin_K = linear_approx_Tm_K(P_target_bar=P_target_bar)
    T_lin_C = T_lin_K - 273.15
    print("Melting point at {0:.1f} bar: {1:.6f} K, {2:.3f} C".format(P_target_bar, Tm_K, Tm_C))
    print("Linear approx at {0:.1f} bar: {1:.6f} K, {2:.3f} C".format(P_target_bar, T_lin_K, T_lin_C))
    print("Final pressure reached (bar): {0:.3f}".format(P_final_bar))
