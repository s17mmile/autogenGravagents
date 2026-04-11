from dataclasses import dataclass

# Preset constants (no user input)
rho_given = 133.2                 # kg/m^3
M_g_per_mol = 18.02                 # g/mol
M_kg_per_mol = M_g_per_mol / 1000.0 # kg/mol
p = 327.6    # atm
T = 776.4    # K
a = 5.464    # L^2 atm / mol^2
b = 0.03049  # L / mol
R = 0.082057 # L atm / (mol K)

@dataclass
class GasEOSConfig:
    p: float
    T: float
    a: float
    b: float
    R: float = 0.082057


def solve_vm_vdw(p: float, T: float, a: float, b: float, R: float = 0.082057) -> float:
    """Solve van der Waals equation for Vm in L/mol: (p + a/Vm^2)*(Vm - b) = R*T.

    Uses robust bracketing around the ideal gas Vm and a short Newton refinement.
    Returns Vm in L/mol.
    """
    RT = R * T

    def f(Vm: float) -> float:
        return (p + a/(Vm*Vm)) * (Vm - b) - RT

    Vm_ideal = RT / p

    # Try several bracket candidates around Vm_ideal
    candidates = [
        (max(b + 1e-12, Vm_ideal*0.5), max(b + 1e-12, Vm_ideal*0.75)),
        (max(b + 1e-12, Vm_ideal*0.75), max(b + 1e-12, Vm_ideal*1.25)),
        (max(b + 1e-12, Vm_ideal*0.9), max(b + 1e-12, Vm_ideal*2.0)),
        (b + 1e-12, max(b + 1e-12, Vm_ideal*2.5))
    ]
    lo = hi = None
    for c_lo, c_hi in candidates:
        if f(c_lo) * f(c_hi) < 0:
            lo, hi = c_lo, c_hi
            break

    if lo is None:
        lo = max(b + 1e-12, Vm_ideal * 0.5)
        hi = max(0.2, Vm_ideal * 2.0)
        f_lo, f_hi = f(lo), f(hi)
        iter_expand = 0
        while f_lo * f_hi > 0 and iter_expand < 60:
            lo = max(b + 1e-12, lo * 0.9)
            hi = hi * 1.5
            f_lo, f_hi = f(lo), f(hi)
            iter_expand += 1
        if f_lo * f_hi > 0:
            raise RuntimeError("Could not bracket Vm root for van der Waals.")

    # Bisection
    for _ in range(150):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-12 or (hi - lo) < 1e-9 * max(1.0, mid):
            Vm = mid
            break
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    else:
        Vm = 0.5 * (lo + hi)

    # Optional Newton refinement for speed and accuracy
    def f_prime(Vm: float) -> float:
        # f(Vm) = (p + a/Vm^2)*(Vm - b) - R*T
        return p + a/(Vm*Vm) - 2*a*(Vm - b)/(Vm*Vm*Vm)

    Vm_nr = Vm
    for _ in range(6):
        f_val = f(Vm_nr)
        d = f_prime(Vm_nr)
        if d == 0:
            break
        delta = f_val / d
        if abs(delta) < 1e-12:
            break
        Vm_next = Vm_nr - delta
        if Vm_next <= b:
            break
        if abs(f(Vm_next)) < abs(f_val):
            Vm_nr = Vm_next
        else:
            break
    return Vm_nr

# Global EOS root (computed once for given p, T, a, b)
config = GasEOSConfig(p=p, T=T, a=a, b=b, R=R)
Vm_VdW_L = solve_vm_vdw(config.p, config.T, config.a, config.b, config.R)
Vm_VdW_m3 = Vm_VdW_L / 1000.0


def main():
    # 1) Molar volume from density
    Vm_m3_from_density = M_kg_per_mol / rho_given
    Vm_L_from_density = Vm_m3_from_density * 1000.0

    # 2) Molar volume from van der Waals EOS
    Vm_vdw_L = Vm_VdW_L
    Vm_vdw_m3 = Vm_VdW_m3

    # Density consistency check from EOS
    rho_EoS = M_kg_per_mol / Vm_vdw_m3  # kg/m^3
    rho_rel_error = abs(rho_EoS - rho_given) / rho_given

    # Compressibility factor
    Z = p * Vm_vdw_L / (R * T)

    return {
        "Vm_density_L_per_mol": Vm_L_from_density,
        "Vm_density_m3_per_mol": Vm_m3_from_density,
        "Vm_vdw_L_per_mol": Vm_vdW_L,
        "Vm_vdw_m3_per_mol": Vm_vdW_m3,
        "rho_EoS": rho_EoS,
        "rho_given": rho_given,
        "rho_rel_error": rho_rel_error,
        "Z": Z
    }

if __name__ == "__main__":
    results = main()
    print("Vm from density: {:.6f} L/mol, {:.6e} m3/mol".format(
        results["Vm_density_L_per_mol"], results["Vm_density_m3_per_mol"]
    ))
    print("Vm from van der Waals: {:.6f} L/mol, {:.6e} m3/mol".format(
        results["Vm_vdw_L_per_mol"], results["Vm_vdw_m3_per_mol"]
    ))
    print("rho_EoS: {:.1f} kg/m^3, rho_given: {:.1f} kg/m^3, rel_error: {:.2%}".format(
        results["rho_EoS"], results["rho_given"], results["rho_rel_error"]
    ))
    print("Z: {:.6f}".format(results["Z"]))
