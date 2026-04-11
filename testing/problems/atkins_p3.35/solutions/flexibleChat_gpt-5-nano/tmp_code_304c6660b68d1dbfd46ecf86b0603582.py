import numpy as np

# Argon EOS data and constants (all in consistent CGS units)
B = -21.13      # cm3 / mol (virial coefficient)
C = 1054.0      # cm6 / mol^2
P_target = 1.0   # atm
T = 100.0          # K
R = 82.057        # cm3 atm / (mol K)
RT = R * T        # cm3 atm / mol


def solve_vm_gas(P, RT=RT, B=B, C=C):
    """Solve P * Vm^3 - RT * Vm^2 - RT * B * Vm - RT * C = 0 for Vm > 0.
    Returns Vm in cm3/mol corresponding to the vapor (gas) phase, i.e. the largest positive real root."""
    coeffs = [P, -RT, -RT * B, -RT * C]
    roots = np.roots(coeffs)
    real_pos = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
    if not real_pos:
        raise RuntimeError("No positive real Vm root for P = {} atm".format(P))
    Vm = max(real_pos)  # vapor-phase root (largest Vm)
    return Vm


def compute_Z(P, Vm, RT=RT):
    return (P * Vm) / RT  # Z = p Vm / (RT)


def ln_phi_from_p_space(P_target, N=1200, P_min=1e-6, RT=RT, B=B, C=C):
    """Compute ln(phi) by integrating (Z-1)/P over P from P_min to P_target.
    Z(P) is obtained from solving Vm(P) for the gas phase."""
    P_grid = np.linspace(P_min, P_target, N)
    Vm_vals = np.array([solve_vm_gas(P, RT=RT, B=B, C=C) for P in P_grid])
    Z_vals = (P_grid * Vm_vals) / RT
    ln_phi = np.trapz((Z_vals - 1.0) / P_grid, P_grid)
    return ln_phi


def ln_phi_from_vm_space(Vm, RT=RT, B=B, C=C):
    """Compute ln(phi) via Vm-space integral (potentially more stable).
    ln(phi) = - integral from Vm to Vmax of F(V) dV, where
    F(V) = [ (B/V + C/V^2) * dP/dV ] / P(V), with P(V) = RT[1/V + B/V^2 + C/V^3]."""
    V_start = Vm
    V_end = max(Vm * 50.0, 1e5)
    V = np.linspace(V_start * 1.0000001, V_end, 2000)
    P_V = RT * (1.0 / V + B / V**2 + C / V**3)
    dP_dV = RT * (-1.0 / V**2 - 2.0 * B / V**3 - 3.0 * C / V**4)
    F_V = ((B / V + C / V**2) * dP_dV) / P_V
    ln_phi = -np.trapz(F_V, V)
    return ln_phi

# Main calculation for the target state
Vm_at_P = solve_vm_gas(P_target, RT=RT, B=B, C=C)
Z_at_P = compute_Z(P_target, Vm_at_P, RT=RT)
ln_phi_p = ln_phi_from_p_space(P_target, N=1200, P_min=1e-6, RT=RT, B=B, C=C)
phi_p = np.exp(ln_phi_p)
ln_phi_vm = ln_phi_from_vm_space(Vm_at_P, RT=RT, B=B, C=C)
phi_vm = np.exp(ln_phi_vm)
f_p = phi_p * P_target
f_vm = phi_vm * P_target

print("Vm at P = {} atm: {} cm3/mol".format(P_target, Vm_at_P))
print("Z at P = {} atm: {}".format(P_target, Z_at_P))
print("phi (P-space) = {}".format(phi_p))
print("f (P-space) = {} atm".format(f_p))
print("phi (Vm-space) = {}".format(phi_vm))
print("f (Vm-space) = {} atm".format(f_vm))