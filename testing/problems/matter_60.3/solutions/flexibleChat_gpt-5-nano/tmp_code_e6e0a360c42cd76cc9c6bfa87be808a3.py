import math
import random

# Constants (CODATA-like)
R = 8.314462618          # J mol-1 K-1
NA = 6.02214076e23        # 1/mol
kB = 1.380649e-23          # J K-1
h = 6.62607015e-34           # J s
c = 2.99792458e8              # m s-1
pi = math.pi

# Energy unit helpers
eV_to_J = 1.602176634e-19

def E_cm_inv_to_J(cm_inv):
    return h * c * cm_inv * 100.0

# Subtask helpers

def electronic_S_from_states(T, states):
    # states: list of dicts with keys Delta_E (float), unit ('eV' or 'cm-1'), g (int)
    # Include ground state as Delta_E = 0 by convention if not present yet
    Q_el = 0.0
    dQdT = 0.0
    for st in states:
        Delta_E = st['Delta_E']
        unit = st['unit']
        g = int(st['g'])
        if Delta_E == 0:
            E_J = 0.0
        else:
            if unit == 'eV':
                E_J = Delta_E * eV_to_J
            elif unit == 'cm-1':
                E_J = E_cm_inv_to_J(Delta_E)
            else:
                raise ValueError('Unknown energy unit for electronic state')
        w = math.exp(-E_J / (kB * T))
        Q_el += g * w
        dQdT += g * w * (E_J / (kB * T * T))
    S_elec = R * (math.log(Q_el) + (T * dQdT) / Q_el)
    return S_elec


def vibrational_S(T, w_tilde_cm, anharm_delta_cm=None):
    # Harmonic oscillator baseline
    theta_vib = (h * c * w_tilde_cm) / kB
    x = theta_vib / T
    if anharm_delta_cm is None or anharm_delta_cm <= 0.0:
        expx = math.exp(x)
        return R * ( x / (expx - 1.0) - math.log(1.0 - math.exp(-x)) )
    # Simple anharmonic placeholder: require an ω_e x_e style parameter for realism
    S_ho = vibrational_S(T, w_tilde_cm, None)
    return S_ho * 1.01

# Rotational with nuclear spin statistics for F nuclei (I = 1/2)
# Mass inputs: fluorine atomic mass including electrons; two atoms form F2- with one extra electron
m_F_u = 18.998403163
amu_to_kg = 1.66053906660e-27
m_F = m_F_u * amu_to_kg
m1 = m_F
m2 = m_F
mu = m1 * m2 / (m1 + m2)  # reduced mass
r_e = 1.90e-10
I = mu * (r_e ** 2)
B_J = (h ** 2) / (8.0 * pi ** 2 * I)  # J

def rotational_S_spin(T):
    Q_rot = 0.0
    dQdT = 0.0
    for J in range(0, 2000):
        E = B_J * J * (J + 1)
        w = math.exp(-E / (kB * T))
        gJ = 1 if (J % 2 == 0) else 3
        term = (2 * J + 1) * gJ * w
        Q_rot += term
        dQdT += (2 * J + 1) * gJ * w * (E / (kB * T * T))
        if w < 1e-14:
            if J > 20:
                break
    S_rot = R * (math.log(Q_rot) + (T * dQdT) / Q_rot)
    return S_rot

# High-T rotational reference (consistency diagnostic)

def S_rot_high_T(T):
    theta_rot_K = B_J / kB
    sigma = 2.0
    return R * (math.log(T / (sigma * theta_rot_K)) + 1.0)

# Translational contribution (1 bar standard state)
# Mass including one extra electron for F2-
# Neutral F2 mass (g/mol) = 2 * 18.998403163; add electron mass in g/mol
neutral_F_mass_g_per_mol = 2.0 * 18.998403163
electron_mass_kg = 9.10938356e-31
electron_mass_per_mol_g = (electron_mass_kg * NA) * 1000.0  # g/mol
M_molar_g = neutral_F_mass_g_per_mol + electron_mass_per_mol_g
M_molar = M_molar_g / 1000.0  # kg/mol
P0 = 1e5  # Pa

def translational_S(T, M_molar, P0):
    m = M_molar / NA  # kg per molecule
    q_trans = ((2.0 * pi * m * kB * T) / (h ** 2)) ** (3.0 / 2.0) * (kB * T / P0)
    return R * (math.log(q_trans) + 2.5)

# Master routine: compute S_deg with optional uncertainty budget
def compute_S_deg(T=298.0, P0=1e5, w_tilde_cm=450.0,
                  electronic_state_base=None,
                  do_uncertainty=False, n_samples=200):
    # Electronic: default state list with ground plus two excited states in eV unit
    if electronic_state_base is None:
        electronic_state_base = [
            {'Delta_E': 0.0, 'unit':'eV', 'g': 2},
            {'Delta_E': 1.609, 'unit':'eV', 'g': 2},
            {'Delta_E': 1.702, 'unit':'eV', 'g': 2},
        ]
    S_elec = electronic_S_from_states(T, electronic_state_base)

    S_vib = vibrational_S(T, w_tilde_cm, anharm_delta_cm=None)
    S_rot = rotational_S_spin(T)
    S_trans = translational_S(T, M_molar, P0)
    S_total = S_trans + S_rot + S_vib + S_elec

    result = {
        'S_trans': S_trans,
        'S_rot': S_rot,
        'S_vib': S_vib,
        'S_elec': S_elec,
        'S_total': S_total
    }

    if do_uncertainty:
        S_totals = []
        for _ in range(n_samples):
            # perturb inputs with sensible, positive priors
            DeltaE1_eV = 1.609 * (1.0 + random.gauss(0.0, 0.01))
            DeltaE2_eV = 1.702 * (1.0 + random.gauss(0.0, 0.01))
            w_tilde_s = w_tilde_cm * (1.0 + random.gauss(0.0, 0.02))
            r_e_s = r_e * (1.0 + random.gauss(0.0, 0.01))
            M_molar_s = M_molar * (1.0 + random.gauss(0.0, 0.005))

            # electronic states perturbed
            st_pert = [
                {'Delta_E': 0.0, 'unit':'eV', 'g': 2},
                {'Delta_E': DeltaE1_eV, 'unit':'eV', 'g': max(1, int(round(2 + random.gauss(0.0, 0.5))))},
                {'Delta_E': DeltaE2_eV, 'unit':'eV', 'g': max(1, int(round(2 + random.gauss(0.0, 0.5))))}
            ]
            S_e = electronic_S_from_states(T, st_pert)
            S_v = vibrational_S(T, w_tilde_s, None)

            # rotational perturbed I via r_e
            mF_local = m_F
            mu_local = mF_local / 2.0
            I_local = mu_local * (r_e_s ** 2)
            B_local = (h ** 2) / (8.0 * pi ** 2 * I_local)
            # recompute S_rot with perturbed I
            Qr = 0.0
            dQr = 0.0
            for J in range(0, 1200):
                E = B_local * J * (J + 1)
                wj = math.exp(-E / (kB * T))
                gj = 1 if (J % 2 == 0) else 3
                Qr += (2 * J + 1) * gj * wj
                dQr += (2 * J + 1) * gj * wj * (E / (kB * T * T))
                if wj < 1e-14:
                    break
            S_r = R * (math.log(Qr) + (T * dQr) / Qr)

            # translational perturbed mass
            M_molar_l = M_molar_s
            m_per_mol_local = M_molar_l / NA
            q_tr = ((2 * pi * (m_per_mol_local / NA) * kB * T) / (h ** 2)) ** (3/2) * (kB * T / P0)
            S_t = R * (math.log(q_tr) + 2.5)

            S_tot = S_t + S_r + S_v + S_e
            S_totals.append(S_tot)
        mean = sum(S_totals) / len(S_totals)
        var = sum((x - mean) ** 2 for x in S_totals) / (len(S_totals) - 1)
        result['S_total_mean'] = mean
        result['S_total_std'] = math.sqrt(var)

    return result

# NOMINAL RUN
T = 298.0
S0 = compute_S_deg(T=T, P0=1e5, w_tilde_cm=450.0, do_uncertainty=True, n_samples=200)

print("F2- standard molar entropy S°(298 K) breakdown (J/mol-K):")
print(" S_trans =", S0['S_trans'])
print(" S_rot   =", S0['S_rot'])
print(" S_vib   =", S0['S_vib'])
print(" S_elec  =", S0['S_elec'])
print(" S_total =", S0['S_total'])
if 'S_total_mean' in S0:
    print(" S_total (mean +/- std) = ", S0['S_total_mean']," +/- ", S0['S_total_std'])

# Sanity check: rotation close to high-T limit for the computed B_J
print("High-T rotation check S_rot_high_T =", S_rot_high_T(T))
